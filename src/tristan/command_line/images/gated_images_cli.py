"""
Bin events into images, gated with trigger signals.

Events will be binned into as many images as there are gate signals, one image per gate.
Each 'gate-open' signal is taken as the start of an exposure and the next 'gate-close'
signal is taken as the end of the exposure.
"""

import sys

import dask
import h5py
import numpy as np
import pandas as pd
import zarr
from dask import array as da
from dask import dataframe as dd
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from hdf5plugin import Bitshuffle
from nexgen.nxs_copy import copy_tristan_nexus

from ... import compute_with_progress
from ...binning import create_cache, find_time_bins, make_images
from ...data import (
    cue_keys,
    cue_times,
    cues,
    event_location_key,
    event_time_dtype,
    event_time_key,
    find_start_end,
    latrd_data,
    pixel_index,
    time_bin_key,
    valid_events,
)
from .. import check_output_file, data_files, triggers
from . import determine_image_size


def main(args):
    """Utility to bin events into a sequence of images according to a gating signal."""
    write_mode = "w" if args.force else "x"
    output_file = check_output_file(args.output_file, args.stem, "images", args.force)

    input_nexus = args.data_dir / f"{args.stem}.nxs"
    if not input_nexus.exists():
        print(
            "Could not find a NeXus file containing experiment metadata.\n"
            "Resorting to writing raw image data without accompanying metadata."
        )

    image_size = args.image_size or determine_image_size(input_nexus)

    raw_files = data_files(args.data_dir, args.stem)

    # If gate_close isn't specified, default to the complementary signal to gate_open.
    gate_open = triggers.get(args.gate_open)
    gate_close = triggers.get(args.gate_close) or gate_open ^ (1 << 5)

    with latrd_data(raw_files, keys=cue_keys) as cues_data:
        print("Finding detector shutter open and close times.")
        with ProgressBar():
            start, end = find_start_end(cues_data)

        print("Finding gate signal times.")
        # Here we assume no synchronization issues:
        # falling edges always recorded after rising edges.
        open_times = cue_times(cues_data, gate_open, after=start)
        close_times = cue_times(cues_data, gate_close, before=end)
        with ProgressBar():
            open_times, close_times = dask.compute(open_times, close_times)

    if not open_times.size:
        sys.exit(f"Could not find a '{cues[gate_open]}' signal.")
    if not close_times.size:
        sys.exit(f"Could not find a '{cues[gate_close]}' signal.")

    open_times = np.sort(open_times)
    close_times = np.sort(close_times)

    if not open_times.size == close_times.size:
        # If size difference is just one, look for missing one right before/after
        # shutters and use shutter open/close timestamp as first/last gate
        if abs(open_times.size - close_times.size) > 1:
            sys.exit(
                "Found a non-matching number of gate open and close signals:\n\t"
                f"Number of '{cues[gate_open]}' signals: {open_times.size}\n\t"
                f"Number of '{cues[gate_close]}' signals: {close_times.size}\n"
                f"Note that signals before the shutter open time are ignored."
            )
        else:
            if open_times[-1] > close_times[-1]:
                print(
                    "WARNING! \n\t"
                    f"Missing last '{cues[gate_close]}' signal.\n\t"
                    f"Shutter close timestamp will be used instead for last image."
                )
                # Append shutter close to close_times
                close_times = np.append(close_times, end)
            elif open_times[0] > close_times[0]:
                print(
                    "WARNING! \n\t"
                    f"Missing first '{cues[gate_open]}' signal.\n\t"
                    f"Shutter open timestamp will be used instead for first image."
                )
                # Insert shutter open to open times
                open_times = np.insert(open_times, 0, start)
            else:
                sys.exit(
                    "Found a non-matching number of gate open and close signals:\n\t"
                    f"Number of '{cues[gate_open]}' signals: {open_times.size}\n\t"
                    f"Number of '{cues[gate_close]}' signals: {close_times.size}\n"
                )

    num_images = open_times.size
    bins = np.linspace(0, num_images, num_images + 1, dtype=event_time_dtype)

    if input_nexus.exists():
        try:
            # Write output NeXus file if we have an input NeXus file.
            output_nexus = copy_tristan_nexus.serial_images_nexus(
                output_file,
                input_nexus,
                nbins=num_images,
                write_mode=write_mode,
            )
        except FileExistsError:
            sys.exit(
                f"This output file already exists:\n\t"
                f"{output_file.with_suffix('.nxs')}\n"
                "Use '-f' to override, "
                "or specify a different output file path with '-o'."
            )
    else:
        output_nexus = None

    print(f"Binning events into {num_images} images.")

    # Make a cache for the images.
    images = create_cache(output_file, num_images, image_size)

    with latrd_data(raw_files, keys=(event_location_key, event_time_key)) as data:
        # Consider only those events that occur between the start and end times.
        data = valid_events(data, start, end)

        # Gate the events.
        event_times = data[event_time_key].astype(np.int64).values
        open_index = da.digitize(event_times, open_times) - 1
        close_index = da.digitize(event_times, close_times)
        # Look for events that happen after gate open and before gate close
        # Eliminate invalid events by looking at the open and close index
        valid = open_index == close_index
        valid = dd.from_dask_array(valid, index=data.index)

        # Convert the event IDs to a form that is suitable for a NumPy bincount.
        data[event_location_key] = pixel_index(data[event_location_key], image_size)

        columns = event_location_key, time_bin_key
        dtypes = data.dtypes
        dtypes[time_bin_key] = dtypes.pop(event_time_key)

        meta = pd.DataFrame(columns=columns).astype(dtype=dtypes)
        # Enumerate the image in the stack to which each event belongs
        data = data.map_partitions(find_time_bins, bins=bins, meta=meta)
        data[time_bin_key] = open_index
        data = data[valid]

        # Bin to images, partition by partition.
        data = dd.map_partitions(
            make_images, data, image_size, images, meta=meta, enforce_metadata=False
        )

        print("Computing the binned images.")
        # Use multi-threading, rather than multi-processing.
        with Client(processes=False):
            compute_with_progress(data)

    print("Transferring the images to the output file.")
    with h5py.File(output_file, write_mode) as f:
        zarr.copy_all(zarr.open(images.store), f, **Bitshuffle())

    # Delete the Zarr store.
    images.store.clear()

    print(f"Images written to\n\t{output_nexus or output_file}")
