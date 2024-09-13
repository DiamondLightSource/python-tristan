"""
Bin events into images, gated with trigger signals.

Events will be binned into as many images as there are gate signals, one image per gate.
Each 'gate-open' signal is taken as the start of an exposure and the next 'gate-close'
signal is taken as the end of the exposure.
"""

from __future__ import annotations

# Some platforms seem to raise OSError: [Errno -101] NetCDF: HDF error:
# '/path/to/HDF5/file.h5'
# Importing netCDF4 before h5py here seems to fix it.  ¯\_(ツ)_/¯
import netCDF4  # noqa F401

import sys

import h5py
import numpy as np
import sparse
import zarr
from hdf5plugin import Bitshuffle
from nexgen.nxs_copy import copy_tristan_nexus

from ... import WithLocalDistributedCluster, compute_with_progress
from ...binning import event_block_to_image_cache, find_preceding_bin_edge_index
from ...data import (
    cue_keys,
    cue_times,
    cues,
    event_location_key,
    event_time_dtype,
    event_time_key,
    find_start_end,
    image_dtype,
    latrd_mf_data,
    pixel_index,
    time_bin_key,
    valid_events,
)
from ...storage import create_cache
from .. import check_output_file, data_files, triggers
from . import determine_image_size


@WithLocalDistributedCluster()
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

    cues_data = latrd_mf_data(raw_files, keys=cue_keys)
    start, end = find_start_end(cues_data)

    print("Finding gate signal times.")
    # Here we assume no synchronization issues:
    # falling edges always recorded after rising edges.
    open_times = cue_times(cues_data, gate_open, after=start)
    close_times = cue_times(cues_data, gate_close, before=end)
    num_images = open_times.size

    if not open_times.size:
        sys.exit(f"Could not find a '{cues[gate_open]}' signal.")
    if not close_times.size:
        sys.exit(f"Could not find a '{cues[gate_close]}' signal.")

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

    if input_nexus.exists():
        try:
            # Write output NeXus file if we have an input NeXus file.
            output_nexus = copy_tristan_nexus.serial_images_nexus(
                output_file,
                input_nexus,
                nbins=num_images,
                write_mode=write_mode,
            )
            pass
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
    shape = num_images, *image_size
    cache = create_cache(output_file, shape)

    events_data = latrd_mf_data(raw_files, keys=(event_time_key, event_location_key))
    # Consider only those events that occur between the start and end times.
    events_data = valid_events(events_data, start, end)

    # Gate the events.
    event_times = events_data[event_time_key].astype(event_time_dtype)
    meta = (event_time_key, event_time_dtype)
    open_index = event_times.map_partitions(
        find_preceding_bin_edge_index, bins=open_times, meta=meta
    )
    close_index = event_times.map_partitions(np.digitize, bins=close_times, meta=meta)
    events_data = events_data.rename(columns={event_time_key: time_bin_key})
    events_data[time_bin_key] = open_index

    # Look for events that happen after gate open and before gate close
    # Eliminate invalid events by looking at the open and close index
    valid = open_index == close_index
    events_data = events_data[valid]

    # Convert the event IDs to indices of pixels in the flattened array.
    events_data = pixel_index(events_data, image_size)

    # Dummy metadata for dask.array.map_blocks.
    empty_coords = np.empty(shape=(len(shape), 0), dtype=int)
    empty_coo = sparse.COO(empty_coords, data=image_dtype(()), shape=shape)

    # Bin to images, partition by partition.
    coords = events_data.values.T
    events_data = coords.map_blocks(
        event_block_to_image_cache, shape=shape, cache=cache, meta=empty_coo
    )

    print("Computing the binned images.")
    compute_with_progress(events_data)

    print("Transferring the images to the output file.")
    with h5py.File(output_file, write_mode) as f:
        zarr.copy_all(zarr.open(cache.store), f, **Bitshuffle())

    print(f"Images written to\n\t{output_nexus or output_file}")
