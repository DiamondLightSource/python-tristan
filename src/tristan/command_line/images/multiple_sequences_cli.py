import sys
from contextlib import ExitStack

import dask
import h5py
import numpy as np
import pandas as pd
from dask import array as da
from dask import dataframe as dd
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from hdf5plugin import Bitshuffle
from nexgen.nxs_copy import copy_tristan_nexus

from ... import compute_with_progress
from ...binning import create_cache, find_start_end, find_time_bins, make_images
from ...data import (
    cue_keys,
    cue_times,
    cues,
    event_location_key,
    event_time_dtype,
    event_time_key,
    latrd_data,
    pixel_index,
    time_bin_key,
    valid_events,
)
from .. import check_multiple_output_files, data_files, triggers
from . import determine_image_size, exposure


def main(args):
    """
    Utility for making multiple image sequences from a pump-probe data collection.

    The time between one pump trigger signal and the next is subdivided into a number
    of intervals of equal duration, quantising the time elapsed since the most recent
    trigger pulse.  Events are labelled according to the interval into which they fall
    and, for each interval in turn, all the events so labelled are binned into a
    sequence of images, providing a stack of image sequences that captures the
    evolution of the response of the measurement to a pump signal.
    """
    write_mode = "w" if args.force else "x"

    input_nexus = args.data_dir / f"{args.stem}.nxs"
    if not input_nexus.exists():
        print(
            "Could not find a NeXus file containing experiment metadata.\n"
            "Resorting to writing raw image data without accompanying metadata."
        )

    image_size = args.image_size or determine_image_size(input_nexus)

    raw_files = data_files(args.data_dir, args.stem)

    trigger_type = triggers.get(args.trigger_type)

    print("Finding trigger signal times.")

    with latrd_data(raw_files, keys=cue_keys) as cues_data:
        trigger_times = cue_times(cues_data, trigger_type)
        with ProgressBar():
            trigger_times = trigger_times.astype(int).compute()

        if not trigger_times.size:
            sys.exit(f"Could not find a '{cues[trigger_type]}' signal.")
        elif not trigger_times.size > 1:
            sys.exit(
                f"Only one '{cues[trigger_type]}' signal found.  Two or more needed."
            )

        print("Finding detector shutter open and close times.")
        with ProgressBar():
            start, end = find_start_end(cues_data)

    intervals_end = da.diff(trigger_times).min()
    interval_time, _, num_intervals = exposure(
        0, intervals_end, args.interval, args.num_sequences
    )
    # Find the bins denoting to which image sequence each event belongs.
    interval_bins = np.linspace(
        0, intervals_end, num_intervals + 1, dtype=event_time_dtype
    )

    output_files, out_file_pattern = check_multiple_output_files(
        num_intervals, args.output_file, args.stem, "images", args.force
    )

    exposure_time, exposure_cycles, num_images = exposure(
        start, end, args.exposure_time, args.num_images
    )
    # Find the bins denoting images within a sequence.
    bins = np.linspace(start, end, num_images + 1, dtype=event_time_dtype)

    print(
        f"Using '{cues[trigger_type]}' as the pump signal,\n"
        f"binning events into {num_intervals} sequences, corresponding to "
        f"successive pump-probe delay intervals of {interval_time:.3g~#P}.\n"
        f"Each sequence consists of {num_images} images with an effective exposure "
        f"time of {exposure_time / num_intervals:.3g~#P}."
    )

    out_file_stem = out_file_pattern.stem

    n_dig = len(str(num_intervals))
    out_file_pattern = out_file_pattern.parent / f"{out_file_stem}_{'#' * n_dig}.h5"

    if input_nexus.exists():
        # Write output NeXus files if we have an input NeXus file.
        output_nexus_pattern = out_file_pattern.with_suffix(".nxs")
        for output_file in output_files:
            try:
                copy_tristan_nexus.multiple_images_nexus(
                    output_file,
                    input_nexus,
                    nbins=num_images,
                    write_mode=write_mode,
                )
            except FileExistsError:
                sys.exit(
                    f"One or more output files already exist, "
                    f"matching the pattern:\n\t"
                    f"{output_nexus_pattern}\n"
                    "Use '-f' to override, "
                    "or specify a different output file path with '-o'."
                )
    else:
        output_nexus_pattern = None

    # Make a cache for the images.
    images = create_cache(out_file_pattern, num_intervals * num_images, image_size)

    # Get the events data.
    events_keys = (event_location_key, event_time_key)
    with latrd_data(raw_files, keys=events_keys) as events_data:
        events_data = valid_events(events_data, start, end)

        # Find the time elapsed since the most recent trigger signal.
        event_time = events_data[event_time_key].astype(np.int64).values
        trigger_index = da.digitize(event_time, trigger_times) - 1
        pump_probe_time = event_time - da.take(trigger_times, trigger_index)
        # Enumerate the sequence to which each event belongs.
        sequence = da.digitize(pump_probe_time, interval_bins) - 1
        # Eliminate invalid sequence numbers (negative, or ≥ num_intervals).
        valid = (0 <= sequence) & (sequence < num_intervals)
        valid = dd.from_dask_array(valid, index=events_data.index)

        # Convert the event IDs to a form that is suitable for a NumPy bincount.
        events_data[event_location_key] = pixel_index(
            events_data[event_location_key], image_size
        )

        columns = event_location_key, time_bin_key
        dtypes = events_data.dtypes
        dtypes[time_bin_key] = dtypes.pop(event_time_key)
        meta = pd.DataFrame(columns=columns).astype(dtype=dtypes)
        # Enumerate the image in the stack to which each event belongs.
        events_data = events_data.map_partitions(find_time_bins, bins=bins, meta=meta)
        events_data[time_bin_key] += sequence * num_images
        events_data = events_data[valid]

        # Bin to images, partition by partition.
        events_data = dd.map_partitions(
            make_images,
            events_data,
            image_size,
            images,
            meta=meta,
            enforce_metadata=False,
        )
        print("Computing the binned images.")
        # Use multi-threading, rather than multi-processing.
        with Client(processes=False):
            compute_with_progress(events_data)

    print("Transferring the images to the output files.")
    store = images.store
    images = da.from_zarr(images)
    stack_shape = num_intervals, num_images, *image_size
    # Silence a large chunks warning, since we immediately rechunk to one-image chunks.
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        images = images.reshape(stack_shape).rechunk((1, 1, *image_size))
    images = list(images)

    # Multi-threaded copy from Zarr to HDF5.
    with ExitStack() as stack:
        files = (stack.enter_context(h5py.File(f, write_mode)) for f in output_files)
        dsets = [
            f.require_dataset(
                "data",
                shape=images[0].shape,
                dtype=images[0].dtype,
                chunks=images[0].chunksize,
                **Bitshuffle(),
            )
            for f in files
        ]
        with ProgressBar():
            da.store(images, dsets)

    # Delete the Zarr store.
    store.clear()

    print(f"Images written to\n\t{output_nexus_pattern or out_file_pattern}")
