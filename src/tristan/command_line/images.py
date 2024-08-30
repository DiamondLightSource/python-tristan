"""Aggregate the events from a LATRD Tristan data collection into one or more images."""

from __future__ import annotations

import argparse
import sys
from contextlib import ExitStack
from operator import mul
from pathlib import Path

import dask
import h5py
import numpy as np
import pandas as pd
import pint
import zarr
from dask import array as da
from dask import dataframe as dd
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from hdf5plugin import Bitshuffle
from nexgen.nxs_copy import copy_tristan_nexus

from .. import clock_frequency, compute_with_progress
from ..binning import (
    align_bins,
    create_cache,
    events_to_images,
    find_start_end,
    find_time_bins,
    make_images,
)
from ..data import (
    cue_keys,
    cue_times,
    cues,
    event_location_key,
    event_time_key,
    first_cue_time,
    latrd_data,
    pixel_index,
    seconds,
    valid_events,
)
from . import (
    check_multiple_output_files,
    check_output_file,
    data_files,
    exposure_parser,
    gate_parser,
    image_output_parser,
    input_parser,
    interval_parser,
    trigger_parser,
    triggers,
    version_parser,
)


def determine_image_size(nexus_file: Path) -> tuple[int, int]:
    """Find the image size from metadata."""
    recommend = "Please specify the detector dimensions in (x, y) with '--image-size'."
    try:
        with h5py.File(nexus_file) as f:
            # For the sake of some functions like zarr.create, ensure that the image
            # dimensions are definitely tuple[int, int], not tuple[np.int64,
            # np.int64] or anything else.
            y, x = f["entry/instrument/detector/module/data_size"][()].astype(int)
            return y, x
    except (FileNotFoundError, OSError):
        sys.exit(f"Cannot find NeXus file:\n\t{nexus_file}\n{recommend}")
    except KeyError:
        sys.exit(f"Input NeXus file does not contain image size metadata.\n{recommend}")


def exposure(
    start: int, end: int, exposure_time: pint.Quantity = None, num_images: int = None
) -> tuple(pint.Quantity, int, int):
    """
    Find the exposure time or number of images.

    From a start time and an end time, either derive an exposure time from the
    number of images, or derive the number of images from the exposure time.

    Args:
        start:          Start time in clock cycles.
        end:            End time in clock cycles.
        exposure_time:  Exposure time in any unit of time (optional).
        num_images:     Number of images (optional).

    Returns:
        The exposure time in seconds, the exposure time in clock cycles, the number
        of images.
    """
    if exposure_time:
        exposure_time = exposure_time.to_base_units().to_compact()
        exposure_cycles = (exposure_time * clock_frequency).to_base_units().magnitude
        num_images = int((end - start) // exposure_cycles)
    else:
        # Because they are expected to be mutually exclusive, if there is no
        # exposure_time, there must be a num_images.
        exposure_cycles = int((end - start) / num_images)
        exposure_time = seconds(exposure_cycles)

    return exposure_time, exposure_cycles, num_images


def single_image_cli(args):
    """Utility for making a single image from event-mode data."""
    write_mode = "w" if args.force else "x"
    output_file = check_output_file(
        args.output_file, args.stem, "single_image", args.force
    )
    input_nexus = args.data_dir / f"{args.stem}.nxs"
    if input_nexus.exists():
        try:
            # Write output NeXus file if we have an input NeXus file.
            output_nexus = copy_tristan_nexus.single_image_nexus(
                output_file, input_nexus, write_mode=write_mode
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
        print(
            "Could not find a NeXus file containing experiment metadata.\n"
            "Resorting to writing raw image data without accompanying metadata."
        )

    image_size = args.image_size or determine_image_size(input_nexus)

    raw_files = data_files(args.data_dir, args.stem)

    print("Finding detector shutter open and close times.")
    with latrd_data(raw_files, keys=cue_keys) as data, ProgressBar():
        start, end = find_start_end(data)

    print("Binning events into a single image.")
    with latrd_data(raw_files, keys=(event_location_key, event_time_key)) as data:
        data = valid_events(data, start, end)
        data[event_location_key] = pixel_index(data[event_location_key], image_size)
        image = da.bincount(data[event_location_key], minlength=mul(*image_size))
        image = image.astype(np.uint32).reshape(1, *image_size)

        with ProgressBar(), h5py.File(output_file, write_mode) as f:
            data_set = f.require_dataset(
                "data",
                shape=image.shape,
                dtype=image.dtype,
                chunks=image.chunksize,
                **Bitshuffle(),
            )
            image.store(data_set)

    print(f"Image written to\n\t{output_nexus or output_file}")


def multiple_images_cli(args):
    """
    Utility for making multiple images from event-mode data.

    The time between the start and end of the data collection is subdivided into a
    number of exposures of equal duration, providing a chronological stack of images.
    """
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

    with latrd_data(raw_files, keys=cue_keys) as data:
        print("Finding detector shutter open and close times.")
        with ProgressBar():
            start, end = map(int, find_start_end(data))
        exposure_time, exposure_cycles, num_images = exposure(
            start, end, args.exposure_time, args.num_images
        )

        if args.align_trigger:
            trigger_type = triggers[args.align_trigger]
            print(
                f"Image start and end times will be chosen such that the first "
                f"'{cues[trigger_type]}' after the detector shutter open signal is "
                f"aligned with an image boundary."
            )
            # Note we are assuming that the first trigger time is after shutter open.
            trigger_time = first_cue_time(data, trigger_type, after=start)
            if trigger_time is None:
                sys.exit(
                    f"Could not find a '{cues[trigger_type]}' signal after the "
                    f"detector shutter open signal."
                )
            trigger_time = int(trigger_time.compute())

            if args.exposure_time:
                # Adjust the start time to align a bin edge with the trigger time.
                n_bins_before = (trigger_time - start) // exposure_cycles
                start = trigger_time - n_bins_before * exposure_cycles
                num_images = (end - start) // exposure_cycles
            # It is assumed that start ≤ trigger_time ≤ end.
            else:
                start, exposure_cycles = align_bins(
                    start, trigger_time, end, num_images
                )

        end = start + num_images * exposure_cycles
        bins = np.linspace(start, end, num_images + 1, dtype=np.uint64)

    if input_nexus.exists():
        try:
            # Write output NeXus file if we have an input NeXus file.
            output_nexus = copy_tristan_nexus.multiple_images_nexus(
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

    print(
        f"Binning events into {num_images} images with an exposure time of "
        f"{exposure_time:.3g~#P}."
    )

    # Make a cache for the images.
    images = create_cache(output_file, num_images, image_size)

    with latrd_data(raw_files, keys=(event_location_key, event_time_key)) as data:
        data = events_to_images(data, bins, image_size, images)

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


def pump_probe_cli(args):
    """
    Utility for making multiple images from a pump-probe data collection.

    The time between one pump trigger signal and the next is subdivided into a number
    of exposures of equal duration.  Data from all such pump-to-pump intervals is
    aggregated, providing a single stack of images that captures the evolution of the
    response of the measurement to a pump signal.
    """
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

    trigger_type = triggers.get(args.trigger_type)

    with latrd_data(raw_files, keys=cue_keys) as cues_data:
        print("Finding trigger signal times.")
        trigger_times = cue_times(cues_data, trigger_type)
        with ProgressBar():
            # Assumes the trigger times can be held in memory.
            trigger_times = trigger_times.astype(int).compute()

    if not trigger_times.size:
        sys.exit(f"Could not find a '{cues[trigger_type]}' signal.")
    elif not trigger_times.size > 1:
        sys.exit(f"Only one '{cues[trigger_type]}' signal found.  Two or more needed.")

    end = da.diff(trigger_times).min()
    exposure_time, num_images = args.exposure_time, args.num_images
    exposure_time, _, num_images = exposure(0, end, exposure_time, num_images)
    bins = np.linspace(0, end, num_images + 1, dtype=np.uint64)

    if input_nexus.exists():
        try:
            # Write output NeXus file if we have an input NeXus file.
            output_nexus = copy_tristan_nexus.single_image_nexus(
                output_file,
                input_nexus,
                write_mode=write_mode,
                pump_probe_bins=num_images,
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

    print(
        f"Binning events into {num_images} images with an exposure time of "
        f"{exposure_time:.3g~#P} according to the time elapsed since the most "
        f"recent '{cues[trigger_type]}' signal."
    )

    # Make a cache for the images.
    images = create_cache(output_file, num_images, image_size)

    # Get the events data.
    keys = (event_location_key, event_time_key)
    with latrd_data(raw_files, keys=keys) as events_data:
        # Measure the event time as time elapsed since the most recent trigger signal.
        events_data = events_data.astype({event_time_key: np.int64})
        event_times = events_data[event_time_key].values
        trigger_index = da.digitize(event_times, trigger_times) - 1
        events_data[event_time_key] -= da.take(trigger_times, trigger_index)

        # Bin the events into images.
        events_data = events_to_images(events_data, bins, image_size, images)

        print("Computing the binned images.")
        # Use multi-threading, rather than multi-processing.
        with Client(processes=False):
            compute_with_progress(events_data)

    print("Transferring the images to the output file.")
    with h5py.File(output_file, write_mode) as f:
        zarr.copy_all(zarr.open(images.store), f, **Bitshuffle())

    # Delete the Zarr store.
    images.store.clear()

    print(f"Images written to\n\t{output_nexus or output_file}")


def multiple_sequences_cli(args):
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
    interval_bins = np.linspace(0, intervals_end, num_intervals + 1, dtype=np.uint64)

    output_files, out_file_pattern = check_multiple_output_files(
        num_intervals, args.output_file, args.stem, "images", args.force
    )

    exposure_time, exposure_cycles, num_images = exposure(
        start, end, args.exposure_time, args.num_images
    )
    # Find the bins denoting images within a sequence.
    bins = np.linspace(start, end, num_images + 1, dtype=np.uint64)

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

        columns = event_location_key, "time_bin"
        dtypes = events_data.dtypes
        dtypes["time_bin"] = dtypes.pop(event_time_key)
        meta = pd.DataFrame(columns=columns).astype(dtype=dtypes)
        # Enumerate the image in the stack to which each event belongs.
        events_data = events_data.map_partitions(find_time_bins, bins=bins, meta=meta)
        events_data["time_bin"] += sequence * num_images
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


def gated_images_cli(args):
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
    bins = np.linspace(0, num_images, num_images + 1, dtype=np.uint64)

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

        columns = event_location_key, "time_bin"
        dtypes = data.dtypes
        dtypes["time_bin"] = dtypes.pop(event_time_key)

        meta = pd.DataFrame(columns=columns).astype(dtype=dtypes)
        # Enumerate the image in the stack to which each event belongs
        data = data.map_partitions(find_time_bins, bins=bins, meta=meta)
        data["time_bin"] = open_index
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


parser = argparse.ArgumentParser(description=__doc__, parents=[version_parser])
subparsers = parser.add_subparsers(
    help="Choose the manner in which to create images.",
    required=True,
    dest="sub-command",
)

parser_single = subparsers.add_parser(
    "single",
    aliases=["1"],
    description=(
        "Aggregate all the events from a LATRD Tristan data collection "
        "into a single image."
    ),
    parents=[version_parser, input_parser, image_output_parser],
)
parser_single.set_defaults(func=single_image_cli)

parser_multiple = subparsers.add_parser(
    "multiple",
    aliases=["multi"],
    description=(
        "Bin the events from a LATRD Tristan data collection into multiple images."
    ),
    parents=[version_parser, input_parser, image_output_parser, exposure_parser],
)
parser_multiple.add_argument(
    "-a",
    "--align-trigger",
    help="Align the start and end time of images such that the first trigger signal of "
    "the chosen type is matched up with an image start time.  Useful for examining "
    "effects in the data before and after a single trigger pulse.",
    choices=triggers.keys(),
)
parser_multiple.set_defaults(func=multiple_images_cli)

parser_pump_probe = subparsers.add_parser(
    "pump-probe",
    aliases=["pp"],
    description="Bin events into images representing different pump-probe delays.\n\n"
    "With LATRD data from a pump-probe experiment, where the pump signal has a fairly "
    "constant repeat rate, bin events into a stack of images spanning the range of "
    "pump-probe delay times, from shortest to longest.",
    parents=[
        version_parser,
        input_parser,
        image_output_parser,
        trigger_parser,
        exposure_parser,
    ],
)
parser_pump_probe.set_defaults(func=pump_probe_cli)

parser_multiple_sequences = subparsers.add_parser(
    "sequences",
    aliases=["sweeps"],
    description="Bin events into several sequences of images, each corresponding to "
    "a different pump-probe delay time interval.\n\n"
    "With LATRD data from a pump-probe experiment, where the pump signal has a fairly "
    "constant repeat rate, the recorded events are separated into groups corresponding "
    "to the time elapsed since the most recent pump trigger signal.  Each group is "
    "binned into a sequence of chronological images.  Each sequence is saved to a "
    "separate output file, numbered from the shortest pump-probe delay to the longest.",
    parents=[
        version_parser,
        input_parser,
        image_output_parser,
        trigger_parser,
        exposure_parser,
        interval_parser,
    ],
)
parser_multiple_sequences.set_defaults(func=multiple_sequences_cli)

parser_serial = subparsers.add_parser(
    "serial",
    description="Bin events into images, gated with trigger signals.\n\n"
    "Events will be binned into as many images as there are gate signals, one image "
    "per gate.  Each 'gate-open' signal is taken as the start of an exposure and the "
    "next 'gate-close' signal is taken as the end of the exposure.",
    parents=[version_parser, input_parser, image_output_parser, gate_parser],
)
parser_serial.set_defaults(func=gated_images_cli)


def main(args=None):
    """Perform the image binning with a user-specified sub-command."""
    args = parser.parse_args(args)
    args.func(args)
