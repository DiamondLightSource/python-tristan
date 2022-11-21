"""Aggregate the events from a LATRD Tristan data collection into one or more images."""

from __future__ import annotations

import argparse
import sys
from contextlib import ExitStack
from functools import partial
from operator import mul
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pint
import zarr
from dask import array as da
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from hdf5plugin import Bitshuffle
from nexgen.nxs_copy import CopyTristanNexus

from .. import blockwise_selection, clock_frequency, compute_with_progress
from ..binning import (
    align_bins,
    create_cache,
    events_to_images,
    find_start_end,
    make_images,
    valid_events,
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
)
from . import (
    check_multiple_output_files,
    check_output_file,
    data_files,
    exposure_parser,
    image_output_parser,
    input_parser,
    interval_parser,
    trigger_parser,
    triggers,
    version_parser,
)


def determine_image_size(nexus_file: Path) -> tuple[int, int]:
    """Find the image size from metadata."""
    recommend = f"Please specify the detector dimensions in (x, y) with '--image-size'."
    try:
        with h5py.File(nexus_file) as f:
            # For the sake of some functions like zarr.create, ensure that the image
            # dimensions are definitely tuple[int, int], not tuple[np.int64,
            # np.int64] or anything else.
            y, x = map(int, f["entry/instrument/detector/module/data_size"][()])
            return y, x
    except (FileNotFoundError, OSError):
        sys.exit(f"Cannot find NeXus file:\n\t{nexus_file}\n{recommend}")
    except KeyError:
        sys.exit(f"Input NeXus file does not contain image size metadata.\n{recommend}")


def exposure(
    start: int, end: int, exposure_time: pint.Quantity = None, num_images: int = None
) -> (pint.Quantity, int, int):
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
            output_nexus = CopyTristanNexus.single_image_nexus(
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

    raw_files, _ = data_files(args.data_dir, args.stem)

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

    raw_files, _ = data_files(args.data_dir, args.stem)

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
            output_nexus = CopyTristanNexus.multiple_images_nexus(
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

        print("Calculating the binned images.")
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
    if input_nexus.exists():
        try:
            # Write output NeXus file if we have an input NeXus file.
            output_nexus = CopyTristanNexus.single_image_nexus(
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

    raw_files, _ = data_files(args.data_dir, args.stem)

    trigger_type = triggers.get(args.trigger_type)

    with ExitStack() as stack:
        # Get the cues data, from which to extract the trigger times.
        cues_data = stack.enter_context(latrd_data(raw_files, keys=cue_keys))

        print("Finding trigger signal times.")
        trigger_times = cue_times(cues_data, trigger_type)
        with ProgressBar():
            # Assumes the trigger times can be held in memory.
            trigger_times = trigger_times.astype(int).compute()

        if not trigger_times.size:
            sys.exit(f"Could not find a '{cues[trigger_type]}' signal.")
        elif not trigger_times.size > 1:
            sys.exit(
                f"Only one '{cues[trigger_type]}' signal found.  Two or more needed."
            )

        end = da.diff(trigger_times).min()
        exposure_time, num_images = args.exposure_time, args.num_images
        exposure_time, _, num_images = exposure(0, end, exposure_time, num_images)
        bins = np.linspace(0, end, num_images + 1, dtype=np.uint64)

        print(
            f"Binning events into {num_images} images with an exposure time of "
            f"{exposure_time:.3g~#P} according to the time elapsed since the most "
            f"recent '{cues[trigger_type]}' signal."
        )

        # Make a cache for the images.
        images = create_cache(output_file, num_images, image_size)

        # Get the events data.
        keys = (event_location_key, event_time_key)
        events_data = stack.enter_context(latrd_data(raw_files, keys=keys))

        # Measure the event time as time elapsed since the most recent trigger signal.
        events_data = events_data.astype({event_time_key: np.int64})
        event_times = events_data[event_time_key].values
        trigger_index = da.digitize(event_times, trigger_times) - 1
        events_data[event_time_key] -= da.take(trigger_times, trigger_index)

        # Bin the events into images.
        events_data = events_to_images(events_data, bins, image_size, images)

        print("Calculating the binned images.")
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
    trigger pulse.  Events are labelled according to which interval they fall in and,
    for each interval in turn, all the events so labelled are aggregated, providing a
    stack of image sequences that captures the evolution of the response of the
    measurement to a pump signal.
    """
    write_mode = "w" if args.force else "x"

    input_nexus = args.data_dir / f"{args.stem}.nxs"
    if not input_nexus.exists():
        print(
            "Could not find a NeXus file containing experiment metadata.\n"
            "Resorting to writing raw image data without accompanying metadata."
        )

    image_size = args.image_size or determine_image_size(input_nexus)

    raw_files, _ = data_files(args.data_dir, args.stem)

    trigger_type = triggers.get(args.trigger_type)

    print("Finding trigger signal times.")

    with ExitStack() as stack:
        data = stack.enter_context(latrd_data(raw_files, keys=cue_keys))

        trigger_times = cue_times(data, trigger_type)
        with ProgressBar():
            trigger_times = trigger_times.astype(int).compute_chunk_sizes()

        if not trigger_times.size:
            sys.exit(f"Could not find a '{cues[trigger_type]}' signal.")

        intervals_end = da.diff(trigger_times).min().compute()

        print("Finding detector shutter open and close times.")
        with ProgressBar():
            start, end = find_start_end(data)

        interval_time, _, num_intervals = exposure(
            0, intervals_end, args.interval, args.num_sequences
        )
        interval_bins = np.linspace(
            0, intervals_end, num_intervals + 1, dtype=np.uint64
        )

        output_files, out_file_pattern = check_multiple_output_files(
            num_intervals, args.output_file, args.stem, "images", args.force
        )

        exposure_time, exposure_cycles, num_images = exposure(
            start, end, args.exposure_time, args.num_images
        )
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
                    CopyTristanNexus.multiple_images_nexus(
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

        data = stack.enter_context(
            latrd_data(raw_files, keys=(event_location_key, event_time_key))
        )
        data = valid_events(data, start, end)

        # Find the time elapsed since the most recent trigger signal.
        pump_probe_time = data[event_time_key].astype(np.int64)
        pump_probe_time -= trigger_times[
            da.searchsorted(trigger_times, pump_probe_time.values, "right") - 1
        ]
        sequence = da.digitize(pump_probe_time.values, interval_bins) - 1

        image_sequence_stack = []
        for i in range(num_intervals):
            select = partial(blockwise_selection, selection=sequence == i)
            interval_data = pd.DataFrame()
            interval_data.event_time_offset = select(data.event_time_offset)
            interval_data.event_id = select(data.event_id)

            image_sequence_stack.append(make_images(interval_data, image_size, bins))

    print("Transferring the images to the output files.")
    # # Multi-threaded copy from Zarr to HDF5.
    # arrays = [da.from_zarr(array) for array in zarr.open(store).values()]
    # with ExitStack() as stack:
    #     files = (stack.enter_context(h5py.File(f, write_mode)) for f in output_files)
    #     dsets = [
    #         f.require_dataset(
    #             "data",
    #             shape=arrays[0].shape,
    #             dtype=arrays[0].dtype,
    #             chunks=arrays[0].chunksize,
    #             **Bitshuffle(),
    #         )
    #         for f in files
    #     ]
    #     with ProgressBar():
    #         da.store(arrays, dsets)
    #
    # # Delete the Zarr store.
    # store.clear()

    print(f"Images written to\n\t{output_nexus_pattern or out_file_pattern}")


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


def main(args=None):
    """Perform the image binning with a user-specified sub-command."""
    args = parser.parse_args(args)
    args.func(args)
