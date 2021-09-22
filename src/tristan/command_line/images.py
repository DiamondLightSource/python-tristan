"""Aggregate the events from a LATRD Tristan data collection into one or more images."""

import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple, Union

import dask
import h5py
import numpy as np
import pint
import zarr
from dask import array as da
from dask import delayed
from dask.diagnostics import ProgressBar
from dask.distributed import Client, progress
from dask.system import CPU_COUNT
from hdf5plugin import Bitshuffle
from nexgen.nxs_copy import CopyTristanNexus

from .. import clock_frequency
from ..binning import find_start_end, make_images, valid_events
from ..data import (
    aggregate_chunks,
    cue_keys,
    cue_times,
    cues,
    event_location_key,
    event_time_key,
    first_cue_time,
    latrd_data,
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


def determine_image_size(nexus_file: Path) -> Tuple[int, int]:
    """Find the image size from metadata."""
    recommend = f"Please specify the detector dimensions in (x, y) with '--image-size'."
    try:
        with h5py.File(nexus_file, "r") as f:
            return f["entry/instrument/detector/module/data_size"][()]
    except (FileNotFoundError, OSError):
        sys.exit(f"Cannot find NeXus file:\n\t{nexus_file}\n{recommend}")
    except KeyError:
        sys.exit(f"Input NeXus file does not contain image size metadata.\n{recommend}")


def exposure(
    start: int, end: int, exposure_time: pint.Quantity = None, num_images: int = None
) -> (pint.Quantity, int, int):
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

    with latrd_data(raw_files, keys=cue_keys) as data:
        start, end = find_start_end(data)

    print("Binning events into a single image.")
    with latrd_data(raw_files, keys=(event_location_key, event_time_key)) as data:
        image = make_images(valid_events(data, start, end), image_size, (start, end))

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


def save_multiple_images(
    array: da.Array, output_file: Path, write_mode: str = "x"
) -> None:
    """
    Calculate and store a Dask array in an HDF5 file without exceeding available memory.

    Use the Dask distributed scheduler to compute a Dask array and store the
    resulting values to a data set 'data' in the root group of an HDF5 file.  The
    distributed scheduler is capable of managing worker memory better than the
    default scheduler.  In the latter case, the workers can sometimes demand more
    than the available amount of memory.  Using the distributed scheduler avoids this
    problem.

    The distributed scheduler cannot write directly to HDF5 files because h5py.File
    objects are not serialisable.  To work around this issue, the data are first
    stored to a Zarr DirectoryStore, then copied to the final HDF5 file and the Zarr
    store deleted.

    Multithreading is used, as the calculation is assumed to be I/O bound.

    Args:
        array:  A Dask array to be calculated and stored.
        output_file:  Path to the output HDF5 file.
        write_mode:  HDF5 file opening mode.  See :class:`h5py.File`.
    """
    # Set a more generous connection timeout than the default 30s.
    with dask.config.set(
        {
            "distributed.comm.timeouts.connect": "60s",
            "distributed.comm.timeouts.tcp": "60s",
            "distributed.deploy.lost-worker-timeout": "60s",
            "distributed.scheduler.idle-timeout": "600s",
            "distributed.scheduler.locks.lease-timeout": "60s",
        }
    ):
        intermediate = str(output_file.with_suffix(".zarr"))

        # Overwrite any pre-existing Zarr storage.  Don't compute immediately but
        # return the Array object so we can compute it with a progress bar.
        method = {"overwrite": True, "compute": False, "return_stored": True}
        # Prepare to save the calculated images to the intermediate Zarr store.
        array = array.to_zarr(intermediate, component="data", **method)
        # Compute the Array and store the values, using a progress bar.
        progress(array.persist())

    print("\nTransferring the images to the output file.")
    store = zarr.DirectoryStore(intermediate)
    with h5py.File(output_file, write_mode) as f:
        zarr.copy_all(zarr.open(store), f, **Bitshuffle())

    # Delete the Zarr store.
    store.clear()


def save_multiple_image_sequences(
    array: da.Array,
    intermediate_store: Union[Path, str],
    output_files: Iterable[Path],
    write_mode: str = "x",
) -> None:
    intermediate_store = Path(intermediate_store).with_suffix(".zarr")

    # Set a more generous connection timeout than the default 30s.
    with dask.config.set(
        {
            "distributed.comm.timeouts.connect": "60s",
            "distributed.comm.timeouts.tcp": "60s",
            "distributed.deploy.lost-worker-timeout": "60s",
            "distributed.scheduler.idle-timeout": "600s",
            "distributed.scheduler.locks.lease-timeout": "60s",
        }
    ):
        # Overwrite any pre-existing Zarr storage.  Don't compute immediately but
        # return the Array object so we can compute it with a progress bar.
        method = {"overwrite": True, "compute": False, "return_stored": True}
        # Prepare to save the calculated images to the intermediate Zarr store.
        array = [
            sub_array.to_zarr(intermediate_store, component=f"{i:d}/data", **method)
            for i, sub_array in enumerate(array)
        ]
        # Compute the Array and store the values, using a progress bar.
        progress([sub_array.persist() for sub_array in array])
        print()

    print("Transferring the images to the output files.")
    store = zarr.DirectoryStore(str(intermediate_store))
    arrays = zarr.open(store)

    @delayed
    def sequence_to_disk(i, output_file):
        with h5py.File(output_file, write_mode) as f:
            return zarr.copy_all(arrays[i], f, **Bitshuffle())

    transfer = [sequence_to_disk(i, o).persist() for i, o in enumerate(output_files)]
    progress(transfer)
    da.compute(transfer)

    print()

    # Delete the Zarr store.
    store.clear()


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
        start, end = find_start_end(data, distributed=True)
        exposure_time, exposure_cycles, num_images = exposure(
            start, end, args.exposure_time, args.num_images
        )

        if args.align_trigger:
            trigger_type = triggers.get(args.align_trigger)
            print(
                f"Image start and end times will be chosen such that the first "
                f"'{cues[trigger_type]}' is aligned with an image boundary."
            )
            # Note we are assuming that the first trigger time is after shutter open.
            trigger_time = first_cue_time(data, trigger_type)
            if trigger_time is None:
                sys.exit(f"Could not find a '{cues[trigger_type]}' signal.")
            trigger_time = trigger_time.compute().astype(int)
            bins_pre = np.arange(
                trigger_time - exposure_cycles, start, -exposure_cycles, dtype=np.uint64
            )[::-1]
            bins_post = np.arange(trigger_time, end, exposure_cycles, dtype=np.uint64)
            bins = np.concatenate((bins_pre, bins_post))
        else:
            bins = np.linspace(start, end, num_images + 1, dtype=np.uint64)

    print(
        f"Binning events into {num_images} images with an exposure time of "
        f"{exposure_time:~.3g}."
    )

    with latrd_data(raw_files, keys=(event_location_key, event_time_key)) as data:
        images = make_images(valid_events(data, start, end), image_size, bins)
        save_multiple_images(images, output_file, write_mode)

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
            output_nexus = CopyTristanNexus.pump_probe_nexus(
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

    print("Finding trigger signal times.")
    with latrd_data(raw_files, keys=cue_keys) as data:
        trigger_times = cue_times(data, trigger_type)
        progress(trigger_times.persist())
        trigger_times = trigger_times.compute().astype(int)

    print()  # Dask distributed progress bar does not end with a newline, so insert one.

    trigger_times = np.sort(trigger_times)

    if not trigger_times.any():
        sys.exit(f"Could not find a '{cues[trigger_type]}' signal.")

    end = np.diff(trigger_times).min()
    exposure_time, _, num_images = exposure(0, end, args.exposure_time, args.num_images)
    bins = np.linspace(0, end, num_images + 1, dtype=np.uint64)

    print(
        f"Binning events into {num_images} images with an exposure time of "
        f"{exposure_time:~.3g} according to the time elapsed since the most recent "
        f"'{cues[trigger_type]}' signal."
    )

    trigger_times = da.from_array(trigger_times)
    with latrd_data(raw_files, keys=(event_location_key, event_time_key)) as data:
        # Measure the event time as time elapsed since the most recent trigger signal.
        data[event_time_key] = data[event_time_key].astype(np.int64)
        data[event_time_key] -= trigger_times[
            da.digitize(data[event_time_key], trigger_times) - 1
        ]

        images = make_images(valid_events(data, 0, end), image_size, bins)
        save_multiple_images(images, output_file, write_mode)

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
    with latrd_data(raw_files, keys=cue_keys) as data:
        trigger_times = cue_times(data, trigger_type)
        progress(trigger_times.persist())
        trigger_times = trigger_times.compute().astype(int)

    print()  # Dask distributed progress bar does not end with a newline, so insert one.

    trigger_times = np.sort(trigger_times)

    if not trigger_times.any():
        sys.exit(f"Could not find a '{cues[trigger_type]}' signal.")

    intervals_end = np.diff(trigger_times).min()
    interval_time, _, num_intervals = exposure(
        0, intervals_end, args.interval, args.num_sequences
    )
    intervals = np.linspace(0, intervals_end, num_intervals + 1, dtype=np.uint64)

    output_files, out_file_pattern = check_multiple_output_files(
        num_intervals, args.output_file, args.stem, "images", args.force
    )

    with latrd_data(raw_files, keys=cue_keys) as data:
        start, end = find_start_end(data, distributed=True)

    exposure_time, exposure_cycles, num_images = exposure(
        start, end, args.exposure_time, args.num_images
    )
    bins = np.linspace(start, end, num_images + 1, dtype=np.uint64)

    print(
        f"Using '{cues[trigger_type]}' as the pump signal,\n"
        f"binning events into {num_intervals} sequences, corresponding to "
        f"successive pump-probe delay intervals of {interval_time:~.3g}.\n"
        f"Each sequence consists of {num_images} images with an exposure time of "
        f"{exposure_time:~.3g}."
    )

    out_file_stem = out_file_pattern.stem

    n_dig = len(str(num_images))
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
                    f"One or more output files already exist, matching the pattern:\n\t"
                    f"{output_nexus_pattern}\n"
                    "Use '-f' to override, "
                    "or specify a different output file path with '-o'."
                )
    else:
        output_nexus_pattern = None

    trigger_times = da.from_array(trigger_times)
    with latrd_data(raw_files, keys=(event_location_key, event_time_key)) as data:
        data = valid_events(data, start, end)

        # Find the time elapsed since the most recent trigger signal.
        pump_probe_time = data[event_time_key].astype(np.int64)
        pump_probe_time -= trigger_times[
            da.digitize(pump_probe_time, trigger_times) - 1
        ]
        sequence = da.digitize(pump_probe_time, intervals) - 1

        image_sequence_stack = []
        for i in range(num_intervals):
            interval_selection = sequence == i

            event_times = data[event_time_key][interval_selection]
            event_locs = data[event_location_key][interval_selection]
            interval = {
                event_time_key: event_times.compute_chunk_sizes(),
                event_location_key: event_locs.compute_chunk_sizes(),
            }

            size = max(data[event_time_key].itemsize, data[event_location_key].itemsize)
            chunks = aggregate_chunks(*interval[event_time_key].chunks, size)
            interval[event_time_key] = interval[event_time_key].rechunk(chunks)

            image_sequence_stack.append(make_images(interval, image_size, bins))

        save_multiple_image_sequences(
            da.stack(image_sequence_stack), out_file_stem, output_files, write_mode
        )

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
    if args.func == single_image_cli:
        #  Use the default scheduler.
        args.func(args)
    else:  # Multi-image binning requires the Dask distributed scheduler.
        # Use threads, rather than processes.
        with Client(processes=False, threads_per_worker=int(0.9 * CPU_COUNT) or 1):
            args.func(args)
