"""Aggregate the events from a LATRD Tristan data collection into one or more images."""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import dask
import h5py
import numpy as np
import pint
import zarr
from dask import array as da
from dask.diagnostics import ProgressBar
from dask.distributed import Client, progress
from dask.system import CPU_COUNT
from hdf5plugin import Bitshuffle
from nexgen.nxs_copy import CopyTristanNexus

from .. import clock_frequency
from ..binning import find_start_end, make_images
from ..data import (
    cue_keys,
    cue_times,
    cues,
    event_location_key,
    event_time_key,
    fem_falling,
    fem_rising,
    first_cue_time,
    latrd_data,
    lvds_falling,
    lvds_rising,
    seconds,
    ttl_falling,
    ttl_rising,
)
from . import (
    check_output_file,
    data_files,
    exposure_parser,
    image_output_parser,
    input_parser,
    version_parser,
)

triggers = {
    "TTL-rising": ttl_rising,
    "TTL-falling": ttl_falling,
    "LVDS-rising": lvds_rising,
    "LVDS-falling": lvds_falling,
    "FEM-rising": fem_rising,
    "FEM-falling": fem_falling,
}


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
        # Write output NeXus file if we have an input NeXus file.
        output_nexus = CopyTristanNexus.single_image_nexus(
            output_file, input_nexus, write_mode=write_mode
        )
    else:
        output_nexus = None
        print(
            "Could not find a NeXus file containing experiment metadata.\n"
            "Resorting to writing raw image data without accompanying metadata."
        )

    image_size = args.image_size or determine_image_size(input_nexus)

    raw_files, _ = data_files(args.data_dir, args.stem)

    keys = (event_location_key, event_time_key) + cue_keys
    with latrd_data(raw_files, keys=keys) as data:
        print("Binning events into a single image.")
        image = make_images(data, image_size, find_start_end(data))

        with ProgressBar(), h5py.File(output_file, write_mode) as f:
            data_set = f.require_dataset(
                "data",
                shape=image.shape,
                dtype=image.dtype,
                chunks=image.chunksize,
                **Bitshuffle(),
            )
            image.store(data_set)

    print(f"Images written to\n\t{output_nexus or output_file}")


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
    dask.config.set(
        {
            "distributed.comm.timeouts.connect": "60s",
            "distributed.comm.timeouts.tcp": "60s",
            "distributed.deploy.lost-worker-timeout": "60s",
            "distributed.scheduler.idle-timeout": "600s",
            "distributed.scheduler.locks.lease-timeout": "60s",
        }
    )

    intermediate = str(output_file.parent / f"{output_file.stem}.zarr")

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

        print(
            f"Binning events into {num_images} images with an exposure time of "
            f"{exposure_time:~.3g}."
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

    with latrd_data(raw_files, keys=(event_location_key, event_time_key)) as data:
        images = make_images(data, image_size, bins)
        save_multiple_images(images, output_file, write_mode)

    if input_nexus.exists():
        # Write output NeXus file if we have an input NeXus file.
        output_nexus = CopyTristanNexus.multiple_images_nexus(
            output_file,
            input_nexus,
            nbins=num_images,
            write_mode=write_mode,
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
        # Write output NeXus file if we have an input NeXus file.
        output_nexus = CopyTristanNexus.pump_probe_nexus(
            output_file, input_nexus, write_mode=write_mode
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
        trigger_type = triggers.get(args.trigger_type)

        trigger_times = cue_times(data, trigger_type).compute().astype(int)
        trigger_times = np.sort(np.unique(trigger_times))
        end = np.diff(trigger_times).min()

        if not np.any(trigger_times):
            sys.exit(f"Could not find a '{cues[trigger_type]}' signal.")

        exposure_time, exposure_cycles, num_images = exposure(
            0, end, args.exposure_time, args.num_images
        )

        print(
            f"Binning events into {num_images} images with an exposure time of "
            f"{exposure_time:~.3g} according to the time elapsed since the mose recent "
            f"'{cues[trigger_type]}' signal."
        )

    with latrd_data(raw_files, keys=(event_location_key, event_time_key)) as data:
        # Measure the event time as time elapsed since the most recent trigger signal.
        trigger_times = da.from_array(trigger_times)
        data[event_time_key] = data[event_time_key].astype(np.int64)
        data[event_time_key] -= trigger_times[
            da.digitize(data[event_time_key], trigger_times) - 1
        ]

        bins = np.linspace(0, end, num_images + 1, dtype=np.uint64)

        images = make_images(data, image_size, bins)
        save_multiple_images(images, output_file, write_mode)

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
    "constant repeat rate, bin events into images spanning a range of pump-probe "
    "delay times.",
    parents=[version_parser, input_parser, image_output_parser, exposure_parser],
)
parser_pump_probe.add_argument(
    "-t",
    "--trigger-type",
    help="The type of trigger signal used as the pump pulse marker.",
    choices=triggers.keys(),
    required=True,
)
parser_pump_probe.set_defaults(func=pump_probe_cli)


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
