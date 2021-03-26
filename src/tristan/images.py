"""
Aggregate all the events from a LATRD Tristan data collection into images.
"""

import argparse
import sys
from contextlib import ExitStack
from operator import mul
from typing import Dict, Tuple

import h5py
import numpy as np
import pint
from dask import array as da
from dask.diagnostics import ProgressBar
from hdf5plugin import Bitshuffle

try:
    from numpy.typing import ArrayLike
except ImportError:
    # NumPy versions compatible with Python 3.6 do not have the numpy.typing module.
    ArrayLike = np.ndarray

from . import (
    clock_frequency,
    cue_keys,
    cues,
    event_location_key,
    event_time_key,
    fem_falling,
    fem_rising,
    first_cue_time,
    lvds_falling,
    lvds_rising,
    pixel_index,
    shutter_close,
    shutter_open,
    ttl_falling,
    ttl_rising,
)
from .data import data_files, find_file_names

triggers = {
    "TTL-rising": ttl_rising,
    "TTL-falling": ttl_falling,
    "LVDS-rising": lvds_rising,
    "LVDS-falling": lvds_falling,
    "FEM-rising": fem_rising,
    "FEM-falling": fem_falling,
}

parser_single = argparse.ArgumentParser(
    description=(
        "Aggregate all the events from a LATRD Tristan data collection "
        "into a single image."
    )
)
parser_single.add_argument(
    "input_file",
    help="Tristan raw data file ('.h5') file containing events data or detector "
    "metadata.",
    metavar="input-file",
)
parser_single.add_argument(
    "-o",
    "--output-file",
    help="File name or location for output image file, defaults to working directory.  "
    "If only a directory location is given, the pattern of the raw data files will be "
    "used, with '<name>_meta.h5' replaced with '<name>_single_image.h5'.",
)
parser_single.add_argument(
    "-s",
    "--image-size",
    help="Dimensions of the detector in pixels, separated by a comma, as 'x,y', i.e. "
    "'fast,slow'.",
)
parser_single.add_argument(
    "-f",
    "--force",
    help="Force the output image file to over-write any existing file with the same "
    "name.",
    action="store_true",
)

parser_multiple = argparse.ArgumentParser(
    description=(
        "Aggregate all the events from a LATRD Tristan data collection "
        "into multiple images."
    )
)
parser_multiple.add_argument(
    "input_file",
    help="Tristan raw data file ('.h5') file containing events data or detector "
    "metadata.",
    metavar="input-file",
)
parser_multiple.add_argument(
    "-e",
    "--exposure-time",
    help="Duration of each image.  This will be used to calculate the number of "
    "images, so if --exposure-time is specified, --num-images will be ignored.  "
    "Specify a value with units like '--exposure-time .5ms', '-e 500µs' or '-e 500us'.",
    type=pint.Quantity,
)
parser_multiple.add_argument("-n", "--num-images", help="Number of images.", type=int)
parser_multiple.add_argument(
    "-a",
    "--align-trigger",
    help="Align the start and end time of images such that the first trigger signal "
    "of a given type is matched up with an image start time.  Useful for "
    "examining effects in the data before and after a single trigger pulse.  The "
    "trigger type should be specified with --trigger-type.",
    action="store_true",
)
parser_multiple.add_argument(
    "-t",
    "--trigger-type",
    help="The relevant trigger signal.",
    choices=triggers.keys(),
)
parser_multiple.add_argument(
    "-s",
    "--image-size",
    help="Dimensions of the detector in pixels, separated by a comma, as 'x,y', i.e. "
    "'fast,slow'.",
)
parser_multiple.add_argument(
    "-o",
    "--output-file",
    help="File name or location for output image file, defaults to working directory.  "
    "If only a directory location is given, the pattern of the raw data files will be "
    "used, with '<name>_meta.h5' replaced with '<name>_images.h5'.",
)
parser_multiple.add_argument(
    "-f",
    "--force",
    help="Force the output image file to over-write any existing file with the same "
    "name.",
    action="store_true",
)


def find_start_end(data):
    start_time = first_cue_time(data, shutter_open)
    end_time = first_cue_time(data, shutter_close)
    return da.compute(start_time, end_time)


def make_single_image(
    data: Dict[str, da.Array], image_size: Tuple[int, int], start: int, end: int
) -> da.Array:
    event_times = data[event_time_key]
    event_locations = data[event_location_key]

    valid_events = (start <= event_times) & (event_times < end)
    event_locations = event_locations[valid_events]
    event_locations = pixel_index(event_locations, image_size)

    image = da.bincount(event_locations, minlength=mul(*image_size))
    return image.astype(np.uint32).reshape(1, *image_size)


def make_multiple_images(
    data: Dict[str, da.Array], image_size: Tuple[int, int], bins: ArrayLike
) -> da.Array:
    event_times = data[event_time_key]
    event_locations = data[event_location_key]

    valid_events = (bins[0] <= event_times) & (event_times < bins[-1])
    event_times = event_times[valid_events]
    event_locations = event_locations[valid_events]

    image_indices = da.digitize(event_times, bins) - 1
    event_locations = pixel_index(event_locations, image_size)
    num_images = bins.size - 1

    image_indices = [
        image_indices == image_number for image_number in range(num_images)
    ]
    images = da.stack(
        [
            da.bincount(event_locations[indices], minlength=mul(*image_size))
            for indices in image_indices
        ]
    )

    return images.astype(np.uint32).reshape(num_images, *image_size)


def main_single_image(args=None):
    """Utility for making a single image from event-mode data."""
    args = parser_single.parse_args(args)
    data_dir, root, output_file = find_file_names(
        args.input_file, args.output_file, "single_image", args.force
    )

    if args.image_size:
        image_size = tuple(map(int, args.image_size.split(",")))[::-1]
    else:
        nexus_file = data_dir / f"{root}.nxs"
        try:
            with h5py.File(nexus_file, "r") as f:
                image_size = f["entry/instrument/detector/module/data_size"][()]
        except (FileNotFoundError, OSError):
            sys.exit(
                f"Cannot find NeXus file:\n\t{nexus_file}\nPlease specify the "
                f"detector dimensions in (x, y) with '--image-size'."
            )

    raw_files, _ = data_files(data_dir, root)

    with ExitStack() as stack:
        files = [stack.enter_context(h5py.File(f, "r")) for f in raw_files]
        data = {
            key: da.concatenate([f[key] for f in files]).rechunk()
            for key in (event_location_key, event_time_key) + cue_keys
        }

        print("Binning events into a single image.")
        image = make_single_image(data, image_size, *find_start_end(data))

        with ProgressBar(), h5py.File(output_file, "w" if args.force else "x") as f:
            data_set = f.require_dataset(
                "data",
                shape=image.shape,
                dtype=image.dtype,
                chunks=image.chunksize,
                **Bitshuffle(),
            )
            image.store(data_set)

    print(f"Image file written to\n\t{output_file}")


def main_multiple_images(args=None):
    """Utility for making multiple images from event-mode data."""
    args = parser_multiple.parse_args(args)
    data_dir, root, output_file = find_file_names(
        args.input_file, args.output_file, "images", args.force
    )

    if args.image_size:
        image_size = tuple(map(int, args.image_size.split(",")))[::-1]
    else:
        nexus_file = data_dir / f"{root}.nxs"
        try:
            with h5py.File(nexus_file, "r") as f:
                image_size = f["entry/instrument/detector/module/data_size"][()]
        except (FileNotFoundError, OSError):
            sys.exit(
                f"Cannot find NeXus file:\n\t{nexus_file}\nPlease specify the "
                f"detector dimensions in (x, y) with '--image-size'."
            )

    raw_files, _ = data_files(data_dir, root)

    with ExitStack() as stack:
        files = [stack.enter_context(h5py.File(f, "r")) for f in raw_files]
        data = {
            key: da.concatenate([f[key] for f in files]).rechunk()
            for key in (event_location_key, event_time_key) + cue_keys
        }

        start, end = find_start_end(data)

        freq = pint.Quantity(clock_frequency, "Hz")
        if args.exposure_time:
            exposure_time = args.exposure_time.to_base_units().to_compact()
            exposure_cycles = (exposure_time * freq).to_base_units().magnitude
            num_images = int((end - start) // exposure_cycles)
        elif args.num_images:
            num_images = args.num_images
            exposure_cycles = (end - start) / num_images
            exposure_time = exposure_cycles / freq
            exposure_time = exposure_time.to_base_units().to_compact()
        else:
            sys.exit("Please specify either --exposure-time or --num-images.")

        print(
            f"Binning events into {num_images} images with an exposure time of "
            f"{exposure_time:~g}."
        )

        trigger_type = triggers.get(args.trigger_type)
        if args.align_trigger and trigger_type:
            print(
                f"Image start and end times will be chosen such that the first "
                f"{cues[trigger_type]} is aligned with an image boundary."
            )
            # Note we are assuming that the first trigger time is after shutter open.
            trigger_time = first_cue_time(data, trigger_type)
            if trigger_time is None:
                sys.exit(f"Could not find a {cues[trigger_type]}.")
            else:
                trigger_time = trigger_time.compute().astype(int)
            bins_pre = np.arange(
                trigger_time - exposure_cycles, start, -exposure_cycles, dtype=np.uint64
            )[::-1]
            bins_post = np.arange(trigger_time, end, exposure_cycles, dtype=np.uint64)
            bins = np.concatenate((bins_pre, bins_post))
        elif args.align_trigger or trigger_type:
            sys.exit(
                "To align images with the first trigger time of a given type, "
                "please specify both --align-trigger and --trigger-type."
            )
        else:
            bins = np.linspace(start, end, num_images + 1, dtype=np.uint64)

        images = make_multiple_images(data, image_size, bins)

        with ProgressBar(), h5py.File(output_file, "w" if args.force else "x") as f:
            data_set = f.require_dataset(
                "data",
                shape=images.shape,
                dtype=images.dtype,
                chunks=images.chunksize,
                **Bitshuffle(),
            )
            images.store(data_set)

    print(f"Images file written to\n\t{output_file}")
