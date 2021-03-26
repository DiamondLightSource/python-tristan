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
from dask import array as da
from dask.diagnostics import ProgressBar
from hdf5plugin import Bitshuffle

from . import (
    cue_keys,
    event_location_key,
    event_time_key,
    first_cue_time,
    pixel_index,
    shutter_close,
    shutter_open,
)
from .data import data_files, find_file_names

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
    help="Dimensions of the detector in pixels, separated by a comma, as 'x,y'.",
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
parser_multiple.add_argument("-n", "--num-images", help="Number of images.", type=int)
parser_multiple.add_argument(
    "-o",
    "--output-file",
    help="File name or location for output image file, defaults to working directory.  "
    "If only a directory location is given, the pattern of the raw data files will be "
    "used, with '<name>_meta.h5' replaced with '<name>_images.h5'.",
)
parser_multiple.add_argument(
    "-s",
    "--image-size",
    help="Dimensions of the detector in pixels, separated by a comma, as 'x,y'.",
)
parser_multiple.add_argument(
    "-f",
    "--force",
    help="Force the output image file to over-write any existing file with the same "
    "name.",
    action="store_true",
)


def make_single_image(
    data: Dict[str, da.Array], image_size: Tuple[int, int]
) -> da.Array:
    start_time = first_cue_time(data, shutter_open)
    end_time = first_cue_time(data, shutter_close)
    start_time, end_time = da.compute(start_time, end_time)

    event_times = data[event_time_key]
    event_locations = data[event_location_key]

    valid_events = (start_time <= event_times) & (event_times < end_time)
    event_locations = event_locations[valid_events]
    event_locations = pixel_index(event_locations, image_size)

    image = da.bincount(event_locations, minlength=mul(*image_size))
    return image.astype(np.uint32).reshape(1, *image_size)


def make_multiple_images(
    data: Dict[str, da.Array], num_images: int, image_size: Tuple[int, int]
) -> da.Array:
    start_time = first_cue_time(data, shutter_open)
    end_time = first_cue_time(data, shutter_close)
    start_time, end_time = da.compute(start_time, end_time)

    event_times = data[event_time_key]
    event_locations = data[event_location_key]

    valid_events = (start_time <= event_times) & (event_times < end_time)
    event_times = event_times[valid_events]
    event_locations = event_locations[valid_events]

    bins = da.linspace(start_time, end_time, num_images + 1, dtype=event_times.dtype)
    image_indices = da.digitize(event_times, bins) - 1
    event_locations = pixel_index(event_locations, image_size)

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

        image = make_single_image(data, image_size)

        print("Binning events into a single image.")
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

        images = make_multiple_images(data, args.num_images, image_size)

        print("Binning events into images.")
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
