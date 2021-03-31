"""Aggregate the events from a LATRD Tristan data collection into one or more images."""

import argparse
import sys
from contextlib import ExitStack

import h5py
import numpy as np
import pint
from dask import array as da
from dask.diagnostics import ProgressBar
from hdf5plugin import Bitshuffle

from .. import (
    clock_frequency,
    cue_keys,
    cues,
    event_location_key,
    event_time_key,
    fem_falling,
    fem_rising,
    lvds_falling,
    lvds_rising,
    ttl_falling,
    ttl_rising,
)
from ..binning import (
    find_start_end,
    first_cue_time,
    make_multiple_images,
    make_single_image,
)
from ..data import data_files, find_file_names
from . import exposure_parser, image_output_parser, input_parser, version_parser

triggers = {
    "TTL-rising": ttl_rising,
    "TTL-falling": ttl_falling,
    "LVDS-rising": lvds_rising,
    "LVDS-falling": lvds_falling,
    "FEM-rising": fem_rising,
    "FEM-falling": fem_falling,
}


def exposure(
    start: int, end: int, exposure_time: pint.Quantity = None, num_images: int = None
):
    freq = pint.Quantity(clock_frequency, "Hz")
    if exposure_time:
        exposure_time = exposure_time.to_base_units().to_compact()
        exposure_cycles = (exposure_time * freq).to_base_units().magnitude
        num_images = int((end - start) // exposure_cycles)
    else:
        # Because they are expected to be mutually exclusive, if there is no
        # exposure_time, there must be a num_images.
        num_images = num_images
        exposure_cycles = (end - start) / num_images
        exposure_time = exposure_cycles / freq
        exposure_time = exposure_time.to_base_units().to_compact()

    return exposure_time, exposure_cycles, num_images


def single_image_cli(args):
    """Utility for making a single image from event-mode data."""
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


def multiple_images_cli(args):
    """Utility for making multiple images from event-mode data."""
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
        exposure_time, exposure_cycles, num_images = exposure(
            start, end, args.exposure_time, args.num_images
        )

        print(
            f"Binning events into {num_images} images with an exposure time of "
            f"{exposure_time:~g}."
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
                sys.exit(f"Could not find a {cues[trigger_type]}.")
            else:
                trigger_time = trigger_time.compute().astype(int)
            bins_pre = np.arange(
                trigger_time - exposure_cycles, start, -exposure_cycles, dtype=np.uint64
            )[::-1]
            bins_post = np.arange(trigger_time, end, exposure_cycles, dtype=np.uint64)
            bins = np.concatenate((bins_pre, bins_post))
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


parser = argparse.ArgumentParser(description=__doc__)
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


def main(args=None):
    """Perform the image binning with a user-specified sub-command."""
    args = parser.parse_args(args)
    args.func(args)
