#!/usr/bin/env python

"""Apply a flat-field correction to reconstructed Tristan images."""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
from operator import mul, truediv

import h5py
import hdf5plugin
import numpy as np
from dask import array as da
from dask.diagnostics import ProgressBar

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "input_file",
    help="A file of reconstructed Tristan images (.h5 or .nxs), containing an image "
    "data set '/data'.",
    metavar="input-file",
    type=pathlib.Path,
)
parser.add_argument(
    "flat_field_file",
    help="A Tristan flat-field correction (.h5 or .nxs) file, containing the "
    "flat-field measurement in a data set '/image'.",
    metavar="flat-field-file",
    type=pathlib.Path,
)
choices = ["multiply", "divide"]
parser.add_argument(
    "method",
    help="Choose whether to multiply or divide the images by the flat-field "
    "correction data set.  Default is 'divide'.",
    choices=choices,
    default="multiply",
)
parser.add_argument(
    "-o",
    "--output-file",
    help="Path for the output file (.h5 or .nxs).  If not provided, this will default "
    "to the input filename with the added suffix '_flat_field_multiplied' or "
    "'_flat_field_divided'.",
    type=pathlib.Path,
)
parser.add_argument(
    "-f",
    "--force",
    help="Force the output image file to overwrite any existing file with the same "
    "name.",
    action="store_true",
)


file_exists = (
    "Output file already exists:\n\t"
    "{}\n"
    "Use '-f' to override, or specify a different output file path with '-o'."
)


def main(args: list[str] | None = None) -> None:
    """
    Apply the flat field data set to an images data set.

    Args:
        args:  Input command line arguments.  If None, defaults to sys.argv[1:].
    """
    args = parser.parse_args(args)

    method = dict(zip(choices, ["multiplied", "divided"]))[args.method]
    output_file = args.output_file or args.input_file.stem + f"_flat_field_{method}.h5"
    output_file = pathlib.Path(output_file).with_suffix(".h5")
    write_mode = "w" if args.force else "x"

    try:
        with h5py.File(args.input_file.with_suffix(".h5")) as f, h5py.File(
            args.flat_field_file
        ) as g, h5py.File(output_file, write_mode) as h, ProgressBar():
            images = da.from_array(f["data"])
            flat_field = g["image"]
            # Multiply or divide the images by the flat-field correction.
            func = dict(zip(choices, [mul, truediv]))[args.method]
            images = func(images, np.where(flat_field, flat_field, 1))
            images = images.astype(np.uint32)
            h.require_dataset(
                "data",
                shape=images.shape,
                dtype=images.dtype,
                chunks=images.chunksize,
                **hdf5plugin.Bitshuffle(),
            )
            images.store(h["data"])
    except FileExistsError:
        sys.exit(file_exists.format(output_file))

    output_nexus = output_file.with_suffix(".nxs")
    if output_nexus.exists() and not args.force:
        sys.exit(file_exists.format(output_nexus))
    else:
        try:
            shutil.copy(args.input_file.with_suffix(".nxs"), output_nexus)
            with h5py.File(output_nexus, "r+") as f:
                del f["entry/data/data"]
                f["entry/data/data"] = h5py.ExternalLink(str(output_file), "data")
                f["entry/instrument/detector/flatfield_applied"][()] = "TRUE"
        except FileNotFoundError:
            sys.exit("Could not find input NeXus file to copy.")
