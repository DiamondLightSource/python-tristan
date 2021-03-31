"""General utilities for the command-line tools."""

import argparse

import pint

from .. import __version__

__all__ = ("version_parser", "input_parser", "image_output_parser", "exposure_parser")


version_parser = argparse.ArgumentParser(add_help=False)
version_parser.add_argument(
    "-V",
    "--version",
    action="version",
    version="%(prog)s:  Tristan tools {version}".format(version=__version__),
)


input_parser = argparse.ArgumentParser(add_help=False)
input_parser.add_argument(
    "input_file",
    help="Tristan metadata ('_meta.h5') or raw data ('_000001.h5', etc.) file.  "
    "This file must be in the same directory as the HDF5 files containing all the "
    "corresponding raw events data.",
    metavar="input-file",
)


image_output_parser = argparse.ArgumentParser(add_help=False)
image_output_parser.add_argument(
    "-o",
    "--output-file",
    help="File name or location for output image file, defaults to the working "
    "directory.  If only a directory location is given, the pattern of the raw data "
    "files will be used, with '<name>_meta.h5' replaced with '<name>_single_image.h5'.",
)
image_output_parser.add_argument(
    "-f",
    "--force",
    help="Force the output image file to over-write any existing file with the same "
    "name.",
    action="store_true",
)

image_output_parser.add_argument(
    "-s",
    "--image-size",
    help="Dimensions of the detector in pixels, separated by a comma, as 'x,y', i.e. "
    "'fast,slow'.",
)


exposure_parser = argparse.ArgumentParser(add_help=False)
group = exposure_parser.add_mutually_exclusive_group(required=True)
group.add_argument(
    "-e",
    "--exposure-time",
    help="Duration of each image.  This will be used to calculate the number of "
    "images.  Specify a value with units like '--exposure-time .5ms', '-e 500µs' or "
    "'-e 500us'.",
    type=pint.Quantity,
)
group.add_argument("-n", "--num-images", help="Number of images.", type=int)
