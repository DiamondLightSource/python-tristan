"""General utilities for the command-line tools."""

import argparse
import glob
import re
import sys
from itertools import filterfalse
from pathlib import Path
from typing import List, Optional, SupportsFloat, SupportsInt, Tuple, Union

import h5py
import numpy as np
import pint.errors

from .. import __version__, ureg

__all__ = (
    "triggers",
    "check_output_file",
    "check_multiple_output_files",
    "data_files",
    "version_parser",
    "input_parser",
    "image_output_parser",
    "trigger_parser",
    "exposure_parser",
    "interval_parser",
)

from ..data import (
    fem_falling,
    fem_rising,
    lvds_falling,
    lvds_rising,
    ttl_falling,
    ttl_rising,
)

Quantity, Unit = ureg.Quantity, ureg.Unit

# Regex for Tristan data file name stems.
meta_file_name_regex = re.compile(r"(.*)_(?:meta|\d+)")


triggers = {
    "TTL-rising": ttl_rising,
    "TTL-falling": ttl_falling,
    "LVDS-rising": lvds_rising,
    "LVDS-falling": lvds_falling,
    "FEM-rising": fem_rising,
    "FEM-falling": fem_falling,
}


def check_output_file(
    out_file: Optional[Union[Path, str]] = None,
    stem: Optional[str] = None,
    suffix: str = "output",
    force: bool = False,
) -> Optional[Path]:
    """
    Find and check the output file path.

    Find the output file path, given either a specified output file name or a file
    name stem and suffix.  Exit if any output file(s) already exist, unless instructed
    to overwrite.

    Args:
        out_file:  A suggested output file path.
        stem:  A file name stem to use if out_file is not provided.
        suffix:  A suffix to append to stem when constructing the output file name.
        force:  Choose to overwrite any existing files.

    Returns:
        The output file path.

    Raises:
        SystemExit:  if the output file exists and force is not true.
    """
    if out_file or stem:
        out_file = Path(out_file or f"{stem}_{suffix}.h5").expanduser().resolve()

        if not force and out_file.exists():
            sys.exit(
                f"This output file already exists:\n\t{out_file}\n"
                "Use '-f' to override, "
                "or specify a different output file path with '-o'."
            )

        return out_file


def check_multiple_output_files(
    quantity: int,
    out_file: Optional[Union[Path, str]] = None,
    stem: Optional[str] = None,
    suffix: str = "output",
    force: bool = False,
) -> Optional[Tuple[List[Path], Path]]:
    """
    Find and check file paths for output of multiple image sequences.

    Find the output file paths, given either a specified output file name pattern or
    a file name stem and suffix.  Exit if the output file already exists,
    unless instructed to overwrite.

    Args:
        quantity:  THe number of output files to be generated.
        out_file:  A suggested output file path.
        stem:  A file name stem to use if out_file is not provided.
        suffix:  A suffix to append to stem when constructing the output file name.
        force:  Choose to overwrite any existing file of the same name.

    Returns:
        The output file paths.

    Raises:
        SystemExit:  if any output file exists and force is not true.
    """
    if out_file or stem:
        n_dig = len(str(quantity))
        out_file_pattern = (
            Path(out_file or f"{stem}_{suffix}.h5").expanduser().resolve()
        )
        out_files = [
            out_file_pattern.parent / (out_file_pattern.stem + f"_{i + 1:0{n_dig}d}.h5")
            for i in range(quantity)
        ]

        exists = "\n\t".join(map(str, filter(Path.exists, out_files)))
        if not force and exists:
            sys.exit(
                f"The following output files already exist:\n\t{exists}\n"
                "Use '-f' to override, "
                "or specify a different output file path with '-o'."
            )

        return out_files, out_file_pattern


def data_files(data_dir: Path, stem: str, n_dig: int = 6) -> (List[Path], Path):
    """
    Extract information about the files containing raw cues and events data.

    Args:
        data_dir: Directory containing the raw data and time slice metadata HDF5 files.
        stem:     Input file name, stripped of '_meta.h5', '_000001.h5', etc..
        n_dig:    Number of digits in the raw file number, e.g. six in '_000001.h5'.

    Returns:
        - Lexicographically sorted list of raw file paths.
        - File path of the time slice metadata file.
    """
    meta_file = data_dir / f"{stem}_meta.h5"
    if not meta_file.exists():
        sys.exit(f"Could not find the expected detector metadata file:\n\t{meta_file}")

    with h5py.File(meta_file, "r") as f:
        n_files = np.sum(f.get("fp_per_module", default=()))

    if n_files:
        raw_files = [data_dir / f"{stem}_{n + 1:0{n_dig}d}.h5" for n in range(n_files)]
        missing = list(filterfalse(Path.exists, raw_files))
        if missing:
            missing = "\n\t".join(map(str, missing))
            sys.exit(f"The following expected data files are missing:\n\t{missing}")
    else:
        print(
            "The detector metadata hold no information about the number of "
            "expected raw data files.  Falling back on finding the data dynamically."
        )
        search_path = str(data_dir / f"{stem}_{n_dig * '[0-9]'}.h5")
        raw_files = [Path(path_str) for path_str in sorted(glob.glob(search_path))]

    return raw_files, meta_file


# A simple version parser.  Print the version and exit.
version_parser = argparse.ArgumentParser(add_help=False)
version_parser.add_argument(
    "-V",
    "--version",
    action="version",
    version="%(prog)s:  Tristan tools {version}".format(version=__version__),
)


class _InputFileAction(argparse.Action):
    """
    From an input file name argument, find the directory and file name stem.

    Set them as attributes of the argument parser instance.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        data_dir, file_name_stem = self.find_input_file_name(values)
        setattr(namespace, "data_dir", data_dir)
        setattr(namespace, "stem", file_name_stem)

    @staticmethod
    def find_input_file_name(in_file: Union[Path, str]) -> (Path, str):
        """
        Resolve the input file name into a directory and a file name stem.

        The file name stem is the file name stem stripped of the last _(meta|\\d+).

        Args:
            in_file:  The input file path.

        Returns:
            The data directory path and the file name stem.
        """
        in_file = Path(in_file).expanduser().resolve()

        if in_file.is_dir():
            data_dir = in_file

            try:
                (file_name,) = data_dir.glob("*_meta.h5")
            except ValueError:
                sys.exit(
                    "Could not find a single unique '<filename>_meta.h5' file in the "
                    "specified directory.\n"
                    "Please specify the desired input file name instead."
                )
            file_name_stem = meta_file_name_regex.fullmatch(file_name.stem)[1]
        else:
            data_dir = in_file.parent

            # Get the segment 'name_stem' from 'name_stem_meta.h5' or
            # 'name_stem_000001.h5'.
            file_name_stem = meta_file_name_regex.fullmatch(in_file.stem)
            if file_name_stem:
                file_name_stem = file_name_stem[1]
            else:
                sys.exit(
                    "Input file name did not have the expected format "
                    "'<name>_meta.h5':\n"
                    f"\t{in_file}"
                )

        return data_dir, file_name_stem


# A parser with the attributes data_dir and stem,
# representing the input Tristan data files.
input_parser = argparse.ArgumentParser(add_help=False)
input_parser.add_argument(
    "input_file",
    help="Tristan metadata ('_meta.h5') or raw data ('_000001.h5', etc.) file.  "
    "This file must be in the same directory as the HDF5 files containing all the "
    "corresponding raw events data.",
    metavar="input-file",
    action=_InputFileAction,
)


def image_size(size: str) -> (int, int):
    """
    Unpack an image size tuple from a comma-separated string of integers.

    Args:
        size:  A string of comma-separated values, representing the size in pixels of
               the image, in the order (x, y).

    Returns:
        The image size tuple in the order (y, x), for compatibility with row-major data.
    """
    x_size, y_size = map(int, size.strip("()'\"").split(","))
    if x_size < 0 or y_size < 0:
        raise ValueError("Image dimensions must not be negative.")
    if not (x_size and y_size):
        raise ValueError("At least one image dimension must be positive.")
    return y_size, x_size


# A parser to specify
#   - a custom output file path,
#   - whether to allow an existing file to be over-written by the output file,
#   - and what size the output images should have.
image_output_parser = argparse.ArgumentParser(add_help=False)
image_output_parser.add_argument(
    "-o",
    "--output-file",
    help="File name or location for output image file, defaults to the working "
    "directory.  If only a directory location is given, the pattern of the raw data "
    "files will be used.  For multiple-sequence output, a sequence number will be "
    "appended to your choice of output file name.",
)
image_output_parser.add_argument(
    "-f",
    "--force",
    help="Force the output image file to overwrite any existing file with the same "
    "name.",
    action="store_true",
)
image_output_parser.add_argument(
    "-s",
    "--image-size",
    help="Dimensions of the detector in pixels, separated by a comma, as 'x,y', i.e. "
    "'fast,slow'.",
    type=image_size,
)


# A parser to specify a trigger signal type.
trigger_parser = argparse.ArgumentParser(add_help=False)
trigger_parser.add_argument(
    "-t",
    "--trigger-type",
    help="The type of trigger signal used as the pump pulse marker.",
    choices=triggers.keys(),
    required=True,
)


def units_of_time(quantity: Union[Quantity, SupportsFloat, str]) -> Quantity:
    """
    Ensure a quantity of time, has compatible units, defaulting to seconds.

    Args:
        quantity:  Any object that can be interpreted as a Pint quantity, with or
                   without units.

    Returns:
        The initial quantity, with units of seconds applied if it was previously
        dimensionless.

    Raises:
        pint.errors.DimensionalityError:  The specified quantity is not dimensionless
                                          and does not have dimension [time].
    """
    # Catch any UndefinedUnitError (which is a subclass of AttributeError) and
    # re-raise it as a ValueError so that argparse knows that this is a bad argument.
    try:
        quantity = Quantity(quantity)
    except pint.errors.UndefinedUnitError as e:
        raise ValueError(e)

    if quantity <= 0:
        raise ValueError("Time quantity must be positive.")

    quantity = quantity * Unit("s") if quantity.dimensionless else quantity
    if quantity.check("[time]"):
        return quantity
    else:
        raise pint.errors.DimensionalityError(
            quantity,
            "a quantity of",
            quantity.dimensionality,
            pint.Unit("s").dimensionality,
        )


def positive_int(value: SupportsInt) -> int:
    """
    Check that an integer value is positive.

    Args:
        value:  A value that can be cast to an integer.

    Returns:
        'value' as an integer, if it is positive.

    Raises:
        ValueError:  If 'value' does not cast to a positive integer.
    """
    int_value = int(value)
    if not int_value > 0:
        raise ValueError(f"The value {value} does not cast to a positive integer.")
    return int_value


# A parser that determines the desired image exposure time, either from an explicit
# specification, or implicitly from a specified number of images.
exposure_parser = argparse.ArgumentParser(add_help=False)
group = exposure_parser.add_mutually_exclusive_group(required=True)
group.add_argument(
    "-e",
    "--exposure-time",
    help="Duration of each image.  This will be used to calculate the number of "
    "images.  Specify a value with units like '--exposure-time .5ms', '-e 500µs' or "
    "'-e 500us'.  Unspecified units default to seconds.",
    type=units_of_time,
)
group.add_argument("-n", "--num-images", help="Number of images.", type=positive_int)


# A parser for subdividing a regular comb of pump signals into quantised pump-probe
# delay intervals.
interval_parser = argparse.ArgumentParser(add_help=False)
group = interval_parser.add_mutually_exclusive_group(required=True)
group.add_argument(
    "-i",
    "--interval",
    help="Duration of each pump-probe delay interval.  This will be used to calculate "
    "the number of image sequences.  Specify a value with units like "
    "'--exposure-time .5ms', '-e 500µs' or '-e 500us'.  Unspecified units default to "
    "seconds.",
    type=units_of_time,
)
group.add_argument(
    "-x", "--num-sequences", help="Number of image sequences.", type=positive_int
)
