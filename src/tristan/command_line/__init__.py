"""General utilities for the command-line tools."""

import argparse
import glob
import re
import sys
from itertools import filterfalse
from pathlib import Path
from typing import List, Optional, SupportsFloat, Union

import h5py
import numpy as np

from .. import __version__, ureg

__all__ = (
    "check_output_file",
    "data_files",
    "version_parser",
    "input_parser",
    "image_output_parser",
    "exposure_parser",
)


Quantity, Unit = ureg.Quantity, ureg.Unit

# Regex for Tristan data file name stems.
meta_file_name_regex = re.compile(r"(.*)_(?:meta|\d+)")


def default_unit(quantity: Union[Quantity, SupportsFloat], unit: str = "s") -> Quantity:
    """Provide units for a quantity if it doesn't already have any."""
    quantity = Quantity(quantity)
    return quantity * Unit(unit) if quantity.dimensionless else quantity


def find_input_file_name(in_file: Union[Path, str]) -> (Path, str):
    """
    Resolve the input file name into a directory and a file name root.

    The file name root is the file name stem stripped of the last _(meta|\\d+).

    Args:
        in_file:  The input file path.

    Returns:
        The data directory path and the file name root.
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
        file_name_root = meta_file_name_regex.fullmatch(file_name.stem)[1]
    else:
        data_dir = in_file.parent

        # Get the segment 'name_root' from 'name_root_meta.h5' or 'name_root_000001.h5'.
        file_name_root = meta_file_name_regex.fullmatch(in_file.stem)
        if file_name_root:
            file_name_root = file_name_root[1]
        else:
            sys.exit(
                "Input file name did not have the expected format '<name>_meta.h5':\n"
                f"\t{in_file}"
            )

    return data_dir, file_name_root


def check_output_file(
    out_file: Optional[Union[Path, str]] = None,
    root: Optional[str] = None,
    suffix: str = "output",
    force: bool = False,
) -> Optional[Path]:
    """
    Find and check the output file path.

    Find the output file path, given either a specified output file name or a file
    name root and suffix.  Exit if the output file already exists, unless instructed
    to over-write.

    Args:
        out_file:  A suggested output file path.
        root:  A file name root to use if out_file is not provided.
        suffix:  A suffix to append to root when constructing the output file name.
        force:  Choose to over-write any existing file of the same name.

    Returns:
        The output file path.
    """

    if out_file or root:
        out_file = Path(out_file or f"{root}_{suffix}.h5").expanduser().resolve()

        if not force and out_file.exists():
            sys.exit(
                f"This output file already exists:\n\t{out_file}\n"
                "Use '-f' to override, "
                "or specify a different output file path with '-o'."
            )

        return out_file


def data_files(data_dir: Path, root: str, n_dig: int = 6) -> (List[Path], Path):
    """
    Extract information about the files containing raw cues and events data.

    Args:
        data_dir: Directory containing the raw data and time slice metadata HDF5 files.
        root:     Input file name, stripped of '_meta.h5', '_000001.h5', etc..
        n_dig:    Number of digits in the raw file number, e.g. six in '_000001.h5'.

    Returns:
        - Lexicographically sorted list of raw file paths.
        - File path of the time slice metadata file.
    """
    meta_file = data_dir / f"{root}_meta.h5"
    if not meta_file.exists():
        sys.exit(f"Could not find the expected detector metadata file:\n\t{meta_file}")

    with h5py.File(meta_file, "r") as f:
        n_files = np.sum(f.get("fp_per_module", default=()))

    if n_files:
        raw_files = [data_dir / f"{root}_{n + 1:0{n_dig}d}.h5" for n in range(n_files)]
        missing = list(filterfalse(Path.exists, raw_files))
        if missing:
            missing = "\n\t".join(map(str, missing))
            sys.exit(f"The following expected data files are missing:\n\t{missing}")
    else:
        print(
            "The detector metadata hold no information about the number of "
            "expected raw data files.  Falling back on finding the data dynamically."
        )
        search_path = str(data_dir / f"{root}_{n_dig * '[0-9]'}.h5")
        raw_files = [Path(path_str) for path_str in sorted(glob.glob(search_path))]

    return raw_files, meta_file


class InputFileAction(argparse.Action):
    """
    From an input file name argument, find the directory and file name root.

    Set them as attributes of the argument parser instance.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        data_dir, file_name_root = find_input_file_name(values)
        setattr(namespace, "data_dir", data_dir)
        setattr(namespace, "root", file_name_root)


# A simple version parser.  Print the version and exit.
version_parser = argparse.ArgumentParser(add_help=False)
version_parser.add_argument(
    "-V",
    "--version",
    action="version",
    version="%(prog)s:  Tristan tools {version}".format(version=__version__),
)


# A parser with the attributes data_dir and root,
# representing the input Tristan data files.
input_parser = argparse.ArgumentParser(add_help=False)
input_parser.add_argument(
    "input_file",
    help="Tristan metadata ('_meta.h5') or raw data ('_000001.h5', etc.) file.  "
    "This file must be in the same directory as the HDF5 files containing all the "
    "corresponding raw events data.",
    metavar="input-file",
    action=InputFileAction,
)


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


# A parser that determines the desired image exposure time, either from an explicit
# specification, or implicitly from a specified number of images.
exposure_parser = argparse.ArgumentParser(add_help=False)
group = exposure_parser.add_mutually_exclusive_group(required=True)
group.add_argument(
    "-e",
    "--exposure-time",
    help="Duration of each image.  This will be used to calculate the number of "
    "images.  Specify a value with units like '--exposure-time .5ms', '-e 500µs' or "
    "'-e 500us'.",
    type=default_unit,
)
group.add_argument("-n", "--num-images", help="Number of images.", type=int)
