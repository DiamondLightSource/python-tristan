"""Tools for extracting data on cues and events from Tristan data files."""

import glob
import re
import sys
from contextlib import ExitStack, contextmanager
from itertools import filterfalse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import h5py
import numpy as np
from dask import array as da

from . import cue_keys, event_keys

# Type alias for collections of raw file paths.
RawFiles = Iterable[Union[str, Path]]

meta_file_name_regex = re.compile(r"(.*)_(?:meta|\d+)")
# Regex for the names of data sets, in the time slice metadata file, representing the
# distribution of time slices across raw data files for each module.
ts_key_regex = re.compile(r"ts_qty_module\d{2}")


def find_input_file_name(in_file):
    """Resolve the input file name."""
    in_file = Path(in_file).expanduser().resolve()

    data_dir = in_file.parent

    # Get the segment 'name_root' from 'name_root_meta.h5' or 'name_root_000001.h5'.
    file_name_root = re.fullmatch(r"(.*)_(?:meta|\d+)", in_file.stem)
    if file_name_root:
        file_name_root = file_name_root[1]
    else:
        sys.exit(
            "Input file name did not have the expected format '<name>_meta.h5':\n"
            f"\t{in_file}"
        )

    return data_dir, file_name_root


def find_file_names(
    in_file: str, out_file: Optional[str], default_out_label: str, force: bool
) -> (Path, str, Path):
    """Resolve the input and output file names."""
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

    if out_file:
        out_file = Path(out_file).expanduser().resolve()
    else:
        out_file = Path(f"{file_name_root}_{default_out_label}.h5")

    if not force and out_file.exists():
        sys.exit(
            f"This output file already exists:\n\t{out_file}\n"
            f"Use '-f' to override or specify a different output file path with '-o'."
        )

    return data_dir, file_name_root, out_file


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


@contextmanager
def latrd_data(
    raw_file_paths: RawFiles, keys: Iterable[str] = cue_keys + event_keys
) -> Dict[str, da.Array]:
    """
    A context manager to read LATRD data sets from multiple files.

    The yielded dictionary has an entry for each of the specified LATRD data keys.
    Each key must be a valid LATRD data key and the corresponding value is a Dask
    array containing the corresponding LATRD data from all the raw data files.

    Args:
        raw_file_paths:  The paths of the raw LATRD data files.
        keys:  The set of LATRD data keys to be read.

    Yields:
        A dictionary of LATRD data keys and arrays of the corresponding data.
    """
    with ExitStack() as stack:
        files = [stack.enter_context(h5py.File(path, "r")) for path in raw_file_paths]

        yield {key: da.concatenate([f[key] for f in files]).rechunk() for key in keys}
