"""Tools for extracting data on cues and events from Tristan data files."""
import re
import sys
from itertools import filterfalse
from pathlib import Path
from typing import List

import h5py
import numpy as np

# Regex for the names of data sets, in the time slice metadata file, representing the
# distribution of time slices across raw data files for each module.
ts_key_regex = re.compile(r"ts_qty_module\d{2}")


def data_files(data_dir: Path, root: str, n_dig: int = 6) -> (List[Path], Path):
    """
    Extract information about the layout of events data on disk, for creating a VDS.

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
        num_files = np.sum(f["fp_per_module"])
    raw_files = [data_dir / f"{root}_{n + 1:0{n_dig}d}.h5" for n in range(num_files)]
    missing_files = list(filterfalse(Path.exists, raw_files))
    if missing_files:
        missing_files = "\n\t".join(map(str, missing_files))
        sys.exit(f"The following expected data files are missing:\n\t{missing_files}")

    return raw_files, meta_file
