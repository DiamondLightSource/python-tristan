"""Tools for extracting data on cues and events from Tristan data files."""
import re
import sys
from itertools import chain, cycle, filterfalse, zip_longest
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

import h5py
import numpy as np

TimeSliceInfo = List[slice], np.ndarray, List[slice], List[slice]

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


def time_slice_info_from_metadata(meta_file: h5py.File) -> TimeSliceInfo:
    """
    Assemble information about the event data time slices from the metadata file.

    Args:
        meta_file:  Metadata ('_meta.h5') file.  Assumes metadata version 1.

    Returns:
        - List of slices with which to divide a lexicographically sorted list of file
          paths into sub-lists, each corresponding to a different detector module.
          Ordered by module number.  Length is the number of modules in the detector.
        - List of the number of events in each time slice, in the order that the time
          slices will appear in the VDS.  Length is the number of time slices recorded.
        - List of slice objects, each representing the position of a time slice in
          the virtual layouts of events data.  Length and order correspond to
          'events_per_ts'.
        - List of slice objects used to select each time slice from the virtual source
          objects in order to populate the virtual layouts.  Length and order
          correspond to 'events_per_ts'.
    """
    fp_per_module = meta_file["fp_per_module"][()]

    ts_keys = sorted(filter(ts_key_regex.match, meta_file.keys()))
    ts_data = [meta_file[ts_key] for ts_key in ts_keys]

    time_slices = []
    num_events_per_ts = []
    # Loop through the modules, acting on the time slice metadata for each in turn.
    for num_files, ts_counts in zip(fp_per_module, ts_data):
        ts_counts = ts_counts[()]
        # Reshape the time slice metadata for a single module into a rectangular array
        # with shape (number of time slices per file, number of files), so as to be
        # able to generate file-specific slices.
        num_ts_per_fp = -(-ts_counts.size // num_files)
        ts_counts.resize(num_ts_per_fp * num_files)
        # Keep a separate record of each module's array of event counts per time slice.
        num_events_per_ts.append(ts_counts)
        ts_counts = ts_counts.reshape(num_ts_per_fp, num_files)
        # Generate the cumulative count of events per time slice for each file.
        ts_per_module = np.pad(np.cumsum(ts_counts, axis=0), ((1, 0), (0, 0)))
        # Turn these counts into slices to select from a virtual source for each file.
        time_slices.append(
            map(slice, ts_per_module[:-1].flatten(), ts_per_module[1:].flatten())
        )

    # Assemble all the source slices into a single list, ordered first
    # chronologically, then by module number.  Where modules have recorded different
    # numbers of time slices, zip_longest will pad with None.
    time_slices = list(chain.from_iterable(zip_longest(*time_slices)))

    # Resize each module's array of event counts per time slice so that their sizes
    # match.  This is achieved by zero-padding to match the None-padding of the list
    # of time slices by zip_longest.
    max_size = max(data.size for data in num_events_per_ts)
    num_events_per_ts = np.column_stack(
        [np.pad(data, (0, max_size - data.size)) for data in num_events_per_ts]
    ).flatten()
    # Compute the cumulative count of events in the order of time slices in the VDS.
    event_slices = np.pad(num_events_per_ts.cumsum(), (1, 0))
    # Use this to generate the slices used to assign to the virtual layout.
    event_slices = list(map(slice, event_slices[:-1], event_slices[1:]))

    # Generate a slice for each module, with which to select the corresponding raw
    # data files from a lexicographically sorted list.
    file_slices = np.pad(np.cumsum(fp_per_module), (1, 0))
    file_slices = list(map(slice, file_slices[:-1], file_slices[1:]))

    return file_slices, num_events_per_ts, event_slices, time_slices


def source_carousel(sources: Sequence, source_slices: Iterable) -> Iterator:
    """
    TODO: Add docstring

    Args:
        sources:
        source_slices:

    Returns:

    """
    carousel = zip(*(cycle(sources[source_slice]) for source_slice in source_slices))
    return chain.from_iterable(carousel)
