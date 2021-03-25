"""
Create an HDF5 virtual data set (VDS) file to aggregate raw Tristan events data.

By default, this file will be saved in the same directory as the raw data and
detector metadata files, retaining the same naming convention.  So if the metadata
file is named 'my_data_1_meta.h5', then the new VDS file will be named
'my_data_1_vds.h5'.
"""

import argparse
import re
import sys
from contextlib import ExitStack
from itertools import chain, compress, cycle, zip_longest
from pathlib import Path
from typing import Dict, Iterable, List

import h5py
import numpy as np

from . import cue_keys, event_keys
from .data import data_files, ts_key_regex

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "input_file",
    help="Tristan metadata ('_meta.h5') or raw data ('_000001.h5', etc.) file.  "
    "This file must be in the same directory as the HDF5 files containing the "
    "corresponding raw events data.",
    metavar="input-file",
)
parser.add_argument(
    "-o",
    "--output-file",
    help="File name for output VDS file.  "
    "By default, the pattern of the input file will be used, with '_meta.h5' "
    "replaced with '_vds.h5', and the VDS will be saved in the same directory "
    "as the input file.",
)
parser.add_argument(
    "-f",
    "--force",
    help="Force the output file to over-write any existing file with the same name.",
    action="store_true",
)

Sources = Dict[str, Iterable[h5py.VirtualSource]]
TimeSliceInfo = List[slice], np.ndarray, List[slice], List[slice]
VirtualSourceInfo = Sources, Sources, List[int], Dict[str, type]
Layouts = Dict[str, h5py.VirtualLayout]


def find_file_names(in_file: str, out_file: str, force: bool) -> (Path, str, Path):
    """Resolve the input and output file names."""
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

    if out_file:
        out_file = Path(out_file).expanduser().resolve()
    else:
        out_file = data_dir / (file_name_root + "_vds.h5")

    if not force and out_file.exists():
        sys.exit(
            f"This output file already exists:\n\t{out_file}\n"
            f"Use '-f' to override or specify a different output file path with '-o'."
        )

    return data_dir, file_name_root, out_file


def time_slice_info(meta_file: h5py.File) -> TimeSliceInfo:
    """
    Assemble information about the event data time slices from the metadata file.

    Args:
        meta_file:  Metadata ('_meta.h5') file.  Assumes metadata version 1.

    Returns:
        - List of slice objects used to select each time slice from the virtual source
          objects in order to populate the virtual layouts.  Length and order
          correspond to 'events_per_ts'.
        - List of the number of events in each time slice, in the order that the time
          slices will appear in the VDS.  Length is the number of time slices recorded.
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

    return time_slices, num_events_per_ts


def virtual_sources(files: List[Path], meta_file: h5py.File) -> VirtualSourceInfo:
    """
    Create HDF5 virtual sources and collate ancillary information from raw data files.

    Args:
        files:      Lexicographically sorted list of raw file paths.
        meta_file:  Tristan detector metadata file object.

    Returns:
        - Dictionary of event data set names and iterators of corresponding HDF5 virtual
          sources.  The iterator of sources for each data set is based on
          itertools.cycle and so repeats indefinitely in the order in which successive
          event slices should be selected to build the virtual data set.
        - Dictionary of cue data set names and lists of corresponding HDF5 virtual
          sources.  The lists of sources have length and order as per the list of input
          files.
        - List of the number of cues in each data file after zero-padding has been
          stripped.  Length and order as per the list of input files.
        - Dictionary of data set names and corresponding data types.
    """
    event_sources = {key: [] for key in event_keys}
    cue_sources = {key: [] for key in cue_keys}
    num_cues_per_file = []

    with ExitStack() as stack:
        raw_files = [stack.enter_context(h5py.File(path, "r")) for path in files]

        dtypes = {key: raw_files[0][key].dtype for key in event_keys + cue_keys}

        for raw_file in raw_files:
            # The cues are padded with zeroes.  Find the first so we can slice them off.
            num_cues_per_file.append(np.argmax(raw_file["cue_id"][()] == 0))
            for key in event_keys:
                event_sources[key].append(h5py.VirtualSource(raw_file[key]))
            for key in cue_keys:
                cue_sources[key].append(h5py.VirtualSource(raw_file[key]))

    # Make a list of slices with which to divide the lexicographically sorted list of
    # file paths into sub-lists, each slice corresponding to a different detector
    # module.  Ordered by module number.  Length is equal to the number of modules in
    # the detector.
    file_slices = np.pad(np.cumsum(meta_file["fp_per_module"]), (1, 0))
    file_slices = list(map(slice, file_slices[:-1], file_slices[1:]))

    # Construct a carousel to select time slices in the order in which they should
    # appear in the virtual layout.
    for key, sources in event_sources.items():
        carousel = zip(*(cycle(sources[file_slice]) for file_slice in file_slices))
        event_sources[key] = chain.from_iterable(carousel)

    return event_sources, cue_sources, num_cues_per_file, dtypes


def virtual_layouts(num_events: int, num_cues: int, dtypes: Dict[str, type]) -> Layouts:
    """Create a dictionary of data set names and corresponding HDF5 virtual layouts."""
    layouts = {}
    for key in event_keys:
        layouts[key] = h5py.VirtualLayout(shape=(num_events,), dtype=dtypes[key])
    for key in cue_keys:
        layouts[key] = h5py.VirtualLayout(shape=(num_cues,), dtype=dtypes[key])
    return layouts


def virtual_data_set(
    raw_files: List[Path],
    meta_file: h5py.File,
    time_slices: List[slice],
    events_per_ts: Iterable[int],
) -> Layouts:
    """
    Define a virtual data set in the form of virtual layouts linked to virtual sources.

    The time slices of events will appear in the VDS ordered chronologically, then by
    module number to resolve ties.  Cue data will simply be aggregated in
    lexicographical order of the raw files from which they are sourced.

    Args:
        raw_files:      Lexicographically sorted list of raw file paths.
        meta_file:      Tristan detector metadata file object.
        time_slices:    List of slice objects used to select each time slice from the
                        virtual source objects in order to populate the virtual
                        layouts.  Length and order correspond to 'events_per_ts'.
        events_per_ts:  List of the number of events in each time slice, in the order
                        that the time slices will appear in the VDS.  Length is the
                        number of time slices recorded.

    Returns:
        A dictionary of raw data set names and corresponding HDF5 virtual layouts.
    """
    event_sources, cue_sources, cues_per_file, dtypes = virtual_sources(
        raw_files, meta_file
    )

    # Generate the slices used to assign the cue data to their virtual layouts.
    cue_slices = np.pad(np.cumsum(cues_per_file), (1, 0))
    num_cues = cue_slices[-1]
    cue_slices = list(map(slice, cue_slices[:-1], cue_slices[1:]))

    # Generate the slices used to assign the event data to their virtual layouts.
    event_slices = np.pad(np.cumsum(events_per_ts), (1, 0))
    num_events = event_slices[-1]
    event_slices = list(map(slice, event_slices[:-1], event_slices[1:]))

    layouts = virtual_layouts(num_events, num_cues, dtypes)

    # Map virtual source slices to virtual layout slices.
    for key, sources in event_sources.items():
        layout = layouts[key]
        # HDF5 VDS doesn't like empty slices, so only assign a source time slice to a
        # layout slice if the corresponding number of events is non-zero. Trying to
        # assign empty slices to empty slices somehow results in a corrupted VDS.
        mapping = compress(zip(event_slices, sources, time_slices), events_per_ts)
        for event_slice, source, time_slice in mapping:
            layout[event_slice] = source[time_slice]
    for key, sources in cue_sources.items():
        layout = layouts[key]
        for cue_slice, source, count in zip(cue_slices, sources, cues_per_file):
            layout[cue_slice] = source[:count]

    return layouts


def main(args=None):
    """Utility for making an HDF5 VDS from raw Tristan data."""
    args = parser.parse_args(args)
    data_dir, root, output_file = find_file_names(
        args.input_file, args.output_file, args.force
    )

    raw_files, meta_file = data_files(data_dir, root)
    with h5py.File(meta_file, "r") as f:
        ts_info = time_slice_info(f)
        layouts = virtual_data_set(raw_files, f, *ts_info)

    with h5py.File(output_file, "w" if args.force else "x") as f:
        for layout in layouts.items():
            f.create_virtual_dataset(*layout)

    print(f"Virtual data set file written to\n\t{output_file}")