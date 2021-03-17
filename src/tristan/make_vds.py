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
from itertools import compress
from pathlib import Path
from typing import Dict, Iterable, List

import h5py
import numpy as np

from . import cue_keys, event_keys
from .data import data_files, source_carousel, time_slice_info_from_metadata

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

Sources = Dict[str, List[h5py.VirtualSource]]
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


def virtual_sources(files: List[Path]) -> VirtualSourceInfo:
    """
    Create HDF5 virtual sources and collate ancillary information from raw data files.

    Args:
        files:  Lexicographically sorted list of raw file paths.

    Returns:
        - Dictionary of event data set names and lists of corresponding HDF5 virtual
          sources.  The lists of sources have length and order as per the list of input
          files.
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
    file_slices: List[slice],
    events_per_ts: Iterable,
    event_slices: List[slice],
    time_slices: List[slice],
) -> Layouts:
    """
    Define a virtual data set in the form of virtual layouts linked to virtual sources.

    The time slices of events will appear in the VDS ordered chronologically, then by
    module number to resolve ties.  Cue data will simply be aggregated in
    lexicographical order of the raw files from which they are sourced.

    Args:
        raw_files:      Lexicographically sorted list of raw file paths.
        file_slices:    List of slices to divide the above list of file paths into
                        sub-lists, each corresponding to a different detector module.
                        Ordered by module number.  Length is the number of modules in
                        the detector.
        events_per_ts:  List of the number of events in each time slice, in the order
                        that the time slices will appear in the VDS.  Length is the
                        number of time slices recorded.
        event_slices:   List of slice objects, each representing the position of a time
                        slice in the virtual layouts of events data.  Length and order
                        correspond to 'events_per_ts'.
        time_slices:    List of slice objects used to select each time slice from the
                        virtual source objects in order to populate the virtual
                        layouts.  Length and order correspond to 'events_per_ts'.

    Returns:
        A dictionary of raw data set names and corresponding HDF5 virtual layouts.
    """
    event_sources, cue_sources, num_cues_per_file, dtypes = virtual_sources(raw_files)

    # In order to assemble the event time slices in the desired order (chronological,
    # then by  module), we need to create a carousel of the event data sources.
    for key in event_keys:
        event_sources[key] = source_carousel(event_sources[key], file_slices)

    # Generate the slices used to assign the cue data to their virtual layouts.
    cue_slices = np.pad(np.cumsum(num_cues_per_file), (1, 0))
    cue_slices = list(map(slice, cue_slices[:-1], cue_slices[1:]))

    num_events = event_slices[-1].stop
    num_cues = cue_slices[-1].stop
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
        for cue_slice, source, count in zip(cue_slices, sources, num_cues_per_file):
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
        time_slice_info = time_slice_info_from_metadata(f)

    layouts = virtual_data_set(raw_files, *time_slice_info)

    with h5py.File(output_file, "w" if args.force else "x") as f:
        for layout in layouts.items():
            f.create_virtual_dataset(*layout)

    print(f"Virtual data set file written to\n\t{output_file}")
