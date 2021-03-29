"""Summarise the cue messages in Tristan data."""

import argparse
import re
import sys
from contextlib import ExitStack
from pathlib import Path

import h5py
import numpy as np
from dask import array as da

from . import cue_id_key, cue_keys, cue_time_key, cues, reserved, seconds
from .data import data_files

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "input_file",
    help="Tristan raw data file ('.h5') file containing events data or detector "
    "metadata.",
    metavar="input-file",
)


def find_file_name(in_file):
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


def main(args=None):
    args = parser.parse_args(args)

    data_dir, root = find_file_name(args.input_file)
    raw_files, _ = data_files(data_dir, root)

    with ExitStack() as stack:
        files = [stack.enter_context(h5py.File(f, "r")) for f in raw_files]
        data = {
            key: da.concatenate([f[key] for f in files]).rechunk() for key in cue_keys
        }

        relevant = (data[cue_id_key] > 0) & (data[cue_id_key] != reserved)
        cue_ids = data[cue_id_key][relevant].compute()
        cue_times = data[cue_time_key][relevant].compute()

    unique_cues = np.sort(np.unique(cue_ids))

    print("\nSummary of cue messages:")

    for cue in unique_cues:
        cues_sel = cue_ids == cue
        cue_times_sel = cue_times[cues_sel]
        deduplicated = np.sort(np.unique(cue_times_sel))

        if deduplicated.size > 1:
            time_diffs = np.diff(deduplicated)
            min_diff = time_diffs.min()
            max_diff = time_diffs.max()
            avg_diff = time_diffs.mean()

            print(
                f"""
{cues.get(cue, f'Unknown (0x{cue:04x})')}:
Found {cue_times_sel.size} instances.
Found {deduplicated.size} de-duplicated instances with
\tsmallest time difference: {min_diff} cycles ({seconds(min_diff):~.3g}),
\tlargest time difference: {max_diff} cycles ({seconds(max_diff):~.3g}),
\tmean time difference: {avg_diff:.2f} cycles ({seconds(avg_diff):~.3g})."""
            )
        elif cue_times_sel.size > 1:
            n = cue_times_sel.size
            print(
                f"\n{cues.get(cue, f'Unknown (0x{cue:04x})')}:  Found {n} instances,\n"
                f"\tall with the same timestamp."
            )
        else:
            print(f"\n{cues.get(cue, f'Unknown (0x{cue:04x})')}:  Found 1 instance.")
