"""Summarise the cue messages in Tristan data."""

import argparse
from contextlib import ExitStack

import h5py
import numpy as np
from dask import array as da

from .. import cue_id_key, cue_keys, cue_time_key, cues, reserved, seconds
from ..data import data_files, find_input_file_name
from . import input_parser, version_parser

parser = argparse.ArgumentParser(
    description=__doc__, parents=[version_parser, input_parser]
)


def main(args=None):
    args = parser.parse_args(args)

    data_dir, root = find_input_file_name(args.input_file)
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
