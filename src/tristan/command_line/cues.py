"""Summarise the cue messages in Tristan data."""

from __future__ import annotations

import argparse

import numpy as np

from ..data import cue_keys, cues, latrd_data, reserved, seconds
from . import data_files, input_parser, version_parser

parser = argparse.ArgumentParser(
    description=__doc__, parents=[version_parser, input_parser]
)


def main(args=None):
    """Print a human-readable summary of the cue messages in a LATRD data set."""
    args = parser.parse_args(args)

    raw_files, _ = data_files(args.data_dir, args.stem)

    with latrd_data(raw_files, keys=cue_keys) as data:
        relevant = (data.cue_id > 0) & (data.cue_id != reserved)
        cue_ids = data.cue_id[relevant].compute()
        cue_times = data.cue_timestamp_zero[relevant].compute()

    unique_cues = np.sort(np.unique(cue_ids))

    print("\nSummary of cue messages:")

    for cue in unique_cues:
        cue_description = cues.get(cue, f"Unknown (0x{cue:04x})")
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
{cue_description}:
Found {cue_times_sel.size} instances.
Found {deduplicated.size} de-duplicated instances with
\tsmallest time difference: {min_diff} cycles ({seconds(min_diff):.3g~#P}),
\tlargest time difference: {max_diff} cycles ({seconds(max_diff):.3g~#P}),
\tmean time difference: {avg_diff:.2f} cycles ({seconds(avg_diff):.3g~#P})."""
            )
        elif cue_times_sel.size > 1:
            n = cue_times_sel.size
            print(
                f"\n{cue_description}:  Found {n} instances,\n"
                f"\tall with the same time stamp."
            )
        else:
            print(f"\n{cue_description}:  Found 1 instance.")
