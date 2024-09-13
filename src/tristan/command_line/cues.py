"""Summarise the cue messages in Tristan data."""

from __future__ import annotations

import argparse

import numpy as np
from dask.distributed import Client

from .. import compute_with_progress
from ..data import cue_keys, cue_time_key, cues, latrd_mf_data, reserved, seconds
from . import data_files, input_parser, version_parser

parser = argparse.ArgumentParser(
    description=__doc__, parents=[version_parser, input_parser]
)


def main(args=None):
    """Print a human-readable summary of the cue messages in a LATRD data set."""
    args = parser.parse_args(args)

    raw_files = data_files(args.data_dir, args.stem)

    data = latrd_mf_data(raw_files, keys=cue_keys)
    relevant = (data.cue_id > 0) & (data.cue_id != reserved)
    data = data[relevant]
    # Deduplicate identical cues with the same timestamp, but keep a count of them.
    data = data.groupby(data.columns.tolist()).size()
    print("Gathering the cue messages:")
    with Client(processes=False):
        (data,) = compute_with_progress(data, gather=True)

    data.rename("size", inplace=True)
    data.sort_index(inplace=True)
    data = data.reset_index(level=cue_time_key)

    print("\nSummary of cue messages:")

    for cue in data.index.unique():
        cue_description = cues.get(cue, f"Unknown (0x{cue:04x})")
        selection = data.loc[[cue]]

        if len(selection) > 1:
            time_diffs = np.diff(selection[cue_time_key])
            min_diff = time_diffs.min()
            max_diff = time_diffs.max()
            mean_diff = time_diffs.mean()

            print(
                f"{cue_description}:"
                f"Found {selection['size'].sum()} instances."
                f"Found {selection['size'].count()} de-duplicated instances with"
                f"\tsmallest time difference: {min_diff} cycles "
                f"({seconds(min_diff):.3g~#P}),"
                f"\tlargest time difference: {max_diff} cycles "
                f"({seconds(max_diff):.3g~#P}),"
                f"\tmean time difference: {mean_diff:.2f} cycles "
                f"({seconds(mean_diff):.3g~#P})."
            )
        elif (n := selection["size"].item()) > 1:
            print(
                f"\n{cue_description}:  Found {n} instances,\n"
                f"\tall with the same time stamp."
            )
        else:
            print(f"\n{cue_description}:  Found 1 instance.")
