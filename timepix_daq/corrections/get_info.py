#!/usr/bin/env python

import argparse
import os

import h5py

parser = argparse.ArgumentParser(description="Get comments from original nexus file")
parser.add_argument(
    "working_directory", help="Directory in which the relevant NeXus file can be found"
)


def run(filename):
    with h5py.File(filename, "r") as fh:
        count_time = fh["entry/instrument/detector/count_time"][()]
        try:
            comment = fh["entry/Comments/Note"][()]
        except KeyError:
            comment = None

    return count_time, comment


if __name__ == "__main__":
    args = parser.parse_args()
    for filename in os.listdir(args.working_directory):
        ext = os.path.splitext(filename)[1]
        if ext == ".nxs":
            count_time, comment = run(os.path.join(args.working_directory, filename))
