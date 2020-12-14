#!/usr/bin/env python

import argparse
import os

import h5py

parser = argparse.ArgumentParser(description="Get comments from original nexus file")
parser.add_argument(
    "working_directory", help="Directory in which the relevant NeXus file can be found"
)


def run(work_dir):
    for filename in os.listdir(work_dir):
        ext = os.path.splitext(filename)[1]
        if ext == ".nxs":
            with h5py.File(os.path.join(work_dir, filename), "r") as fh:
                count_time = fh["entry/instrument/detector/count_time"][()]
                try:
                    comment = fh["entry/Comments/Note"][()]
                except KeyError:
                    comment = None
    return count_time, comment


if __name__ == "__main__":
    args = parser.parse_args()
    count_time, comment = run(args.working_directory)
