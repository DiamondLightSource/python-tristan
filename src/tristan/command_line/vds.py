"""
Create an HDF5 virtual data set (VDS) file to aggregate raw Tristan events data.

By default, this file will be saved in the same directory as the raw data and
detector metadata files, retaining the same naming convention.  So if the metadata
file is named 'my_data_1_meta.h5', then the new VDS file will be named
'my_data_1_vds.h5'.
"""

from __future__ import annotations

import argparse

import h5py

from ..vds import time_slice_info, virtual_data_set
from . import check_output_file, data_files, input_parser, version_parser

parser = argparse.ArgumentParser(
    description=__doc__, parents=[version_parser, input_parser]
)
parser.add_argument(
    "-o",
    "--output-file",
    help="File name for output VDS file.  "
    "By default, the pattern of the input file will be used, with '_meta.h5' "
    "replaced with '_vds.h5'.",
)
parser.add_argument(
    "-f",
    "--force",
    help="Force the output file to over-write any existing file with the same name.",
    action="store_true",
)


def main(args=None):
    """Utility for making an HDF5 VDS from raw Tristan data."""
    args = parser.parse_args(args)
    output_file = check_output_file(args.output_file, suffix="vds", force=args.force)

    raw_files, meta_file = data_files(args.data_dir, args.stem)
    with h5py.File(meta_file, "r") as f:
        ts_info = time_slice_info(f)
        layouts = virtual_data_set(raw_files, f, *ts_info)

    with h5py.File(output_file, "w" if args.force else "x") as f:
        for layout in layouts.items():
            f.create_virtual_dataset(*layout)

    print(f"Virtual data set file written to\n\t{output_file}")
