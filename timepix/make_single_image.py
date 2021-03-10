"""
Aggregate all the events from a LATRD Tristan data collection into a single image.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from dask import array as da
from dask.diagnostics import ProgressBar
from hdf5plugin import Bitshuffle

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "input_file",
    help="Tristan NeXus ('.nxs') file containing events data.",
    metavar="input-file",
)
parser.add_argument(
    "-o",
    "--output-file",
    help="File name or location for output image file, defaults to working directory.  "
    "If only a directory location is given, the pattern of the input file will be "
    "used, with '<name>.nxs' replaced with '<name>_single_image.h5'.",
)
parser.add_argument(
    "-f",
    "--force",
    help="Force the output image file to over-write any existing file with the same "
    "name.",
    action="store_true",
)

shutter_open = 0x840
shutter_close = 0x880

event_keys = "event_id", "event_time_offset"
cue_keys = "cue_id", "cue_timestamp_zero"


def first_cue_time(cue_ids: da.Array, cue_times: da.Array, msg: int) -> Optional[int]:
    """Find the timestamp of the first instance of a message."""
    index = da.argmax(cue_ids == msg)
    if index == 0 and cue_ids[0] != msg:
        return
    return cue_times[index]


def make_image(nexus_file: h5py.File) -> da.Array:
    """Make a single image from all events occurring between shutter open and close."""
    image_shape = nexus_file["/entry/instrument/detector/module/data_size"][()]

    cue_ids = da.from_array(nexus_file["entry/data/data/cue_id"])
    cue_times = da.from_array(nexus_file["entry/data/data/cue_timestamp_zero"])
    event_ids = da.from_array(nexus_file["entry/data/data/event_id"])
    event_times = da.from_array((nexus_file["/entry/data/data/event_time_offset"]))

    start_time = first_cue_time(cue_ids, cue_times, shutter_open).compute()
    end_time = first_cue_time(cue_ids, cue_times, shutter_close).compute()
    measurement_window = (event_times >= start_time) & (event_times < end_time)

    locations = event_ids[measurement_window]
    x, y = da.divmod(locations, 0x2000)
    locations = x + y * image_shape[1]

    image = da.bincount(locations, minlength=np.prod(image_shape))
    image = image.reshape((1, *image_shape))

    return image


def find_file_names(args: argparse.Namespace) -> (Path, Path):
    """Resolve the input and output file names."""
    input_file = Path(args.input_file).expanduser().resolve()
    output_file = Path(args.output_file or "").expanduser().resolve()

    # Crudely check the input file is a NeXus file.  We need the its data_size data set.
    if input_file.suffix != ".nxs":
        sys.exit(
            "Input file name did not have the expected format '<name>.nxs':\n"
            f"\t{input_file}"
        )

    if output_file.is_dir():
        output_file /= input_file.stem + "_single_image.h5"

    if not args.force and output_file.exists():
        sys.exit(
            f"This output file already exists:\n\t{output_file}\n"
            f"Use '-f' to override or specify a different output file path with '-o'."
        )

    return input_file, output_file


if __name__ == "__main__":
    args = parser.parse_args()
    input_file, output_file = find_file_names(args)

    with h5py.File(input_file, "r") as f:
        image = make_image(f)
        print(f"Writing image to {output_file}.")
        with ProgressBar():
            image.to_hdf5(output_file, "data", **Bitshuffle(nelems=0, lz4=True))
