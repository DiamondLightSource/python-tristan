"""
Aggregate all the events from a LATRD Tristan data collection into a single image.
"""

from __future__ import annotations

# Some platforms seem to raise OSError: [Errno -101] NetCDF: HDF error:
# '/path/to/HDF5/file.h5'
# Importing netCDF4 before h5py here seems to fix it.  ¯\_(ツ)_/¯
import netCDF4  # noqa F401

import sys
from operator import mul

import h5py
from dask import array as da
from hdf5plugin import Bitshuffle
from nexgen.nxs_copy import copy_tristan_nexus

from ... import WithLocalDistributedCluster, compute_with_progress
from ...data import (
    cue_keys,
    event_location_key,
    event_time_key,
    find_start_end,
    image_dtype,
    latrd_mf_data,
    pixel_index,
    pixel_index_key,
    valid_events,
)
from .. import check_output_file, data_files
from . import determine_image_size


@WithLocalDistributedCluster()
def main(args):
    """Utility for making a single image from event-mode data."""
    write_mode = "w" if args.force else "x"
    output_file = check_output_file(
        args.output_file, args.stem, "single_image", args.force
    )
    input_nexus = args.data_dir / f"{args.stem}.nxs"
    if input_nexus.exists():
        try:
            # Write output NeXus file if we have an input NeXus file.
            output_nexus = copy_tristan_nexus.single_image_nexus(
                output_file, input_nexus, write_mode=write_mode
            )
        except FileExistsError:
            sys.exit(
                f"This output file already exists:\n\t"
                f"{output_file.with_suffix('.nxs')}\n"
                "Use '-f' to override, "
                "or specify a different output file path with '-o'."
            )
    else:
        output_nexus = None
        print(
            "Could not find a NeXus file containing experiment metadata.\n"
            "Resorting to writing raw image data without accompanying metadata."
        )

    image_size = args.image_size or determine_image_size(input_nexus)

    raw_files = data_files(args.data_dir, args.stem)

    cues_data = latrd_mf_data(raw_files, keys=cue_keys)
    start, end = find_start_end(cues_data)

    events_data = latrd_mf_data(raw_files, keys=(event_location_key, event_time_key))
    # Select only those events happening between shutter open and  shutter close.
    events_data = valid_events(events_data, start, end)
    # Convert each event_id to the index of the pixel in the flattened image array.
    events_data = pixel_index(events_data, image_size)
    # Bin to a single image
    image = da.bincount(events_data[pixel_index_key], minlength=mul(*image_size))

    print("Binning events into a single image.")
    (image,) = compute_with_progress(image, gather=True)
    image = image.reshape(image_size).astype(image_dtype)

    image_stack_shape = (1, *image_size)
    with h5py.File(output_file, write_mode) as f:
        f.require_dataset(
            "data",
            shape=image_stack_shape,
            dtype=image.dtype,
            chunks=image_stack_shape,
            **Bitshuffle(),
        )
        f["data"][()] = image

    print(f"Image written to\n\t{output_nexus or output_file}")
