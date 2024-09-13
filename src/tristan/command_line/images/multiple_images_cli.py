"""Bin the events from a LATRD Tristan data collection into multiple images."""

from __future__ import annotations

# Some platforms seem to raise OSError: [Errno -101] NetCDF: HDF error:
# '/path/to/HDF5/file.h5'
# Importing netCDF4 before h5py here seems to fix it.  ¯\_(ツ)_/¯
import netCDF4  # noqa F401

import sys

import h5py
import numpy as np
import zarr
from hdf5plugin import Bitshuffle
from nexgen.nxs_copy import copy_tristan_nexus

from ... import WithLocalDistributedCluster, compute_with_progress
from ...binning import align_bins, events_to_images
from ...data import (
    cue_keys,
    cues,
    event_location_key,
    event_time_dtype,
    event_time_key,
    find_start_end,
    first_cue_time,
    latrd_mf_data,
)
from ...storage import create_cache
from .. import check_output_file, data_files, triggers
from . import determine_image_size, exposure


@WithLocalDistributedCluster()
def main(args):
    """
    Utility for making multiple images from event-mode data.

    The time between the start and end of the data collection is subdivided into a
    number of exposures of equal duration, providing a chronological stack of images.
    """
    write_mode = "w" if args.force else "x"
    output_file = check_output_file(args.output_file, args.stem, "images", args.force)

    input_nexus = args.data_dir / f"{args.stem}.nxs"
    if not input_nexus.exists():
        print(
            "Could not find a NeXus file containing experiment metadata.\n"
            "Resorting to writing raw image data without accompanying metadata."
        )

    image_size = args.image_size or determine_image_size(input_nexus)

    raw_files = data_files(args.data_dir, args.stem)

    cues_data = latrd_mf_data(raw_files, keys=cue_keys)

    start, end = map(int, find_start_end(cues_data))
    exposure_time, exposure_cycles, num_images = exposure(
        start, end, args.exposure_time, args.num_images
    )

    if args.align_trigger:
        trigger_type = triggers[args.align_trigger]
        print(
            f"Image start and end times will be chosen such that the first "
            f"'{cues[trigger_type]}' after the detector shutter open signal is "
            f"aligned with an image boundary."
        )
        # Note we are assuming that the first trigger time is after shutter open.
        trigger_time = first_cue_time(cues_data, trigger_type, after=start)
        if trigger_time is None:
            sys.exit(
                f"Could not find a '{cues[trigger_type]}' signal after the "
                f"detector shutter open signal."
            )
        trigger_time = int(trigger_time)

        if args.exposure_time:
            # Adjust the start time to align a bin edge with the trigger time.
            n_bins_before = (trigger_time - start) // exposure_cycles
            start = trigger_time - n_bins_before * exposure_cycles
            num_images = (end - start) // exposure_cycles
        # It is assumed that start ≤ trigger_time ≤ end.
        else:
            start, exposure_cycles = align_bins(start, trigger_time, end, num_images)

    end = start + num_images * exposure_cycles
    bins = np.linspace(start, end, num_images + 1, dtype=event_time_dtype)

    if input_nexus.exists():
        try:
            # Write output NeXus file if we have an input NeXus file.
            output_nexus = copy_tristan_nexus.multiple_images_nexus(
                output_file,
                input_nexus,
                nbins=num_images,
                write_mode=write_mode,
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
        f"Binning events into {num_images} images with an exposure time of "
        f"{exposure_time:.3g~#P}."
    )

    # Make a cache for the images.
    shape = num_images, *image_size
    cache = create_cache(output_file, shape)

    events_data = latrd_mf_data(raw_files, keys=(event_time_key, event_location_key))
    images = events_to_images(events_data, bins, shape, cache)

    print("Computing the binned images.")
    compute_with_progress(images)

    print("Transferring the images to the output file.")
    with h5py.File(output_file, write_mode) as f:
        zarr.copy_all(zarr.open(cache.store), f, **Bitshuffle())

    print(f"Images written to\n\t{output_nexus or output_file}")
