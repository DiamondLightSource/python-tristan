"""
Bin events into images representing different pump-probe delays.

With LATRD data from a pump-probe experiment, where the pump signal has a fairly
constant repeat rate, bin events into a stack of images spanning the range of pump-probe
delay times, from shortest to longest.
"""

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
from ...binning import events_to_images, find_preceding_bin_edge
from ...data import (
    cue_keys,
    cue_times,
    cues,
    event_location_key,
    event_time_key,
    latrd_mf_data,
)
from ...storage import create_cache
from .. import check_output_file, data_files, triggers
from . import determine_image_size, exposure


@WithLocalDistributedCluster()
def main(args):
    """
    Utility for making multiple images from a pump-probe data collection.

    The time between one pump trigger signal and the next is subdivided into a number
    of exposures of equal duration.  Data from all such pump-to-pump intervals is
    aggregated, providing a single stack of images that captures the evolution of the
    response of the measurement to a pump signal.
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

    trigger_type = triggers.get(args.trigger_type)

    cues_data = latrd_mf_data(raw_files, keys=cue_keys)
    print("Finding trigger signal times.")
    trigger_times = cue_times(cues_data, trigger_type)
    trigger_times = trigger_times.astype(np.int64)

    if not trigger_times.size:
        sys.exit(f"Could not find a '{cues[trigger_type]}' signal.")
    elif not trigger_times.size > 1:
        sys.exit(f"Only one '{cues[trigger_type]}' signal found.  Two or more needed.")

    end = np.diff(trigger_times).min()
    exposure_time, num_images = args.exposure_time, args.num_images
    exposure_time, _, num_images = exposure(0, end, exposure_time, num_images)
    bins = np.linspace(0, end, num_images + 1, dtype=np.int64)

    if input_nexus.exists():
        try:
            # Write output NeXus file if we have an input NeXus file.
            output_nexus = copy_tristan_nexus.single_image_nexus(
                output_file,
                input_nexus,
                write_mode=write_mode,
                pump_probe_bins=num_images,
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
        f"{exposure_time:.3g~#P} according to the time elapsed since the most "
        f"recent '{cues[trigger_type]}' signal."
    )

    # Make a cache for the images.
    shape = num_images, *image_size
    images = create_cache(output_file, shape)

    # Get the events data.
    keys = event_time_key, event_location_key
    events_data = latrd_mf_data(raw_files, keys=keys)
    # Measure the event time as time elapsed since the most recent trigger signal.
    events_data = events_data.astype({event_time_key: np.int64})
    events_data[event_time_key] -= events_data[event_time_key].map_partitions(
        find_preceding_bin_edge, bins=trigger_times, meta=(event_time_key, np.int64)
    )

    # Bin the events into images.
    events_data = events_to_images(events_data, bins, shape, images)

    print("Computing the binned images.")
    compute_with_progress(events_data)

    print("Transferring the images to the output file.")
    with h5py.File(output_file, write_mode) as f:
        zarr.copy_all(zarr.open(images.store), f, **Bitshuffle())

    print(f"Images written to\n\t{output_nexus or output_file}")
