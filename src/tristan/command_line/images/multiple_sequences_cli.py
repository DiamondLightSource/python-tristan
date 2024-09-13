"""
Bin events into several sequences of images, each corresponding to a different
pump-probe delay time interval.

With LATRD data from a pump-probe experiment, where the pump signal has a fairly
constant repeat rate, the recorded events are separated into groups corresponding to the
time elapsed since the most recent pump trigger signal.  Each group is binned into a
sequence of chronological images.  Each sequence is saved to a separate output file,
numbered from the shortest pump-probe delay to the longest.
"""

from __future__ import annotations

# Some platforms seem to raise OSError: [Errno -101] NetCDF: HDF error:
# '/path/to/HDF5/file.h5'
# Importing netCDF4 before h5py here seems to fix it.  ¯\_(ツ)_/¯
import netCDF4  # noqa F401

import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import zarr
from dask import array as da
from dask.diagnostics import ProgressBar
from hdf5plugin import Bitshuffle
from nexgen.nxs_copy import copy_tristan_nexus

from ... import WithLocalDistributedCluster, compute_with_progress
from ...binning import (
    events_to_images,
    find_preceding_bin_edge,
    find_preceding_bin_edge_index,
)
from ...data import (
    cue_keys,
    cue_times,
    cues,
    event_location_key,
    event_time_dtype,
    event_time_key,
    find_start_end,
    latrd_mf_data,
)
from ...storage import create_cache
from .. import check_multiple_output_files, data_files, triggers
from . import determine_image_size, exposure


def main(args):
    transfer_to_hdf5(*bin_image_sequences(args))


@WithLocalDistributedCluster()
def bin_image_sequences(args):
    """
    Utility for making multiple image sequences from a pump-probe data collection.

    The time between one pump trigger signal and the next is subdivided into a number
    of intervals of equal duration, quantising the time elapsed since the most recent
    trigger pulse.  Events are labelled according to the interval into which they fall
    and, for each interval in turn, all the events so labelled are binned into a
    sequence of images, providing a stack of image sequences that captures the
    evolution of the response of the measurement to a pump signal.
    """
    write_mode = "w" if args.force else "x"

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
    trigger_times = trigger_times.astype(int)

    if not trigger_times.size:
        sys.exit(f"Could not find a '{cues[trigger_type]}' signal.")
    elif not trigger_times.size > 1:
        sys.exit(f"Only one '{cues[trigger_type]}' signal found.  Two or more needed.")

    start, end = find_start_end(cues_data)

    intervals_end = np.diff(trigger_times).min()
    interval_time, _, num_intervals = exposure(
        0, intervals_end, args.interval, args.num_sequences
    )
    # Find the bins denoting to which image sequence each event belongs.
    interval_bins = np.linspace(
        0, intervals_end, num_intervals + 1, dtype=event_time_dtype
    )

    output_files, out_file_pattern = check_multiple_output_files(
        num_intervals, args.output_file, args.stem, "images", args.force
    )

    exposure_time, exposure_cycles, num_images = exposure(
        start, end, args.exposure_time, args.num_images
    )
    # Find the bins denoting images within a sequence.
    bins = np.linspace(start, end, num_images + 1, dtype=event_time_dtype)

    print(
        f"Using '{cues[trigger_type]}' as the pump signal,\n"
        f"binning events into {num_intervals} sequences, corresponding to "
        f"successive pump-probe delay intervals of {interval_time:.3g~#P}.\n"
        f"Each sequence consists of {num_images} images with an effective exposure "
        f"time of {exposure_time / num_intervals:.3g~#P}."
    )

    out_file_stem = out_file_pattern.stem

    n_dig = len(str(num_intervals))
    out_file_pattern = out_file_pattern.parent / f"{out_file_stem}_{'#' * n_dig}.h5"

    if input_nexus.exists():
        # Write output NeXus files if we have an input NeXus file.
        output_nexus_pattern = out_file_pattern.with_suffix(".nxs")
        for output_file in output_files:
            try:
                copy_tristan_nexus.multiple_images_nexus(
                    output_file,
                    input_nexus,
                    nbins=num_images,
                    write_mode=write_mode,
                )
            except FileExistsError:
                sys.exit(
                    f"One or more output files already exist, "
                    f"matching the pattern:\n\t"
                    f"{output_nexus_pattern}\n"
                    "Use '-f' to override, "
                    "or specify a different output file path with '-o'."
                )
    else:
        output_nexus_pattern = None

    # Get the events data.
    events_keys = event_time_key, event_location_key
    events_data = latrd_mf_data(raw_files, keys=events_keys)

    # Find the time elapsed since the most recent trigger signal.
    events_data = events_data.astype({event_time_key: int})
    pump_time = events_data[event_time_key].map_partitions(
        find_preceding_bin_edge, bins=trigger_times, meta=(event_time_key, int)
    )
    pump_probe_time = events_data[event_time_key] - pump_time
    # Enumerate the image sequence in the stack to which each event belongs.
    events_data["sequence"] = pump_probe_time.map_partitions(
        find_preceding_bin_edge_index, bins=interval_bins, meta=(event_time_key, int)
    )
    # Eliminate invalid sequence numbers (negative, or ≥ num_intervals).
    valid_sequence = 0 <= events_data["sequence"]
    valid_sequence &= events_data["sequence"] < num_intervals
    events_data = events_data[valid_sequence]

    # Make a cache for the image sequence stack.
    shape = num_intervals, num_images, *image_size
    cache = create_cache(out_file_pattern, shape=shape)

    # Set column order for binning.
    events_data = events_data[["sequence", event_time_key, event_location_key]]
    images = events_to_images(events_data, bins, shape, cache)

    print("Computing the binned images.")
    compute_with_progress(images)

    return cache, output_files, output_nexus_pattern or out_file_pattern, write_mode


def transfer_to_hdf5(
    cache: zarr.Array,
    output_files: list[Path | str],
    output_pattern: Path | str,
    write_mode: Literal["w"] | Literal["x"],
):
    """
    Transfer the contents of the cached image stack into several HDF5 files, one per
    image sequence.

    This must be done outside the context of a ``dask.distributed.Client``, since it
    requires the use of ``h5py.File`` objects, which cannot be serialised.
    """
    shape = cache.shape
    dtype = cache.dtype
    chunks = cache.chunks
    separate_arrays = list(da.from_zarr(cache))

    # Multi-threaded copy from Zarr to HDF5.
    with ExitStack() as stack:
        files = (stack.enter_context(h5py.File(f, write_mode)) for f in output_files)
        dsets = [
            f.require_dataset(
                "data", shape=shape[1:], dtype=dtype, chunks=chunks[1:], **Bitshuffle()
            )
            for f in files
        ]
        print("Transferring the images to the output files.")
        with ProgressBar():
            da.store(separate_arrays, dsets)

    print(f"Images written to\n\t{output_pattern}")
