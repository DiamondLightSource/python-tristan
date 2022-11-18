"""Tools for binning events to images."""
from __future__ import annotations

from operator import mul
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import zarr
from dask import dataframe as dd
from dask import distributed
from numpy.typing import ArrayLike

from .data import (
    cue_id_key,
    cue_time_key,
    event_location_key,
    event_time_key,
    pixel_index,
    shutter_close,
    shutter_open,
    valid_events,
)


def find_start_end(data: dd.DataFrame) -> (int, int):
    """
    Find the shutter open and shutter close timestamps.

    Args:
        data:  LATRD data.  Must contain one 'cue_id' entry and one
               'cue_timestamp_zero' entry.  The two arrays are assumed to have the
               same length.

    Returns:
        The shutter open and shutter close timestamps, in clock cycles.
    """
    indices = (data[cue_id_key] == shutter_open) | (data[cue_id_key] == shutter_close)
    times = data[cue_time_key][indices]
    start, end = np.unique(times.compute())

    return start, end


def align_bins(start: int, align: int, end: int, n_bins: int) -> (int, int):
    """
    Divide an interval into a specified number of bins, aligning with a given value.

    Take three integers, ``start`` ≤ ``align`` ≤ ``end``, and find a way to span the
    largest possible interval between ``start`` and ``end`` with a specified number of
    bins, while ensuring that one of the bin edges is aligned with a specified value.

    Args:
        start:   The start of the interval.
        align:   The value to which a bin edge should be aligned.
        end:     The end of the interval.
        n_bins:  The number of bins.

    Returns:
        The first bin edge and the bin width, from which all the bin edges can be
        derived.
    """
    if align <= (start + end) / 2:
        # Find the bin width by pinning a bin edge to the align time and the last bin
        # edge to the end time, then maximising the number of bins we can fit between
        # the start time and the trigger time.
        # At least half the images must happen after the trigger time.
        n_bins_after = np.arange(n_bins // 2 or 1, n_bins + 1)
        bin_width = ((end - align) / n_bins_after).astype(int)
        new_start = end - n_bins * bin_width
        best = np.argmax(new_start >= start)
        start = new_start[best]
        bin_width = bin_width[best]
    else:
        # Find the bin width by pinning a bin edge to the align time and the first
        # bin edge to the start time, then maximising the number of bins we can fit
        # between the trigger time and the end time.
        # At least half the images must happen before the trigger time.
        n_bins_before = np.arange(n_bins // 2 or 1, n_bins + 1)
        bin_width = ((align - start) / n_bins_before).astype(int)
        new_end = start + n_bins * bin_width
        best = np.argmax(new_end <= end)
        bin_width = bin_width[best]

    return start, bin_width


def create_cache(
    output_file: Path | str, num_images: int, image_size: tuple[int, int]
) -> zarr.Array:
    """
    Make a Zarr array of zeros, suitable for using as an image binning cache.

    The array will have shape (num_images, *image_size) and will be chunked by image,
    i.e. the chunk shape will be (1, *image_size).

    Args:
        output_file:  Output file name.  Any file extension will be replaced with .zarr.
        num_images:   The number of images in the array.
        image_size:   The size of an image in the array.

    Returns:

    """
    shape = num_images, *image_size
    chunks = 1, *image_size
    output = Path(output_file).with_suffix(".zarr")
    return zarr.zeros(
        shape, chunks=chunks, dtype=np.int32, overwrite=True, store=output, path="data"
    )


def find_time_bins(data: pd.DataFrame, bins: Sequence[int]):
    """
    Convert the event timestamps in LATRD data to time bin indices.

    For each event, determine the index of the bin into which the event will fall.

    Args:
        data:  LATRD events data.  Must have an ``event_time_offset`` column.
        bins:  The time bin edges of the images (in clock cycles, to match the event
               timestamps).

    Returns:
        A DataFrame which matches the input data except that the
        ``event_time_offset`` column is replaced with a column of ``time_bin`` indices.
    """
    num_images = len(bins) - 1

    # Find the index of the image to which each event belongs.
    if num_images > 1:
        data[event_time_key] = np.digitize(data[event_time_key], bins) - 1
    elif num_images:
        data[event_time_key] = 0

    return data.rename(columns={event_time_key: "time_bin"})


def make_images(data: pd.DataFrame, image_size: tuple[int, int], cache: ArrayLike):
    """
    Bin LATRD events data into images of event counts.

    Given a collection of events data, a known image shape and an array of the
    desired time bin edges, make an image for each time bin, representing the number
    of events recorded at each pixel.  Add the binned images to an array representing
    the full image stack.

    Args:
        data:        LATRD data.  Must have an ``event_id`` column and an
                    ``image_index`` column.
        image_size:  The (y, x), i.e. (slow, fast) dimensions (number of pixels) of
                     the image.
        cache:       Array representing the image stack, to which the binned events
                     should be added.  This might be a Zarr array, in which case it
                     functions as an on-disk cache of the binned images.
    """
    # Construct a stack of images using dask.array.bincount and add them to the cache.
    for image_index in data["time_bin"].unique():
        image_index = int(image_index)  # Convert to a serialisable type.
        locations = data[event_location_key][data["time_bin"] == image_index]
        pixel_counts = np.bincount(locations, minlength=mul(*image_size))
        pixel_counts = pixel_counts.astype(np.uint32).reshape(image_size)
        with distributed.Lock(image_index):
            # Beware!  When the cache is a Zarr array, using inplace addition (+=)
            # here means the locking fails to prevent the race condition of multiple
            # make_images calls in separate threads accessing the same chunk of the
            # cache simultaneously.  Presumably this is due to Zarr releasing the GIL
            # before it has finished flushing the result of the inplace addition to
            # disk.
            cache[image_index] = cache[image_index] + pixel_counts

    return pd.DataFrame(columns=data.columns)


def events_to_images(
    data: dd.DataFrame,
    bins: Sequence[int],
    image_size: tuple[int, int],
    cache: ArrayLike,
) -> dd.DataFrame:
    """
    Construct a stack of images from events data.

    From a sequence of LATRD events data, bin the events to images and store the
    binned images in a cache array.  The cache may be backed with on-disk storage,
    as in the case of a Zarr array, or may be a simple in-memory object, like a NumPy
    array.

    Args:
        data:        LATRD events data.  Must have an ``event_time_offset`` column
                     and an ``event_id`` column.
        bins:        The time bin edges of the images (in clock cycles, to match the
                     event timestamps).
        image_size:  The size of each image.
        cache:       An array representing the eventual image stack, having shape
                     ``(len(bins) - 1, *image_size)``, to which the pixel counts from
                     this binning operation will be added.

    Returns:
        A Dask collection representing the lazy image binning computation.
    """
    # Consider only those events that occur between the start and end times.
    data = valid_events(data, bins[0], bins[-1])
    # Convert the event IDs to a form that is suitable for a NumPy bincount.
    data[event_location_key] = pixel_index(data[event_location_key], image_size)

    columns = event_location_key, "time_bin"
    dtypes = data.dtypes
    dtypes["time_bin"] = dtypes.pop(event_time_key)
    meta = pd.DataFrame(columns=columns).astype(dtype=dtypes)
    data = data.map_partitions(find_time_bins, bins=bins, meta=meta)

    # Bin to images, partition by partition.
    data = dd.map_partitions(
        make_images, data, image_size, cache, meta=meta, enforce_metadata=False
    )

    return data
