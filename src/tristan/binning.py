"""Tools for binning events to images."""
from __future__ import annotations

from contextlib import nullcontext
from operator import mul

import numpy as np
import pandas as pd
from dask import array as da
from dask import dataframe as dd
from dask import distributed
from dask.diagnostics import ProgressBar
from numpy.typing import ArrayLike

from .data import (
    cue_id_key,
    cue_time_key,
    event_location_key,
    event_time_key,
    shutter_close,
    shutter_open,
)


def find_start_end(data: dd.DataFrame, show_progress: bool = False) -> (int, int):
    """
    Find the shutter open and shutter close timestamps.

    Args:
        data:           LATRD data.  Must contain one 'cue_id' entry and one
                        'cue_timestamp_zero' entry.  The two arrays are assumed to have
                        the same length.
        show_progress:  Whether to show a progress bar.

    Returns:
        The shutter open and shutter close timestamps, in clock cycles.
    """
    if show_progress:
        print("Finding detector shutter open and close times.")
        context = ProgressBar
    else:
        context = nullcontext

    indices = (data[cue_id_key] == shutter_open) | (data[cue_id_key] == shutter_close)
    times = data[cue_time_key][indices]

    with context():
        start, end = np.unique(da.compute(times))

    return start, end


def valid_events(data: dd.DataFrame, start: int, end: int) -> dd.DataFrame:
    """
    Return those events that have a timestamp in the specified range.

    Args:
        data:   LATRD data, containing an 'event_time_offset' column and optional
                'event_id' and 'event_energy' columns.
        start:  The start time of the accepted range, in clock cycles.
        end:    The end time of the accepted range, in clock cycles.

    Returns:
        The valid events.
    """
    valid = (start <= data[event_time_key]) & (data[event_time_key] < end)

    return data[valid]


def make_images(data: pd.DataFrame, image_size: tuple[int, int], cache: ArrayLike):
    """
    Bin LATRD events data into images of event counts.

    Given a collection of events data, a known image shape and an array of the
    desired time bin edges, make an image for each time bin, representing the number
    of events recorded at each pixel.  Add the binned images to an array representing
    the full image stack.

    Args:
        data:        LATRD data.  Must have one 'event_id' column and one
                     'event_time_offset' column.
        image_size:  The (y, x), i.e. (slow, fast) dimensions (number of pixels) of
                     the image.
        cache:       Array representing the image stack, to which the binned events
                     should be added.  This might be a Zarr array, in which case it
                     functions as an on-disk cache of the binned images.
    """
    # Construct a stack of images using dask.array.bincount and add them to the cache.
    for image_index in data[event_time_key].unique():
        locations = data[event_location_key][data[event_time_key] == image_index]
        pixel_counts = np.bincount(locations, minlength=mul(*image_size))
        pixel_counts = pixel_counts.astype(np.uint32).reshape(image_size)
        with distributed.Lock(image_index):
            # Beware!  Using inplace addition (+=) here causes the locking to fail
            # when the cache is a Zarr array.  Presumably this is due to Zarr
            # releasing the GIL before it has finished flushing the result of the
            # inplace addition to disk.
            cache[image_index] = cache[image_index] + pixel_counts

    return pd.DataFrame(columns=data.columns)


def find_image_indices(data: pd.DataFrame, bins: ArrayLike):
    """
    FIXME

    Args:
        data:  FIXME
        bins:  The time bin edges of the images (in clock cycles, to match the event
               timestamps).

    Returns:
        FIXME
    """
    num_images = len(bins) - 1

    # Find the index of the image to which each event belongs.
    if num_images > 1:
        data[event_time_key] = np.digitize(data[event_time_key], bins) - 1
    elif num_images:
        data[event_time_key] = 0

    return data
