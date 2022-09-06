"""Tools for binning events to images."""
from __future__ import annotations

from operator import mul

import numpy as np
import pandas as pd
from dask import array as da
from dask import dataframe as dd
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


def find_start_end(data: dd.DataFrame, show_progress: bool = False) -> tuple[int, int]:
    """
    Find the shutter open and shutter close timestamps.

    Args:
        data:           LATRD data.  Must contain one 'cue_id' column and one
                        'cue_timestamp_zero' column.
        show_progress:  Whether to show a progress bar.

    Returns:
        The shutter open and shutter close timestamps, in clock cycles.
    """
    if show_progress:
        print("Finding detector shutter open and close times.")

    cues = data[cue_id_key].values.compute_chunk_sizes()
    start = da.argmax(cues == shutter_open)
    end = da.argmax(cues == shutter_close)

    # Optionally, show progress.
    if show_progress:
        with ProgressBar():
            start, end = da.compute(start, end)
    else:
        start, end = da.compute(start, end)

    start, end = data[cue_time_key].values.compute_chunk_sizes()[[start, end]]
    return da.compute(start, end)


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


def make_images(
    data: pd.DataFrame, image_index: int, image_size: tuple[int, int], cache: ArrayLike
):
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
        bins:        The time bin edges of the images (in clock cycles, to match the
                     event timestamps).
        cache:       Array representing the image stack, to which the binned events
                     should be added.  This might be a Zarr array, in which case it
                     functions as an on-disk cache of the binned images.
    """
    # Construct a stack of images using dask.array.bincount and add them to the cache.
    location = data[event_location_key]
    i = data[event_time_key]
    image = np.bincount(location[i == image_index], minlength=mul(*image_size))
    cache[image_index] += image.astype(np.uint32).reshape(image_size)


def find_image_indices(data: pd.DataFrame, bins: ArrayLike):
    num_images = len(bins) - 1

    # Find the index of the image to which each event belongs.
    if num_images > 1:
        data[event_time_key] = np.digitize(data[event_time_key], bins) - 1
        image_indices = data[event_time_key].unique()
    elif num_images:
        data[event_time_key] = 0
        image_indices = [0]
    else:
        image_indices = []

    data.rename({event_time_key: "image_index"})
    return data, image_indices
