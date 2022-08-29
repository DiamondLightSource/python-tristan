"""Tools for binning events to images."""
from __future__ import annotations

from operator import mul

import numpy as np
import pandas as pd
import zarr
from dask import array as da
from dask import dataframe as dd
from dask.diagnostics import ProgressBar
from numpy.typing import ArrayLike

from .data import (
    cue_id_key,
    cue_time_key,
    event_location_key,
    event_time_key,
    pixel_index,
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
    data: pd.DataFrame, image_size: tuple[int, int], bins: ArrayLike, cache: ArrayLike
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
    cache = cache.vindex if isinstance(cache, zarr.Array) else cache

    # For empty data, do nothing.
    return_value = pd.DataFrame(columns=data.columns)
    if data.empty:
        return return_value

    # We need to ensure that the chunk layout of the event location array matches
    # that of the event time array, so that we can perform matching blockwise iterations
    data[event_location_key] = pixel_index(data[event_location_key], image_size)

    num_images = len(bins) - 1

    if num_images > 1:
        # We cannot perform a single bincount of the entire data set because that
        # would require allocating enough memory for the entire image stack.

        # Find the index of the image to which each event belongs.
        data[event_time_key] = np.digitize(data[event_time_key], bins) - 1
        image_indices = data[event_time_key].unique()

    else:
        image_indices = [0]

    # Construct a stack of images using dask.array.bincount and add them to the cache.
    for i in image_indices:
        image = np.bincount(
            data[event_location_key][data[event_time_key] == i],
            minlength=mul(*image_size),
        )
        cache[i] += image.astype(np.uint32).reshape(image_size)

    return return_value
