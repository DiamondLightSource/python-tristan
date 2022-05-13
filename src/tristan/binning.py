"""Tools for binning events to images."""
from __future__ import annotations

from operator import mul

import numpy as np
from dask import array as da
from dask.diagnostics import ProgressBar
from numpy.typing import ArrayLike

from . import blockwise_selection
from .data import LatrdData, event_keys, pixel_index, shutter_close, shutter_open


def find_start_end(data: LatrdData, show_progress: bool = False) -> tuple[int, int]:
    """
    Find the shutter open and shutter close timestamps.

    Args:
        data:           LATRD data.  Must contain one 'cue_id' field and one
                        'cue_timestamp_zero' field.  The two arrays are assumed to have
                        the same length.
        show_progress:  Whether to show a progress bar.

    Returns:
        The shutter open and shutter close timestamps, in clock cycles.
    """
    if show_progress:
        print("Finding detector shutter open and close times.")

    start_index = da.argmax(data.cue_id == shutter_open)
    end_index = da.argmax(data.cue_id == shutter_close)

    # Optionally, show progress.
    if show_progress:
        with ProgressBar():
            start_index, end_index = da.compute(start_index, end_index)
    else:
        start_index, end_index = da.compute(start_index, end_index)

    start_end = data.cue_timestamp_zero[[start_index, end_index]]

    return tuple(start_end.compute())


def valid_events(data: LatrdData, start: int, end: int) -> LatrdData:
    """
    Return those events that have a timestamp in the specified range.

    Args:
        data:   LATRD data, containing an 'event_time_offset' field and optional
                'event_id' and 'event_energy' fields.
        start:  The start time of the accepted range, in clock cycles.
        end:    The end time of the accepted range, in clock cycles.

    Returns:
        A dictionary containing only the valid events.
    """
    valid = (start <= data.event_time_offset) & (data.event_time_offset < end)

    for key in event_keys:
        value = getattr(data, key)
        if value is not None:
            value = value.rechunk(data.event_time_offset.chunks)
            setattr(data, key, blockwise_selection(value, valid))

    return data


def make_images(
    data: LatrdData, image_size: tuple[int, int], bins: ArrayLike
) -> da.Array:
    """
    Bin LATRD events data into images of event counts.

    Given a collection of events data, a known image shape and an array of the
    desired time bin edges, make an image for each time bin, representing the number
    of events recorded at each pixel.

    Args:
        data:        LATRD data.  Must have one 'event_id' field and one
                     'event_time_offset' field.  The two arrays are assumed to have
                     the same length.
        image_size:  The (y, x), i.e. (slow, fast) dimensions (number of pixels) of
                     the image.
        bins:        The time bin edges of the images (in clock cycles, to match the
                     event timestamps).

    Returns:
        A dask array representing the calculations required to obtain the
        resulting image.
    """
    # We need to ensure that the chunk layout of the event location array matches
    # that of the event time array, so that we can perform matching blockwise iterations
    event_locations = data.event_id.rechunk(data.event_time_offset.chunks)
    event_locations = pixel_index(event_locations, image_size)

    num_images = len(bins) - 1

    if num_images > 1:
        # We cannot perform a single bincount of the entire data set because that
        # would require allocating enough memory for the entire image stack.

        # Find the index of the image to which each event belongs.
        image_indices = da.digitize(data.event_time_offset, bins) - 1

        # Construct a stack of images using dask.array.bincount.
        images = []
        for i in range(num_images):
            image_events = blockwise_selection(event_locations, image_indices == i)
            images.append(da.bincount(image_events, minlength=mul(*image_size)))

        images = da.stack(images)
    else:
        images = da.bincount(event_locations, minlength=mul(*image_size))

    return images.astype(np.uint32).reshape(num_images, *image_size)
