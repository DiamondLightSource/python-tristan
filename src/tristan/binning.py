"""Tools for binning events to images."""


from operator import mul
from typing import Dict, Tuple

import numpy as np
from dask import array as da
from numpy.typing import ArrayLike

from .data import (
    event_location_key,
    event_time_key,
    first_cue_time,
    pixel_index,
    shutter_close,
    shutter_open,
)


def find_start_end(data: Dict[str, da.Array]) -> (int, int):
    """
    Find the shutter open and shutter close timestamps.

    Args:
        data:  A LATRD data dictionary (a dictionary with data set names as keys
               and Dask arrays as values).  Must contain one entry for cue id
               messages and one for cue timestamps.  The two arrays are assumed
               to have the same length.

    Returns:
        The shutter open and shutter close timestamps, in clock cycles.
    """
    start_time = first_cue_time(data, shutter_open)
    end_time = first_cue_time(data, shutter_close)
    return da.compute(start_time, end_time)


def make_images(
    data: Dict[str, da.Array], image_size: Tuple[int, int], bins: ArrayLike
) -> da.Array:
    """
    Bin LATRD events data into images of event counts.

    Given a collection of events data, a known image shape and an array of the
    desired time bin edges, make an image for each time bin, representing the number
    of events recorded at each pixel.

    Args:
        data:        A LATRD data dictionary (a dictionary with data set names as keys
                     and Dask arrays as values).  Must contain one entry for event
                     location messages and one for event timestamps.  The two arrays are
                     assumed to have the same length.
        image_size:  The (y, x), i.e. (slow, fast) dimensions (number of pixels) of
                     the image.
        bins:        The time bin edges of the images (in clock cycles, to match the
                     event timestamps).

    Returns:
        A dask array representing the calculations required to obtain the
        resulting image.
    """
    event_times = data[event_time_key]
    event_locations = data[event_location_key]

    valid_events = (bins[0] <= event_times) & (event_times < bins[-1])
    event_times = event_times[valid_events]
    event_locations = event_locations[valid_events]
    event_locations = pixel_index(event_locations, image_size)

    num_images = len(bins) - 1

    if num_images > 1:
        # We cannot perform a single bincount of the entire data set because that
        # would require allocating enough memory for the entire image stack.
        image_indices = da.digitize(event_times, bins) - 1

        images = da.stack(
            [
                da.bincount(
                    event_locations[image_indices == i], minlength=mul(*image_size)
                )
                for i in range(num_images)
            ]
        )
    else:
        images = da.bincount(event_locations, minlength=mul(*image_size))

    return images.astype(np.uint32).reshape(num_images, *image_size)
