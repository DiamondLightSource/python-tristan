"""Tools for binning events to images."""


from operator import mul
from typing import Dict, Tuple

import numpy as np
from dask import array as da

try:
    from numpy.typing import ArrayLike
except ImportError:
    # NumPy versions compatible with Python 3.6 do not have the numpy.typing module.
    ArrayLike = np.ndarray

from . import (
    event_location_key,
    event_time_key,
    first_cue_time,
    pixel_index,
    shutter_close,
    shutter_open,
)


def find_start_end(data):
    start_time = first_cue_time(data, shutter_open)
    end_time = first_cue_time(data, shutter_close)
    return da.compute(start_time, end_time)


def make_single_image(
    data: Dict[str, da.Array], image_size: Tuple[int, int], start: int, end: int
) -> da.Array:
    event_times = data[event_time_key]
    event_locations = data[event_location_key]

    valid_events = (start <= event_times) & (event_times < end)
    event_locations = event_locations[valid_events]
    event_locations = pixel_index(event_locations, image_size)

    image = da.bincount(event_locations, minlength=mul(*image_size))
    return image.astype(np.uint32).reshape(1, *image_size)


def make_multiple_images(
    data: Dict[str, da.Array], image_size: Tuple[int, int], bins: ArrayLike
) -> da.Array:
    event_times = data[event_time_key]
    event_locations = data[event_location_key]

    valid_events = (bins[0] <= event_times) & (event_times < bins[-1])
    event_times = event_times[valid_events]
    event_locations = event_locations[valid_events]

    image_indices = da.digitize(event_times, bins) - 1
    event_locations = pixel_index(event_locations, image_size)
    num_images = bins.size - 1

    image_indices = [
        image_indices == image_number for image_number in range(num_images)
    ]
    images = da.stack(
        [
            da.bincount(event_locations[indices], minlength=mul(*image_size))
            for indices in image_indices
        ]
    )

    return images.astype(np.uint32).reshape(num_images, *image_size)
