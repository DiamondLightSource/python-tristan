"""Tools for binning events to images."""


from operator import mul
from typing import Dict, Tuple

import numpy as np
from dask import array as da
from dask.distributed import progress
from numpy.typing import ArrayLike

from .data import (
    event_location_key,
    event_time_key,
    first_cue_time,
    pixel_index,
    shutter_close,
    shutter_open,
)


def find_start_end(data: Dict[str, da.Array], distributed: bool = False) -> (int, int):
    """
    Find the shutter open and shutter close timestamps.

    Args:
        data:  A LATRD data dictionary (a dictionary with data set names as keys
               and Dask arrays as values).  Must contain one entry for cue id
               messages and one for cue timestamps.  The two arrays are assumed
               to have the same length.
        distributed:  Whether the computation uses the Dask distributed scheduler,
                      in which case a progress bar will be displayed.

    Returns:
        The shutter open and shutter close timestamps, in clock cycles.
    """
    start_time = first_cue_time(data, shutter_open)
    end_time = first_cue_time(data, shutter_close)

    # If we are using the distributed scheduler (for multiple images), show progress.
    if distributed:
        print("\nFinding detector shutter open and close times.")
        progress([start_time.persist(), end_time.persist()])
    start_time, end_time = da.compute(start_time, end_time)
    print("")

    return start_time, end_time


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
    event_times = event_times[valid_events].compute_chunk_sizes()
    event_locations = event_locations[valid_events].compute_chunk_sizes()
    # We need to ensure that the chunk layout of the event location array matches
    # that of the event time array, so that we can perform matching blockwise iterations
    event_locations = event_locations.rechunk(event_times.chunks)
    event_locations = pixel_index(event_locations, image_size)

    num_images = len(bins) - 1

    if num_images > 1:
        # We cannot perform a single bincount of the entire data set because that
        # would require allocating enough memory for the entire image stack.

        # Find the index of the image to which each event belongs.
        image_indices = da.digitize(event_times, bins) - 1

        (images_in_block,) = da.compute(map(np.unique, image_indices.blocks))

        # Construct a stack of images using dask.array.bincount.
        images = []
        for i in range(num_images):
            # When searching for events with a given image index, we already know we
            # can exclude some blocks and thereby save some computation time.
            contains_index = [i in indices for indices in images_in_block]

            image_events = event_locations.blocks[contains_index][
                image_indices.blocks[contains_index] == i
            ]
            images.append(da.bincount(image_events, minlength=mul(*image_size)))

        images = da.stack(images)
    else:
        images = da.bincount(event_locations, minlength=mul(*image_size))

    return images.astype(np.uint32).reshape(num_images, *image_size)
