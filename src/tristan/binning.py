"""Tools for binning events to images."""
from __future__ import annotations

from functools import partial
from itertools import zip_longest
from operator import mul

import dask
import numpy as np
from dask import array as da
from dask.array.reductions import _tree_reduce
from dask.array.routines import array_safe
from dask.array.utils import meta_from_array
from dask.base import tokenize
from dask.diagnostics import ProgressBar
from dask.distributed import progress
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

    # If we are using the distributed scheduler (for multiple images), show progress.
    if show_progress:
        try:
            print(progress(start_index.persist(), end_index.persist()) or "")
            start_index, end_index = da.compute(start_index, end_index)
        except ValueError:  # No client found when using the default scheduler.
            with ProgressBar():
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
        value = data.getattr(key)
        if value is not None:
            value = value.rechunk(data.event_time_offset.chunks)
            setattr(data, key, blockwise_selection(value, valid))

    return data


def _bincount_block_to_image(
    x: ArrayLike,
    weights: ArrayLike | None = None,
    shape: tuple[int, int] | None = None,
    dtype: type | None = None,
):
    minlength = mul(*shape)
    binned = np.bincount(x, weights=weights, minlength=minlength)
    return binned[:minlength].reshape(shape).astype(dtype)


def _bincount_to_image_agg(bincounts: da.Array, **kwargs):
    if not isinstance(bincounts, list):
        return bincounts

    return np.sum(bincounts, axis=0)


def bincount_to_image(
    x: da.Array,
    shape: tuple[int, int],
    weights: da.Array | None = None,
    split_every: int | None = None,
):
    if x.ndim != 1:
        raise ValueError("Input array must be one dimensional. Try using x.ravel()")
    if weights is not None:
        if weights.chunks != x.chunks:
            raise ValueError("Chunks of input array x and weights must match.")

    split_every = split_every or dask.config.get("split_every", 4)
    grouped_chunks = zip_longest(*[iter(x.chunks[0])] * split_every)
    grouped_chunks = map(partial(filter, lambda a: a is not None), grouped_chunks)
    grouped_chunks = tuple(map(sum, grouped_chunks))
    x.rechunk(grouped_chunks)

    token = tokenize(x, weights, shape)
    args = [x, "i"]
    if weights is not None:
        meta = array_safe(np.bincount([1], weights=[1]), like=meta_from_array(x))
        meta = meta.astype(np.float32)
        args.extend([weights, "i"])
    else:
        meta = array_safe(np.bincount([]), like=meta_from_array(x)).astype(np.uint32)

    chunked_counts = da.blockwise(
        partial(_bincount_block_to_image, shape=shape, dtype=meta.dtype),
        "ij",
        *args,
        new_axes={"j": shape[1]},
        dtype=meta.dtype,
        token=token,
        meta=meta,
    )
    chunked_counts._chunks = ((shape[0],) * len(chunked_counts.chunks[0]), (shape[1],))

    output = _tree_reduce(
        chunked_counts,
        aggregate=_bincount_to_image_agg,
        axis=(0,),
        keepdims=True,
        dtype=meta.dtype,
        split_every=split_every,
        concatenate=False,
    )
    output._chunks = ((shape[0],), (shape[1],))
    output._meta = meta
    return output


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
            images.append(bincount_to_image(image_events, shape=image_size))

        images = da.stack(images)
    else:
        images = bincount_to_image(event_locations, shape=image_size)[np.newaxis]

    return images
