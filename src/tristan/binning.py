"""Tools for binning events to images."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import sparse
from dask import array as da
from dask import dataframe as dd
from numpy.typing import ArrayLike, NDArray

from .data import event_time_key, image_dtype, pixel_index, time_bin_key, valid_events
from .storage import IAddArray


def align_bins(start: int, align: int, end: int, n_bins: int) -> tuple[int, int]:
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


def find_preceding_bin_edge_index(a: NDArray, bins: NDArray) -> NDArray:
    """
    From sorted bin edges, find the index of the bin edge preceding each value in an
    array.

    Args:
        a:     An array of values for which we seek the the indices in ``bin`` of the
               preceding bin edges.
        bins:  A sorted array of bin edges.

    Returns:
        An array with the same length as ``a``.  Each entry is the index in ``bins`` of
        the bin edge immediately preceding the corresponding entry in ``a``.
    """
    return np.digitize(a, bins) - 1


def find_preceding_bin_edge(a: NDArray, bins: NDArray) -> NDArray:
    """
    From sorted bin edges, find the bin edge preceding each value in an array.

    Args:
        a:     An array of values for which corresponding preceding bin edges are
               sought.
        bins:  A sorted array of bin edges.

    Returns:
        An array with the same length as ``a``.  Each entry is the bin edge immediately
        preceding the corresponding entry in ``a``.
    """
    return bins[find_preceding_bin_edge_index(a, bins)]


def find_time_bins(data: pd.DataFrame, bins: Sequence[int]):
    """
    Convert the event timestamps in LATRD data to time bin indices.

    For each event, determine the index of the bin into which the event will fall.

    Args:
        data:  LATRD events data.  Must have an ``event_time_offset`` column.
        bins:  The time bin edges of the images (in clock cycles, to match the event
               timestamps).

    Returns:
        A DataFrame which matches the input data except that the ``event_time_offset``
        column is replaced with a column of ``time_bin`` indices.
    """
    num_images = len(bins) - 1

    # Find the index of the image to which each event belongs.
    if num_images > 1:
        data[event_time_key] = np.digitize(data[event_time_key], bins) - 1
    elif num_images:
        data[event_time_key] = 0

    return data.rename(columns={event_time_key: time_bin_key})


def make_images(coords: ArrayLike, shape: tuple[int, ...]) -> sparse.COO:
    """
    Bin LATRD events data into images of event counts.

    Given an array of events data, and a known shape of the image or stack of images to
    which the events should be binned, bin the events to images, with pixel intensities
    recording the number of events recorded at each pixel.

    ``shape`` should have length ``n_dims`` ≥ 2, representing the dimensions of the
    image or stack of images.  Events will first be binned into a (stack of) flattened
    image(s), i.e. having dimensions ``(*shape[:-2], shape[-2] * shape[-1])`` before
    being reshaped to ``shape``.

    ``coords`` should have shape ``(n_dims - 1, n_events)``, where ``n_events`` is the
    number of events.  The last column, ``coords[-1]``, should contain each event's
    pixel index in the flattened array of the image (i.e. for a 20px × 30px image, the
    values would run from 0 to 599).

    Args:
        coords:  The coordinates of each event.
        shape:   The shape of the image or image stack.
    """
    # Get the shape of the image stack with the image axes flattened.
    *slower_dims, y, x = shape
    flattened_shape = *slower_dims, x * y
    # Assign these events to pixels in the image stack, with the image axes flattened.
    image_data = sparse.COO(coords=coords, data=image_dtype(1), shape=flattened_shape)
    # Reshape the image data to the true shape of the image stack.
    return image_data.reshape(shape)


def cache_array(array: ArrayLike | sparse.COO, cache: ArrayLike) -> None:
    """
    Cache the contents of ``array`` in ``cache`` by addition.

    If ``array`` is a ``sparse.COO``, write to the cache more efficiently by writing
    only the non-null data.

    Args:
        array:  Array to be added to the cache.  This may be sparse.
        cache:  Cache to which ``array`` should be added.  This must be dense.  This may
                be a Zarr array or HDF5 dataset, if an on-disk cache is necessary.  If a
                tristan.storage.IAddArray, the inplace addition sycnhronisation method
                will be used.
    """
    # If cache is an IAddArray, use thread-safe inplace addition.
    if isinstance(array, sparse.COO):
        index = tuple(array.coords)
        array = array.data
        if isinstance(cache, IAddArray):
            cache.iadd_coordinate_selection(index, array)
        else:
            cache[index] += array
    else:
        cache[...] += array

    return None


def event_block_to_image_cache(
    coords: ArrayLike, shape: tuple[int, ...], cache: ArrayLike
):
    """
    Bin a chunk of events to a partial image or image stack and cache the result.

    Args:
        coords:  The coordinates of each event.
        shape:   The shape of the image or image stack.
        cache:   Cache to which ``array`` should be added.  This must be dense.  This may
                 be a Zarr array or HDF5 dataset, if an on-disk cache is necessary.  If a
                 tristan.storage.IAddArray, the inplace addition sycnhronisation method
                 will be used.
    """
    return cache_array(make_images(coords, shape), cache)


def events_to_images(
    data: dd.DataFrame, bins: Sequence[int], shape: tuple[int, ...], cache: ArrayLike
) -> da.Array:
    """
    Construct a stack of images from events data.

    From a sequence of LATRD events data, bin the events to images and store the binned
    images in a cache array.  The cache may be backed with on-disk storage, as in the
    case of a Zarr array, or may be a simple in-memory object, like a NumPy array.

    Args:
        data:   LATRD events data.  Must have columns ``event_id`` and
                ``event_time_offset``.  Column order must be the order of dimensions in
                the resulting image array, i.e. ``(..., event_time_offset, event_id)``
                where ``...`` are optional extra columns.  The number of columns should
                be one fewer than the length of ``shape`` becasue ``event_id`` is
                decoded into two dimensions, the slow and fast axes of the image.
        bins:   The time bin edges of the images (in clock cycles, to match the event
                timestamps).
        shape:  The shape of the image stack.
        cache:  An array representing the eventual image stack, having shape
                ``(len(bins) - 1, *image_size)``, to which the pixel counts from this
                binning operation will be added.

    Returns:
        A Dask collection representing the lazy image binning computation.
    """
    # Consider only those events that occur between the start and end times.
    data = valid_events(data, bins[0], bins[-1])
    # Convert the event IDs to indices of pixels in the flattened array.
    data = pixel_index(data, shape[-2:])

    # Metadata for mapping find_time_bins across partitions.
    columns = [
        time_bin_key if column == event_time_key else column for column in data.columns
    ]
    dtypes = data.dtypes
    dtypes[time_bin_key] = dtypes.pop(event_time_key)
    meta = pd.DataFrame(columns=columns).astype(dtype=dtypes)
    # Determine time bins.
    data = data.map_partitions(find_time_bins, bins=bins, meta=meta)

    # Dummy metadata for dask.array.map_blocks.
    empty_coords = np.empty(shape=(len(shape), 0), dtype=int)
    empty_coo = sparse.COO(empty_coords, data=image_dtype(()), shape=shape)

    # Bin to images, partition by partition.
    coords = data.values.T
    return coords.map_blocks(
        event_block_to_image_cache, shape=shape, cache=cache, meta=empty_coo
    )
