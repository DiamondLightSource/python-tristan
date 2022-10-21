"""
Utilities for processing data from the Large Area Time-Resolved Detector

This module provides tools to interpret NeXus-like data in HDF5 format from the
experimental Timepix-based event-mode detector, codenamed Tristan, at Diamond Light
Source.
"""

from __future__ import annotations

__author__ = "Diamond Light Source — Data Analysis Group"
__email__ = "dataanalysis@diamond.ac.uk"
__version__ = "0.1.16"
__version_tuple__ = tuple(int(x) for x in __version__.split("."))

import dask
import pint
from dask import array as da
from dask.distributed import progress, wait

ureg = pint.UnitRegistry()

clock_frequency = ureg.Quantity(6.4e8, "Hz").to_compact()


def compute_with_progress(collection):
    """
    Compute a Dask collection, showing the progress of the top layer of the task graph.

    Args:
        collection:  A single Dask collection.
    """
    (collection,) = dask.persist(collection)

    # View progress only of the top layer of the task graph, which consists
    # of the rate limiting make_images tasks, to avoid giving a false sense
    # of rapid progress from the quick execution of the large number of
    # other, cheaper tasks.
    *_, top_layer = collection.dask.layers.values()
    futures = list(top_layer.values())
    print(progress(futures) or "")

    wait(collection)


def blockwise_selection(array: da.Array, selection: da.Array) -> da.Array:
    """
    Select from an array in a blockwise fashion, without computing chunk sizes.

    Slicing a dask.Array with an array of bools or indices returns an array with
    unknown chunk sizes, which causes problems for downstream Dask operations.  If
    this is not a concern, (for example, if we are simply lumping the resulting array
    into a bincount call, and have no need of the chunk size information),
    we can save ourselves an expensive pass over the data by not computing the chunk
    sizes at all.  This can be achieved by replacing Dask slicing with NumPy slicing,
    mapped over all the blocks of an array.

    Args:
        array:      The array from which to take the selection.
        selection:  An array of dtype bool, or an array of indices, specifying the
                    desired selection from array.  There must be the same number of
                    chunks as in array, and if the dtype is bool, the chunk sizes
                    must match too.

    Returns:
        A slice from 'array', with the same dtype, and with the shape of 'selection'.
    """
    return da.map_blocks(lambda b, a: a[b], selection, array, dtype=array.dtype)
