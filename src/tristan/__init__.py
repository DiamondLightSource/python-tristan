"""
Utilities for processing data from the Large Area Time-Resolved Detector

This module provides tools to interpret NeXus-like data in HDF5 format from the
experimental Timepix-based event-mode detector, codenamed Tristan, at Diamond Light
Source.
"""

from __future__ import annotations

__author__ = "Diamond Light Source — Data Analysis Group"
__email__ = "dataanalysis@diamond.ac.uk"
__version__ = "0.3.2"
__version_tuple__ = tuple(int(x) for x in __version__.split("."))

import dask
import pint
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
