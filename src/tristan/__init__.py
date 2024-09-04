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

from contextlib import ContextDecorator

import dask
import pint
from dask.distributed import Client, progress, wait

ureg = pint.UnitRegistry()

clock_frequency = ureg.Quantity(6.4e8, "Hz").to_compact()


class WithLocalDistributedCluster(Client, ContextDecorator):
    """
    A decorator to run a function in a distributed.Client context.

    Example:
        Using this decorator like so

        >>> @WithLocalDistributedCluster(processes=False)
        ... def foo(*args):
        ...     ...

        is equivalent to

        >>> def foo(*args):
        ...     with Client(processes=False):
        ...         ...
    """


def compute_with_progress(*collection, gather=False):
    """
    Compute Dask collections, showing a progress bar, assuming a distributed client.

    Args:
        collection:  A Dask object or built-in collection of objects.
        gather:      If true, return the computed result.
    """
    collection = dask.persist(*collection)
    print(progress(collection) or "")

    wait(collection)

    return dask.compute(*collection, sync=True) if gather else None
