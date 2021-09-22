# coding: utf-8

"""
Utilities for processing data from the Large Area Time-Resolved Detector

This module provides tools to interpret NeXus-like data in HDF5 format from the
experimental Timepix-based event-mode detector, codenamed Tristan, at Diamond Light
Source.
"""

__author__ = "Diamond Light Source - Scientific Software"
__email__ = "scientificsoftware@diamond.ac.uk"
__version__ = "0.1.12"
__version_tuple__ = tuple(int(x) for x in __version__.split("."))

import pint

ureg = pint.UnitRegistry()

clock_frequency = ureg.Quantity(6.4e8, "Hz").to_compact()
