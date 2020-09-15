# coding: utf-8

"""
Utilities for processing data from the Large Area Time-Resolved Detector

This module provides tools to interpret NeXus-like data in HDF5 format from the
experimental Timepix-based event-mode detector, codenamed Tristan, at Diamond Light
Source.
"""

from typing import Tuple, Union

import h5py
import numpy as np

_coordinate_type = Union[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]

# Translations of the cue_id messages.
cues = {
    0x000: "Padding",
    0x800: "Extended time stamp, global synchronisation signal",
    0x801: "Extended time stamp, sensor module 1",
    0x802: "Extended time stamp, sensor module 2",
    0x840: "Shutter open time stamp, global",
    0x841: "Shutter open time stamp, sensor module 1",
    0x842: "Shutter open time stamp, sensor module 2",
    0x880: "Shutter close time stamp, global",
    0x881: "Shutter close time stamp, sensor module 1",
    0x882: "Shutter close time stamp, sensor module 2",
    0x8C1: "FEM trigger input, falling edge",
    0x8E1: "FEM trigger input, rising edge",
    0x8C9: "Clock trigger TTL input, falling edge",
    0x8E9: "Clock trigger TTL input, rising edge",
    0x8CA: "Clock trigger LVDS input, falling edge",
    0x8EA: "Clock trigger LVDS input, rising edge",
    0xBC6: "Error: messages out of sync",
    0xBCA: "Error: messages out of sync",
    0xF00: "Reserved",
}

# Keys of event data in the HDF5 file structure.
cue_id_key = "entry/data/data/cue_id"
cue_time_key = "entry/data/data/cue_timestamp_zero"
event_location_key = "entry/data/data/event_id"
event_time_key = "entry/data/data/event_time_offset"
event_energy_key = "entry/data/data/event_energy"


def first_cue_time(data: h5py.File, message: int) -> int:
    """
    Find the timestamp of the first instance of a cue message in a data file.

    Args:
        data:  A NeXus-like LATRD data file.
        message:  The message code, as defined in the Tristan standard.

    Returns:
        The timestamp, measured in clock cycles from the global synchronisation signal.
    """
    index = np.argmax(data[cue_id_key][...] == message)
    return data[cue_time_key][index].astype(int)


def seconds(timestamp: int, reference: int = 0) -> float:
    """
    Convert a Tristan timestamp to seconds, measured from a given time.

    The time between the provided timestamp and a reference timestamp, both provided
    as a number of cock cycles from the same time origin, is converted to units of
    seconds.  By default, the reference timestamp is the beginning of the detector
    epoch (the global synchronisation signal).

    Args:
        timestamp:  A timestamp in number of clock cycles, to be converted to seconds.
        reference:  A reference time stamp in clock cycles.

    Returns:
        The difference between the two timestamps in seconds.
    """
    return (timestamp - reference) / 6.4e8


def coordinates(event_location: Union[int, np.ndarray]) -> _coordinate_type:
    """
    Extract pixel coordinate information from an event location message.

    The pixel coordinates of an event on a Tristan detector are encoded in an
    integer location message with 26 bits of useful information.  Extract the x
    coordinate (the 13 least significant bits) and the y coordinate (the 13 next
    least significant bits).

    Args:
        event_location:  Either a single event location message (an integer) or a NumPy
                         array of multiple integers representing the coordinates for
                         several events.

    Returns:
        A tuple (x, y) of decoded coordinates.
    """
    y, x = divmod(event_location, 0x2000)
    return x, y
