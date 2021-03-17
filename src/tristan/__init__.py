# coding: utf-8

"""
Utilities for processing data from the Large Area Time-Resolved Detector

This module provides tools to interpret NeXus-like data in HDF5 format from the
experimental Timepix-based event-mode detector, codenamed Tristan, at Diamond Light
Source.
"""

import os
from typing import MutableSequence, Optional, Tuple

import h5py
import numpy as np

clock_frequency = 6.4e8

# Translations of the cue_id messages.
padding = 0
sync = 0x800
sync_module_1 = 0x801
sync_module_2 = 0x802
shutter_open = 0x840
shutter_open_module_1 = 0x841
shutter_open_module_2 = 0x842
shutter_close = 0x880
shutter_close_module_1 = 0x881
shutter_close_module_2 = 0x882
fem_falling = 0x8C1
fem_rising = 0x8E1
ttl_falling = 0x8C9
ttl_rising = 0x8E9
lvds_falling = 0x8CA
lvds_rising = 0x8EA
reserved = 0xF00

cues = {
    padding: "Padding",
    sync: "Extended time stamp, global synchronisation signal",
    sync_module_1: "Extended time stamp, sensor module 1",
    sync_module_2: "Extended time stamp, sensor module 2",
    shutter_open: "Shutter open time stamp, global",
    shutter_open_module_1: "Shutter open time stamp, sensor module 1",
    shutter_open_module_2: "Shutter open time stamp, sensor module 2",
    shutter_close: "Shutter close time stamp, global",
    shutter_close_module_1: "Shutter close time stamp, sensor module 1",
    shutter_close_module_2: "Shutter close time stamp, sensor module 2",
    fem_falling: "FEM trigger input, falling edge",
    fem_rising: "FEM trigger input, rising edge",
    ttl_falling: "Clock trigger TTL input, falling edge",
    ttl_rising: "Clock trigger TTL input, rising edge",
    lvds_falling: "Clock trigger LVDS input, falling edge",
    lvds_rising: "Clock trigger LVDS input, rising edge",
    0xBC6: "Error: messages out of sync",
    0xBCA: "Error: messages out of sync",
    reserved: "Reserved",
}

# Keys of event data in the HDF5 file structure.
cue_id_key = "cue_id"
cue_time_key = "cue_timestamp_zero"
event_location_key = "event_id"
event_time_key = "event_time_offset"
event_energy_key = "event_energy"
size_key = "entry/instrument/detector/module/data_size"


def fullpath(path: str) -> str:
    """Get an absolute path with tilde home directory shorthand expanded."""
    return os.path.abspath(os.path.expanduser(path))


def first_cue_time(
    data: h5py.File, message: int, events_group: Optional[str] = "/"
) -> Optional[int]:
    """
    Find the timestamp of the first instance of a cue message in a data file.

    Args:
        data:  A NeXus-like LATRD data file.
        message:  The message code, as defined in the Tristan standard.
        events_group:  HDF5 group containing the events data.

    Returns:
        The timestamp, measured in clock cycles from the global synchronisation signal.
        If the message doesn't exist in the data set, this returns None.
    """
    events_group = events_group or "/"

    index = np.argmax(data[events_group + cue_id_key][...] == message)

    # Catch the case in which the message is not present in the data set.
    if index == 0 and data[events_group + cue_id_key][0] != message:
        return None

    return data[events_group + cue_time_key][index].astype(int)


def cue_times(
    data: h5py.File, message: int, events_group: Optional[str] = "/"
) -> MutableSequence[int]:
    """
    Find the timestamps of all instances of a cue message in a data file.

    Args:
        data:  A NeXus-like LATRD data file.
        message:  The message code, as defined in the Tristan standard.
        events_group:  HDF5 group containing the events data.

    Returns:
        The timestamps, measured in clock cycles from the global synchronisation signal.
    """
    events_group = events_group or "/"

    index = np.nonzero(data[events_group + cue_id_key][...] == message)
    return np.unique(data[events_group + cue_time_key][index].astype(int))


def seconds(timestamp: int, reference: int = 0) -> float:
    """
    Convert a Tristan timestamp to seconds, measured from a given time.

    The time between the provided timestamp and a reference timestamp, both provided
    as a number of cock cycles from the same time origin, is converted to units of
    seconds.  By default, the reference timestamp is zero clock cycles, the beginning
    of the detector epoch.

    Args:
        timestamp:  A timestamp in number of clock cycles, to be converted to seconds.
        reference:  A reference time stamp in clock cycles.

    Returns:
        The difference between the two timestamps in seconds.
    """
    return (timestamp - reference) / clock_frequency


def coordinates(event_location: int) -> Tuple[int, int]:
    """
    Extract pixel coordinate information from an event location message.

    The pixel coordinates of an event on a Tristan detector are encoded in an
    integer location message with 26 bits of useful information.  Extract the y
    coordinate (the 13 least significant bits) and the x coordinate (the 13 next
    least significant bits).

    Args:
        event_location:  Either a single event location message (an integer) or a NumPy
                         array of multiple integers representing the coordinates for
                         several events.

    Returns:
        A tuple (x, y) of decoded coordinates.
    """
    x, y = divmod(event_location, 0x2000)
    if isinstance(event_location, np.ndarray):
        return x.astype(np.int16), y.astype(np.int16)
    else:
        return x, y