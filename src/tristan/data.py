"""Tools for extracting data on cues and events from Tristan data files."""

from __future__ import annotations

import re
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterable

import dask
import h5py
import numpy as np
from dask import array as da
from dask import dataframe as dd
from numpy.typing import ArrayLike
from pint import Quantity

from . import clock_frequency, ureg

# Regex for the names of data sets, in the time slice metadata file, representing the
# distribution of time slices across raw data files for each module.
ts_key_regex = re.compile(r"ts_qty_module\d{2}")

# Translations of the basic cue_id messages.
padding = np.uint16(0)
extended_timestamp = np.uint16(0x800)
shutter_open = np.uint16(0x840)
shutter_close = np.uint16(0x880)
fem_falling = np.uint16(0x8C1)
fem_rising = np.uint16(0x8E1)
ttl_falling = np.uint16(0x8C9)
ttl_rising = np.uint16(0x8E9)
lvds_falling = np.uint16(0x8CA)
lvds_rising = np.uint16(0x8EA)
tzero_falling = np.uint16(0x8CB)
tzero_rising = np.uint16(0x8EB)
sync_falling = np.uint16(0x8CC)
sync_rising = np.uint16(0x8EC)
reserved = np.uint16(0xF00)
cues = {
    padding: "Padding",
    extended_timestamp: "Extended time stamp, global synchronisation",
    shutter_open: "Shutter open time stamp, global",
    shutter_close: "Shutter close time stamp, global",
    fem_falling: "FEM trigger, falling edge",
    fem_rising: "FEM trigger, rising edge",
    ttl_falling: "Clock trigger TTL input, falling edge",
    ttl_rising: "Clock trigger TTL input, rising edge",
    lvds_falling: "Clock trigger LVDS input, falling edge",
    lvds_rising: "Clock trigger LVDS input, rising edge",
    tzero_falling: "Clock trigger TZERO input, falling edge",
    tzero_rising: "Clock trigger TZERO input, rising edge",
    sync_falling: "Clock trigger SYNC input, falling edge",
    sync_rising: "Clock trigger SYNC input, rising edge",
    np.uint16(0xBC6): "Error: messages out of sync",
    np.uint16(0xBCA): "Error: messages out of sync",
    reserved: "Reserved",
    **{
        basic + n: f"{name} time stamp, sensor module {n}"
        for basic, name in (
            (extended_timestamp, "Extended"),
            (shutter_open, "Shutter open"),
            (shutter_close, "Shutter close"),
        )
        for n in np.arange(1, 64, dtype=np.uint16)
    },
}

# Keys of cues and events data in the HDF5 file structure.
cue_id_key = "cue_id"
cue_time_key = "cue_timestamp_zero"
event_location_key = "event_id"
event_time_key = "event_time_offset"
event_energy_key = "event_energy"

cue_keys = cue_id_key, cue_time_key
event_keys = event_location_key, event_time_key, event_energy_key

nx_size_key = "entry/instrument/detector/module/data_size"


@contextmanager
def latrd_data(
    raw_file_paths: Iterable[str | Path],
    keys: Iterable[str] = cue_keys + event_keys,
) -> dd.DataFrame | dict[str, da.Array]:
    """
    A context manager to read LATRD data sets from multiple files.

    The yielded DataFrame has a column for each of the specified LATRD data keys.
    Each key must be a valid LATRD data key and the chosen data sets must all have the
    same length.  The data will be rechunked into partitions approximately the size of
    the default Dask array chunk size, but with chunk boundaries aligned with HDF5
    file boundaries.

    Args:
        raw_file_paths:  The paths of the raw LATRD data files.
        keys:  The set of LATRD data keys to be read.

    Yields:
        The data from all the files.
    """
    with ExitStack() as stack:
        files = [stack.enter_context(h5py.File(p, swmr=True)) for p in raw_file_paths]

        # Determine an appropriate block size for a Dask DataFrame of these data,
        # remembering to leave room for a 64-bit index.
        row_size = sum(files[0][k].dtype.itemsize for k in keys)
        block_size = ureg.Quantity(dask.config.get("array.chunk-size"))
        block_length = int(block_size.m_as("B") / row_size)

        # Construct a single Dask DataFrame from the specified keys.
        data = {
            k: da.concatenate([da.from_array(f[k], chunks=block_length) for f in files])
            for k in keys
        }
        data = dd.concat(
            [dd.from_dask_array(v, columns=k) for k, v in data.items()], axis=1
        )

        yield data


def first_cue_time(
    data: dd.DataFrame, message: int, after: int | None = None
) -> dd.DataFrame | None:
    """
    Find the timestamp of the first instance of a cue message in a Tristan data set.

    Args:
        data:     LATRD data.  Must contain one 'cue_id' column and one
                  'cue_timestamp_zero' column.  The two arrays are assumed to have the
                  same length.
        message:  The message code, as defined in the Tristan standard.
        after:    Ignore instances of the specified message before this timestamp.

    Returns:
        The timestamp, measured in clock cycles from the global synchronisation signal.
        If the message doesn't exist in the data set, this returns None.
    """
    message_incidences = data[cue_id_key] == message
    if after:
        message_incidences &= data[cue_time_key] >= after
    first_index = message_incidences.idxmax().compute()
    if first_index or data[cue_id_key].loc[0].compute().values == message:
        return data[cue_time_key].loc[first_index]


def cue_times(
    data: dd.DataFrame,
    message: int,
    after: int | None = None,
    before: int | None = None,
) -> da.Array:
    """
    Find the timestamps of all instances of a cue message in a Tristan data set.

    The found timestamps are de-duplicated.

    Args:
        data:     A DataFrame of LATRD data.  Must contain one column for cue id
                  messages and one for cue timestamps.
        message:  The message code, as defined in the Tristan standard.
        after:    Ignore instances of the specified message before this timestamp.

    Returns:
        The timestamps, measured in clock cycles from the global synchronisation
        signal, de-duplicated.
    """
    index = data[cue_id_key] == message
    if after:
        index &= data[cue_time_key] >= after
    if before:
        index &= data[cue_time_key] <= before
    return da.unique(data[cue_time_key][index].values)


def seconds(timestamp: int, reference: int = 0) -> Quantity:
    """
    Convert a Tristan timestamp to seconds, measured from a given reference timestamp.

    The time between the provided timestamp and a reference timestamp, both provided
    as a number of clock cycles from the same time origin, is converted to units of
    seconds.  By default, the reference timestamp is zero clock cycles, the beginning
    of the detector epoch.

    Args:
        timestamp:  A timestamp in number of clock cycles, to be converted to seconds.
        reference:  A reference time stamp in clock cycles.

    Returns:
        The difference between the two timestamps in seconds.
    """
    return ((timestamp - reference) / clock_frequency).to_base_units().to_compact()


def valid_events(data: dd.DataFrame, start: int, end: int) -> dd.DataFrame:
    """
    Return those events that have a timestamp in the specified range.

    Args:
        data:   LATRD data, containing an 'event_time_offset' column and optional
                'event_id' and 'event_energy' columns.
        start:  The start time of the accepted range, in clock cycles.
        end:    The end time of the accepted range, in clock cycles.

    Returns:
        The valid events.
    """
    valid = (start <= data[event_time_key]) & (data[event_time_key] < end)

    return data[valid]


def pixel_index(location: ArrayLike, image_size: tuple[int, int]) -> ArrayLike:
    """
    Extract pixel coordinate information from an event location (event_id) message.

    Translate a Tristan event location message to the index of the corresponding
    pixel in the flattened image array (i.e. numbered from zero, in row-major order).

    The pixel coordinates of an event on a Tristan detector are encoded in a 32-bit
    integer location message (the event_id) with 26 bits of useful information.
    Extract the y coordinate (the 13 least significant bits) and the x coordinate
    (the 13 next least significant bits).  Find the corresponding pixel index in the
    flattened image array by multiplying the y value by the size of the array in x,
    and adding the x value.

    This function calls the Python built-in divmod and so can be broadcast over
    array-like data structures.

    Args:
        location:    Event location message (an integer).
        image_size:  Shape of the image array in (y, x), i.e. (slow, fast).

    Returns:
        Index in the flattened image array of the pixel where the event occurred.
    """
    x, y = divmod(location, np.uint32(0x2000))
    # The following is equivalent to, but a little simpler than,
    # return da.ravel_multi_index((y, x), image_size)
    return x + y * image_size[1]
