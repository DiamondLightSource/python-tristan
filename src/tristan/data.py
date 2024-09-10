"""Tools for extracting data on cues and events from Tristan data files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr
from dask import array as da
from dask import dataframe as dd
from pint import Quantity

from . import clock_frequency

# Keys of cues and events data in the HDF5 file structure.
cue_id_key = "cue_id"
cue_time_key = "cue_timestamp_zero"
event_location_key = "event_id"
event_time_key = "event_time_offset"
event_energy_key = "event_energy"
# Key for pixel index data, converted from event_id to index in flattened image array.
pixel_index_key = "pixel_index"

cue_keys = cue_id_key, cue_time_key
event_keys = event_location_key, event_time_key, event_energy_key

# Types of the raw data.
cue_dtype = np.uint16
cue_time_dtype = np.uint64
event_id_dtype = np.uint32
event_time_dtype = np.uint64
event_energy_dtype = np.uint32

# Type for binned image data.
image_dtype = np.uint32

# Translations of the basic cue_id messages.
padding = cue_dtype(0)
extended_timestamp = cue_dtype(0x800)
shutter_open = cue_dtype(0x840)
shutter_close = cue_dtype(0x880)
fem_falling = cue_dtype(0x8C1)
fem_rising = cue_dtype(0x8E1)
ttl_falling = cue_dtype(0x8C9)
ttl_rising = cue_dtype(0x8E9)
lvds_falling = cue_dtype(0x8CA)
lvds_rising = cue_dtype(0x8EA)
tzero_falling = cue_dtype(0x8CB)
tzero_rising = cue_dtype(0x8EB)
sync_falling = cue_dtype(0x8CC)
sync_rising = cue_dtype(0x8EC)
reserved = cue_dtype(0xF00)
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
    cue_dtype(0xBC6): "Error: messages out of sync",
    cue_dtype(0xBCA): "Error: messages out of sync",
    reserved: "Reserved",
    **{
        basic + n: f"{name} time stamp, sensor module {n}"
        for basic, name in (
            (extended_timestamp, "Extended"),
            (shutter_open, "Shutter open"),
            (shutter_close, "Shutter close"),
        )
        for n in np.arange(1, 64, dtype=cue_dtype)
    },
}

# The event_id represents the coordinates of the pixel at which the event was recorded.
# The n least significant bits represent the y coordinate and the n next least
# significant bits represent the x coordinate.  Record the value of n here as
# coordinate_bitdepth.
coordinate_bitdepth = 13

# Key for the image shape data set in the input NeXus file.
nx_size_key = "entry/instrument/detector/module/data_size"

# Regex for the names of data sets, in the time slice metadata file, representing the
# distribution of time slices across raw data files for each module.
ts_key_regex = re.compile(r"ts_qty_module\d{2}")

# Tristan data contain some junk data sets.  Ignore them when reading data files.
ignored_datasets = ["data", "image", "raw_data"]


def latrd_data(path: str | Path, keys: Iterable[str]) -> dd.DataFrame:
    """
    Read LATRD data sets from a file of raw Tristan events data.

    The returned DataFrame has a column for each of the specified LATRD data keys. Each
    key must be a valid LATRD data key and the corresponding data sets must all have the
    same length.

    Args:
        paths:  The path to the raw LATRD data file.
        keys:   The set of LATRD data keys to read.

    Returns:
        The data from all the files.
    """
    data = xr.open_dataset(path, chunks="auto", drop_variables=ignored_datasets)
    return data[list(keys)].unify_chunks().to_dask_dataframe()[list(keys)]


def latrd_mf_data(paths: Iterable[str | Path], keys: Iterable[str]) -> dd.DataFrame:
    """
    Read LATRD data sets from multiple files of raw Tristan events data.

    The returned DataFrame has a column for each of the specified LATRD data keys. Each
    key must be a valid LATRD data key and the corresponding data sets must all have the
    same length.

    Args:
        paths:  The paths to the raw LATRD data files.
        keys:   The set of LATRD data keys to read.

    Returns:
        The data from all the files.
    """
    return dd.concat([latrd_data(path, keys) for path in paths], axis="index")


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
    if first_index or data[cue_id_key].head(1).item() == message:
        return data[cue_time_key].loc[first_index]


def cue_times(
    data: dd.DataFrame,
    message: cue_dtype,
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


def pixel_index(
    data: pd.DataFrame | dd.DataFrame, image_size: tuple[int, int]
) -> pd.DataFrame | dd.DataFrame:
    """
    Extract pixel coordinate information from an event location (event_id) message.

    Translate a Tristan event location message to the index of the corresponding pixel
    in the flattened image array (i.e. numbered from zero, in row-major order).

    The pixel coordinates of an event on a Tristan detector are encoded in a 32-bit
    integer location message (the event_id) with 2n bits of useful information. Extract
    the y coordinate (the n least significant bits) and the x coordinate (the n next
    least significant bits).  The value of n is recorded as
    ``tristan.data.coordinate_bitdepth`` and is usually 13. Find the corresponding pixel
    index in the flattened image array by multiplying the y value by the size of the
    array in x, and adding the x value.

    In the resulting dataframe, the ``"event_id""`` column is replaced with a column
    named ``"pixel_index"``, with values, encoding the positions of each pixel in the
    flattened image array.

    Args:
        data:        A Dask DataFrame, having a column named ``event_location_key``.
        image_size:  Shape of the image array in (y, x), i.e. (slow, fast).

    Returns:
        A Dask DataFrame like the input ``data`` but having the new ``"pixel_index"``
        column in place of the old ``"event_id"`` column.
    """
    x, y = divmod(data[event_location_key], event_id_dtype(1 << coordinate_bitdepth))
    # The following is equivalent to, but a little simpler than,
    # data[event_location_key] = da.ravel_multi_index((y, x), image_size)
    data[event_location_key] = x + y * image_size[1]

    return data.rename(columns={event_location_key: pixel_index_key})
