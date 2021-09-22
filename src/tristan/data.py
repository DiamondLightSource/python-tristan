"""Tools for extracting data on cues and events from Tristan data files."""

import re
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import h5py
from dask import array as da
from dask import config
from numpy.typing import ArrayLike
from pint import Quantity

from . import clock_frequency

# Type alias for collections of raw file paths.
RawFiles = Iterable[Union[str, Path]]

# Regex for the names of data sets, in the time slice metadata file, representing the
# distribution of time slices across raw data files for each module.
ts_key_regex = re.compile(r"ts_qty_module\d{2}")


# Translations of the basic cue_id messages.
padding = 0
sync = 0x800
shutter_open = 0x840
shutter_close = 0x880
fem_falling = 0x8C1
fem_rising = 0x8E1
ttl_falling = 0x8C9
ttl_rising = 0x8E9
lvds_falling = 0x8CA
lvds_rising = 0x8EA
tzero_falling = 0x8CB
tzero_rising = 0x8EB
sync_falling = 0x8CC
sync_rising = 0x8EC
reserved = 0xF00
cues = {
    padding: "Padding",
    sync: "Extended time stamp, global synchronisation",
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
    0xBC6: "Error: messages out of sync",
    0xBCA: "Error: messages out of sync",
    reserved: "Reserved",
    **{
        basic + n: f"{name} time stamp, sensor module {n}"
        for basic, name in (
            (sync, "Extended"),
            (shutter_open, "Shutter open"),
            (shutter_close, "Shutter close"),
        )
        for n in range(1, 64)
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


def aggregate_chunks(
    existing_chunks: Iterable[int], item_size: int, subdivision: int = 1
):
    target_size_bytes = int(Quantity(config.get("array.chunk-size")).m_as("bytes"))

    # The optimal number of data per Dask chunk.
    target_size = target_size_bytes // item_size

    # Try to aggregate the input data into the fewest possible Dask chunks.
    new_chunks = []
    for chunk in existing_chunks:
        # If this input data set will fit into the current chunk, add it.
        if new_chunks and new_chunks[-1] + chunk <= target_size:
            new_chunks[-1] += chunk
        # If the current chunk is full (or the chunks list is empty), add this
        # data set to the next chunk.
        elif chunk <= target_size:
            new_chunks.append(chunk)
        # If this data set is larger than the max Dask chunk size, split it
        # along the HDF5 data set chunk boundaries and put the pieces in
        # separate Dask chunks.
        else:
            n_whole_chunks, remainder = divmod(chunk, target_size)
            dask_chunk_size = target_size // subdivision * subdivision
            new_chunks += [dask_chunk_size] * n_whole_chunks + [remainder]

    return new_chunks


@contextmanager
def latrd_data(
    raw_file_paths: RawFiles, keys: Iterable[str] = cue_keys + event_keys
) -> Dict[str, da.Array]:
    """
    A context manager to read LATRD data sets from multiple files.

    The yielded dictionary has an entry for each of the specified LATRD data keys.
    Each key must be a valid LATRD data key and the corresponding value is a Dask
    array containing the corresponding LATRD data from all the raw data files,
    rechunked into blocks approximately the size of the default Dask array chunk
    size, but with chunk boundaries aligned with HDF5 data set chunk boundaries.

    Args:
        raw_file_paths:  The paths of the raw LATRD data files.
        keys:  The set of LATRD data keys to be read.

    Yields:
        A dictionary of LATRD data keys and arrays of the corresponding data.
    """
    with ExitStack() as stack:
        files = [
            stack.enter_context(h5py.File(path, "r", swmr=True))
            for path in raw_file_paths
        ]

        data = {}
        for key in keys:
            data_sets = [f[key] for f in files]

            sizes = (d.size for d in data_sets)
            hdf5_chunk_size = (data_sets[0].chunks or (data_sets[0].size,))[0]
            chunks = aggregate_chunks(
                sizes, data_sets[0].dtype.itemsize, hdf5_chunk_size
            )

            data[key] = da.concatenate(data_sets).rechunk(chunks)

        yield data


def first_cue_time(data: Dict[str, da.Array], message: int) -> Optional[da.Array]:
    """
    Find the timestamp of the first instance of a cue message in a Tristan data set.

    Args:
        data:     A LATRD data dictionary (a dictionary with data set names as keys
                  and Dask arrays as values).  Must contain one entry for cue id
                  messages and one for cue timestamps.  The two arrays are assumed
                  to have the same length.
        message:  The message code, as defined in the Tristan standard.

    Returns:
        The timestamp, measured in clock cycles from the global synchronisation signal.
        If the message doesn't exist in the data set, this returns None.
    """
    index = da.argmax(data[cue_id_key] == message)
    if index or data[cue_id_key][0] == message:
        return data[cue_time_key][index]


def cue_times(data: Dict[str, da.Array], message: int) -> da.Array:
    """
    Find the timestamps of all instances of a cue message in a Tristan data set.

    The found timestamps are de-duplicated.

    Args:
        data:     A LATRD data dictionary (a dictionary with data set names as keys
                  and Dask arrays as values).  Must contain one entry for cue id
                  messages and one for cue timestamps.  The two arrays are assumed
                  to have the same length.
        message:  The message code, as defined in the Tristan standard.

    Returns:
        The timestamps, measured in clock cycles from the global synchronisation
        signal, de-duplicated.
    """
    index = da.flatnonzero(data[cue_id_key] == message)
    return da.unique(data[cue_time_key][index])


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


def pixel_index(location: ArrayLike, image_size: Tuple[int, int]) -> ArrayLike:
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

    This function calls the Python built-in divmod and so can be broadcast over NumPy
    and Dask arrays.

    Args:
        location:    Event location message (an integer).
        image_size:  Shape of the image array in (y, x), i.e. (slow, fast).

    Returns:
        Index in the flattened image array of the pixel where the event occurred.
    """
    x, y = divmod(location, 0x2000)
    # The following is equivalent to, but a little simpler than,
    # return da.ravel_multi_index((y, x), image_size)
    return x + y * image_size[1]
