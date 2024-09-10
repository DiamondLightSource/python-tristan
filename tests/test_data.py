"""Tests of utilities for handling LATRD Tristan data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pint
import pytest
from dask import dataframe as dd

from tristan.data import (
    cue_dtype,
    cue_id_key,
    cue_keys,
    cue_time_dtype,
    cue_time_key,
    cue_times,
    event_keys,
    event_location_key,
    first_cue_time,
    latrd_mf_data,
    pixel_index,
    pixel_index_key,
    seconds,
)

from .conftest import random_range


@pytest.mark.parametrize("keys", (cue_keys, event_keys), ids=("Cues", "Events"))
def test_latrd_mf_data_specified_keys(dummy_data, keys):
    """Test that the latrd_data context manager reads only the specified keys."""
    data = latrd_mf_data(sorted(dummy_data.iterdir()), keys)
    assert set(data.columns) == set(keys)


def test_first_cue_time(dummy_data):
    """Test the utility for finding the first timestamp of a given cue."""
    data = latrd_mf_data(sorted(dummy_data.iterdir()), cue_keys)
    # Check that we can find the correct timestamp for a given cue message.
    assert (first_cue_time(data, 0) == 6).compute().bool

    # first_cue_time uses da.argmax, which can return zero either if the first
    # entry in a boolean array is True, or if no entry is.  We must check that we
    # distinguish these cases.
    # First, check that the timestamp is found correctly even if the first
    # instance of the desired cue message is the very first cue in the data.
    first_cue_message = data[cue_id_key].head(1).item()
    assert (first_cue_time(data, first_cue_message) == 7).compute().bool
    # Next, check that searching for a cue message that does not appear in the data
    # results in no returned timestamp.
    assert random_range not in data.cue_id.values.compute()
    assert first_cue_time(data, random_range) is None


def test_cue_times(dummy_data):
    """Test the utility for finding all timestamps of a given cue."""
    data = latrd_mf_data(sorted(dummy_data.iterdir()), cue_keys)
    # The cue_id '3' appears six times in the test data,
    # with one duplicate timestamp.
    message = cue_dtype(3)
    index = data[cue_id_key] == message
    np.testing.assert_array_equal(
        data[cue_time_key][index].compute(), cue_time_dtype([7, 8, 0, 4, 2, 0])
    )
    # Check that cue_times finds and de-duplicates these timestamps.
    np.testing.assert_array_equal(
        cue_times(data, message).compute(), cue_time_dtype([0, 2, 4, 7, 8])
    )

    # Check that searching for a cue message that does not appear in the data
    # results in an empty array being returned.
    times_of_absent_cue = cue_times(data, cue_dtype(random_range))
    times_of_absent_cue.compute_chunk_sizes()
    assert not times_of_absent_cue.size


def test_seconds():
    """Test the conversion of timestamp values to seconds."""
    assert seconds(640_000_000) == pint.Quantity(1, "s")
    assert seconds(640_000_000, 320_000_000) == pint.Quantity(0.5, "s")


def test_pixel_index():
    """
    Test the decoding of Tristan pixel coordinates.

    Check that both NumPy and Dask arrays are handled.

    For details of the pixel coordinate specification, see pixel_index.__doc__.
    """
    # Create some dummy coordinates.
    x_size = 10
    y_size = 20
    image_size = y_size, x_size
    x = np.arange(x_size) << 13
    y = np.arange(y_size)
    coords = (y[..., np.newaxis] + x).flatten()
    dummy = pd.DataFrame({event_location_key: coords})

    # Check that we decode the correct pixel indices.
    decoded = pixel_index(dummy, image_size)[pixel_index_key]
    np.testing.assert_array_equal(decoded, np.arange(x_size * y_size))


@pytest.mark.parametrize(
    "pd_or_dd", (pd.DataFrame, dd.DataFrame.from_dict), ids=("Pandas", "Dask Dataframe")
)
def test_single_pixel_index(pd_or_dd):
    """Test that pixel_index can decode a single pixel coordinate from an integer."""
    # Create a dummy coordinate.
    x_size = 10
    y_size = 20
    image_size = y_size, x_size
    x, y = 5, 10
    coord = (x << 13) + y
    dummy = pd_or_dd({event_location_key: [coord]})

    # Check that the coordinate is correctly decoded.
    decoded = pixel_index(dummy, image_size)
    np.testing.assert_array_equal(
        decoded[pixel_index_key], np.ravel_multi_index((y, x), image_size)
    )
