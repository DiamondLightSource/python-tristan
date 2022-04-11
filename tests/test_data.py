"""Tests of utilities for handling LATRD Tristan data."""
from __future__ import annotations

from dataclasses import fields

import numpy as np
import pint
import pytest
from dask import array as da

from tristan.data import (
    cue_keys,
    cue_times,
    event_keys,
    first_cue_time,
    latrd_data,
    pixel_index,
    seconds,
)

from .conftest import random_range


def test_latrd_data_default_keys(dummy_data):
    """
    Test that the latrd_data context manager correctly reconstructs the dummy data.

    Construct a LATRD data dictionary from the dummy data files and check that there
    is an entry for each of the default keys, and that each entry has the expected
    values.
    """
    with latrd_data(sorted(dummy_data.iterdir())) as data:
        assert {field.name for field in fields(data)} == set(cue_keys + event_keys)
        # By using the same seed as when generating the test data,
        # we should be able to reproduce them.
        np.random.seed(0)
        for key in cue_keys + event_keys:
            da.assert_eq(getattr(data, key)[:10], np.random.randint(10, size=10))


def test_latrd_data_specified_keys(dummy_data):
    """Test that the latrd_data context manager reads only the specified keys."""
    with latrd_data(sorted(dummy_data.iterdir()), cue_keys) as data:
        assert {field.name for field in fields(data)} == set(cue_keys)


def test_first_cue_time(dummy_data):
    """Test the utility for finding the first timestamp of a given cue."""
    with latrd_data(sorted(dummy_data.iterdir())) as data:
        # Check that we can find the correct timestamp for a given cue message.
        assert first_cue_time(data, 0).compute() == 6

        # first_cue_time uses da.argmax, which can return zero either if the first
        # entry in a boolean array is True, or if no entry is.  We must check that we
        # distinguish these cases.
        # First, check that the timestamp is found correctly even if the first
        # instance of the desired cue message is the very first cue in the data.
        first_cue_message = data.cue_id_key[0]
        assert first_cue_time(data, first_cue_message).compute() == 7
        # Next, check that searching for a cue message that does not appear in the data
        # results in no returned timestamp.
        assert random_range not in data.cue_id
        assert first_cue_time(data, random_range) is None


def test_cue_times(dummy_data):
    """Test the utility for finding all timestamps of a given cue."""
    with latrd_data(sorted(dummy_data.iterdir())) as data:
        # The cue_id '3' appears four times in the test data,
        # with one duplicate timestamp.
        message = 3
        index = da.flatnonzero(data.cue_id == message).compute()
        assert np.all(data.cue_timestamp_zero[index] == (8, 8, 7, 9))
        # Check that cue_times finds and de-duplicates these timestamps.
        assert np.all(cue_times(data, message).compute() == (7, 8, 9))

        # Check that searching for a cue message that does not appear in the data
        # results in an empty array being returned.
        times_of_absent_cue = cue_times(data, random_range)
        times_of_absent_cue.compute_chunk_sizes()
        assert not times_of_absent_cue.size


def test_seconds():
    """Test the conversion of timestamp values to seconds."""
    assert seconds(640_000_000) == pint.Quantity(1, "s")
    assert seconds(640_000_000, 320_000_000) == pint.Quantity(0.5, "s")


@pytest.mark.parametrize("np_or_da", (np, da), ids=("NumPy", "Dask"))
def test_pixel_index(np_or_da):
    """
    Test the decoding of Tristan pixel coordinates.

    Check that both NumPy and Dask arrays are handled.

    For details of the pixel coordinate specification, see pixel_index.__doc__.
    """
    # Create some dummy coordinates.
    x_size = 10
    y_size = 20
    image_size = y_size, x_size
    x = np_or_da.arange(x_size) << 13
    y = np_or_da.arange(y_size)
    coords = (y[..., np.newaxis] + x).flatten()
    # Check that we decode the correct pixel indices.
    assert np.all(pixel_index(coords, image_size) == np.arange(x_size * y_size))


def test_single_pixel_index():
    """Test that pixel_index can decode a single pixel coordinate from an integer."""
    # Create a dummy coordinate.
    x_size = 10
    y_size = 20
    image_size = y_size, x_size
    x, y = 5, 10
    coord = (x << 13) + y
    # Check that the coordinate is correctly decoded.
    assert pixel_index(coord, image_size) == np.ravel_multi_index((y, x), image_size)
