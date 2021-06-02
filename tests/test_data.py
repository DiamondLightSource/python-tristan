"""Tests of utilities for handling LATRD Tristan data."""

import h5py
import numpy as np
from dask.array import assert_eq
from pytest import fixture

from tristan.data import cue_keys, event_keys, latrd_data

# A set of arbitrary offsets to distinguish the dummy data sets from one another.
offsets = dict(zip(cue_keys + event_keys, range(0, 500, 100)))


@fixture(scope="session")
def dummy_latrd_data(tmp_path_factory):
    """
    Construct a temporary directory containing dummy LATRD data.

    The data are spread across several files, in the manner of a LATRD data
    collection, with each file having an entry for each of the LATRD data keys.
    """
    tmp_path = tmp_path_factory.mktemp("dummy_data", numbered=False)
    ranges = np.arange(15).reshape(3, 5)
    for i, values in enumerate(ranges):
        with h5py.File(tmp_path / f"dummy_{i:06d}.h5", "w") as f:
            for key, offset in offsets.items():
                f.create_dataset(key, data=values + offset)

    return tmp_path


def test_latrd_data_default_keys(dummy_latrd_data):
    """
    Test that the latrd_data context manager correctly reconstructs the dummy data.

    Construct a LATRD data dictionary from the dummy data files and check that there
    is an entry for each of the default keys, and that each entry has the expected
    values.
    """
    with latrd_data(sorted(dummy_latrd_data.iterdir())) as data:
        assert set(data.keys()) == set(cue_keys + event_keys)
        for key, offset in offsets.items():
            assert_eq(data[key], np.arange(15) + offset)


def test_latrd_data_specified_keys(dummy_latrd_data):
    """Test that the latrd_data context manager reads only the specified keys."""
    with latrd_data(sorted(dummy_latrd_data.iterdir()), cue_keys) as data:
        assert set(data.keys()) == set(cue_keys)
