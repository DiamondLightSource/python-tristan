"""Tests of utilities for handling LATRD Tristan data."""
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import pint
import pytest
from dask import array as da

from tristan.data import (
    cue_id_key,
    cue_keys,
    cue_time_key,
    cue_times,
    data_files,
    event_keys,
    find_input_file_name,
    first_cue_time,
    latrd_data,
    pixel_index,
    seconds,
)

random_range = 10


@contextmanager
def dummy_latrd_data(path_factory):
    """

    Construct a temporary directory containing dummy LATRD data.

    The data are spread across several files, in the manner of a LATRD data
    collection, with each file having an entry for each of the LATRD data keys.

    Args:
        path_factory:  The Pytest tmp_path_factory fixture.

    Yields:
        A temporary directory containing dummy data files.
    """
    tmp_path = path_factory.mktemp("dummy_data")
    # Seed for a consistent pseudo-random array.
    np.random.seed(0)
    all_values = np.random.randint(random_range, size=150).reshape(3, 5, 10)
    for i, values in enumerate(all_values, 1):
        with h5py.File(tmp_path / f"dummy_{i:06d}.h5", "w") as f:
            f.update(dict(zip(cue_keys + event_keys, values)))

    yield tmp_path


@pytest.fixture(scope="session")
def dummy_data(tmp_path_factory):
    """A session-scoped dummy LATRD data fixture."""
    with dummy_latrd_data(tmp_path_factory) as data_path:
        yield data_path


@pytest.fixture(scope="function")
def dummy_data_transient(tmp_path_factory):
    """A function-scoped dummy LATRD data fixture."""
    with dummy_latrd_data(tmp_path_factory) as data_path:
        yield data_path


@pytest.mark.parametrize("directory", (".", "/", "~", "test_dir"))
@pytest.mark.parametrize("stem", ("dummy_meta", "dummy_1", "dummy_0001"))
def test_find_input_file_name(directory, stem):
    """Test the determination of input file names."""
    in_file = "/".join([directory, stem + ".h5"])
    expected_dir = Path(directory).expanduser().resolve()
    assert find_input_file_name(in_file) == (expected_dir, "dummy")


def test_find_input_file_name_unexpected():
    """Test that a malformed input file name raises an error."""
    in_file = "dummy_invalid.h5"
    error = (
        f"Input file name did not have the expected format '<name>_meta.h5':\n"
        f"\t.*{in_file}"
    )
    with pytest.raises(SystemExit, match=error):
        find_input_file_name(in_file)


def test_data_files(dummy_data_transient):
    """Test the utility for discovering Tristan data file paths."""
    # Expected file paths.
    root = "dummy"
    meta_file = dummy_data_transient / f"{root}_meta.h5"
    raw_files = sorted(dummy_data_transient.iterdir())

    # Check that the absence of the metadata file raises an error.
    with pytest.raises(
        SystemExit, match="Could not find the expected detector metadata file:"
    ):
        data_files(dummy_data_transient, root)

    # Check that a metadata file with a valid (or missing) frame-processors-per-module
    # metadatum results in the correct file paths being determined.
    for fp_per_module in ((), (1, 1, 1), (3,)):
        with h5py.File(meta_file, "w") as f:
            f["fp_per_module"] = fp_per_module

        assert data_files(dummy_data_transient, root) == (raw_files, meta_file)

    # Check that missing raw files, as determined from the fp-per-module metadatum,
    # raise an error.
    fp_per_module = (4,)
    missing_file = f"{dummy_data_transient / root}_000004.h5"
    with h5py.File(meta_file, "w") as f:
        f["fp_per_module"] = fp_per_module
    with pytest.raises(
        SystemExit,
        match=f"The following expected data files are missing:\n\t{missing_file}",
    ):
        data_files(dummy_data_transient, root)


def test_latrd_data_default_keys(dummy_data):
    """
    Test that the latrd_data context manager correctly reconstructs the dummy data.

    Construct a LATRD data dictionary from the dummy data files and check that there
    is an entry for each of the default keys, and that each entry has the expected
    values.
    """
    with latrd_data(sorted(dummy_data.iterdir())) as data:
        assert set(data.keys()) == set(cue_keys + event_keys)
        # By using the same seed as when generating the test data,
        # we should be able to reproduce them.
        np.random.seed(0)
        for key in cue_keys + event_keys:
            da.assert_eq(data[key][:10], np.random.randint(10, size=10))


def test_latrd_data_specified_keys(dummy_data):
    """Test that the latrd_data context manager reads only the specified keys."""
    with latrd_data(sorted(dummy_data.iterdir()), cue_keys) as data:
        assert set(data.keys()) == set(cue_keys)


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
        first_cue_message = data[cue_id_key][0]
        assert first_cue_time(data, first_cue_message).compute() == 7
        # Next, check that searching for a cue message that does not appear in the data
        # results in no returned timestamp.
        assert random_range not in data[cue_id_key]
        assert first_cue_time(data, random_range) is None


def test_cue_times(dummy_data):
    """Test the utility for finding all timestamps of a given cue."""
    with latrd_data(sorted(dummy_data.iterdir())) as data:
        # The cue_id '3' appears four times in the test data,
        # with one duplicate timestamp.
        message = 3
        index = da.flatnonzero(data[cue_id_key] == message).compute()
        assert np.all(data[cue_time_key][index] == (8, 8, 7, 9))
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
