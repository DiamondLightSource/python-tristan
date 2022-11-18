from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import pytest

from tristan.data import cue_keys, event_keys

random_range = 10


@pytest.fixture
def run_in_tmp_path(tmp_path) -> Path:
    """
    A fixture to change the working directory for the test to a temporary directory.

    The original working directory is restored upon teardown of the fixture.

    Args:
        tmp_path: Pytest tmp_path fixture, see
                  https://docs.pytest.org/en/latest/how-to/tmp_path.html

    Yields:
        The path to the temporary working directory defined by tmp_path.
    """
    cwd = Path.cwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(cwd)


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
