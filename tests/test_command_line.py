from pathlib import Path

import h5py
import pytest

from tristan.command_line import data_files, find_input_file_name


@pytest.mark.parametrize("stem", ("dummy_meta", "dummy_1", "dummy_0001"))
@pytest.mark.parametrize("directory", (".", "/", "~", "test_dir"))
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


@pytest.mark.xfail
def test_find_file_names(tmp_path_factory):
    raise NotImplementedError


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
