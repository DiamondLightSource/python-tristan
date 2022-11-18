from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import pint
import pytest

from tristan import __version__
from tristan.command_line import (
    _InputFileAction,
    check_output_file,
    data_files,
    exposure_parser,
    image_output_parser,
    image_size,
    input_parser,
    positive_int,
    units_of_time,
    version_parser,
)

ureg = pint.UnitRegistry()


def test_check_output_file(tmp_path):
    """Test the function for checking for a valid output file name."""
    assert check_output_file() is None
    assert check_output_file(out_file="test.ext") == Path("test.ext").resolve()
    assert check_output_file(stem="test") == Path("test_output.h5").resolve()
    assert (
        check_output_file(stem="test", suffix="other")
        == Path("test_other.h5").resolve()
    )
    test_file = tmp_path / "test.ext"
    test_file.touch()
    with pytest.raises(SystemExit, match="This output file already exists:"):
        check_output_file(out_file=test_file)
    assert check_output_file(out_file=test_file, force=True) == test_file


def test_data_files(dummy_data_transient):
    """Test the utility for discovering Tristan data file paths."""
    # Expected file paths.
    stem = "dummy"
    meta_file = dummy_data_transient / f"{stem}_meta.h5"
    raw_files = sorted(dummy_data_transient.iterdir())

    # Check that the absence of the metadata file raises an error.
    with pytest.raises(
        SystemExit, match="Could not find the expected detector metadata file:"
    ):
        data_files(dummy_data_transient, stem)

    # Check that a metadata file with a valid (or missing) frame-processors-per-module
    # metadatum results in the correct file paths being determined.
    for fp_per_module in ((), (1, 1, 1), (3,)):
        with h5py.File(meta_file, "w") as f:
            f["fp_per_module"] = fp_per_module

        assert data_files(dummy_data_transient, stem) == (raw_files, meta_file)

    # Check that missing raw files, as determined from the fp-per-module metadatum,
    # raise an error.
    fp_per_module = (4,)
    missing_file = f"{dummy_data_transient / stem}_000004.h5"
    with h5py.File(meta_file, "w") as f:
        f["fp_per_module"] = fp_per_module
    with pytest.raises(
        SystemExit,
        match=f"The following expected data files are missing:\n\t{missing_file}",
    ):
        data_files(dummy_data_transient, stem)


def test_version_parser(capsys):
    """Test that the version parser gives the correct behaviour."""
    for flag in "--version", "-V":
        with pytest.raises(SystemExit, match="0"):
            version_parser.parse_args([flag])
        assert f"Tristan tools {__version__}" in capsys.readouterr().out


def test_version_parser_optional():
    """Check that the version flag is not mandatory."""
    assert version_parser.parse_args([]) == argparse.Namespace()


def test_version_parser_no_help(capsys):
    """Check that the version parser does not introduce a help flag."""
    with pytest.raises(SystemExit, match="2"):
        version_parser.parse_args(["-h"])
    assert "error: unrecognized arguments: -h" in capsys.readouterr().err


@pytest.mark.parametrize(
    "filename", ("dummy_meta.h5", "dummy_1.h5", "dummy_0001.h5", "dummy.nxs")
)
@pytest.mark.parametrize("directory", (".", "/", "~", "test_dir"))
def test_find_input_file_name(directory, filename):
    """Test the determination of input file names."""
    in_file = "/".join([directory, filename])
    expected_dir = Path(directory).expanduser().resolve()
    assert _InputFileAction.find_input_file_name(in_file) == (expected_dir, "dummy")


def test_find_input_file_name_by_directory(tmp_path):
    """Test that the input file name can be found from its parent directory."""
    with h5py.File(tmp_path / "dummy_meta.h5", "w"):
        pass
    assert _InputFileAction.find_input_file_name(tmp_path) == (tmp_path, "dummy")


def test_find_input_file_name_unexpected():
    """Test that a malformed input file name raises an error."""
    in_file = "dummy_invalid.h5"
    error = (
        "Input file name did not have the expected format '<name>_meta.h5' or "
        "'<name>.nxs':\n"
        f"\t{in_file}"
    )
    with pytest.raises(ValueError, match=error):
        _InputFileAction.find_input_file_name(in_file)


def test_find_file_name_empty_directory(tmp_path):
    """
    Test that finding an input file in an empty directory raises an appropriate error.
    """
    error = (
        "Could not find a single unique '<filename>_meta.h5' or '<filename>.nxs' file "
        "in the specified directory."
    )
    with pytest.raises(ValueError, match=error):
        _InputFileAction.find_input_file_name(tmp_path)


def test_input_file_action():
    """Test the custom argparse action for parsing the input file path."""
    action = _InputFileAction(option_strings=(), dest="")
    namespace = argparse.Namespace()
    directory = "some/dummy/path/to"
    stem = "file_name"
    action(argparse.ArgumentParser(), namespace, f"{directory}" f"/{stem}_meta.h5")
    assert namespace.data_dir == Path(directory).resolve()
    assert namespace.stem == stem


def test_input_parser():
    """Test the parser for handling the input file path."""
    directory = "some/dummy/path/to"
    stem = "file_name"
    args = input_parser.parse_args([f"{directory}/{stem}_meta.h5"])
    assert args.data_dir == Path(directory).resolve()
    assert args.stem == stem


def test_input_parser_mandatory(capsys):
    """Check that the input file path is a mandatory argument."""
    with pytest.raises(SystemExit, match="2"):
        input_parser.parse_args([])
    error = capsys.readouterr().err
    assert "error: the following arguments are required: input-file" in error


def test_input_parser_improper_input(capsys):
    """
    Check that a mangled meta file name is caught.

    Check that a file name that doesn't match the expected format is caught with a
    help message.
    """
    filename = "test.h5"
    error = (
        "Input file name did not have the expected format '<name>_meta.h5' or "
        "'<name>.nxs':\n\t"
        f"{filename}"
    )
    with pytest.raises(SystemExit, match="2"):
        input_parser.parse_args([filename])

    assert error in capsys.readouterr().err


@pytest.mark.parametrize(
    "filename", ("dummy_meta.h5", "dummy_1.h5", "dummy_0001.h5", "dummy.nxs")
)
def test_input_parser_cwd(run_in_tmp_path, filename):
    """Check that an undefined directory defaults to the current working directory."""
    tmp_path = run_in_tmp_path
    with h5py.File(tmp_path / filename, "w"):
        pass
    args = input_parser.parse_args([filename])
    assert args.data_dir == tmp_path
    assert args.stem == "dummy"


def test_input_parser_no_help(capsys):
    """Check that the input parser does not introduce a help flag."""
    directory = "some/dummy/path/to"
    stem = "file_name"
    for flag in "-h", "--help":
        with pytest.raises(SystemExit, match="2"):
            input_parser.parse_args([f"{directory}/{stem}_meta.h5", flag])
        assert f"error: unrecognized arguments: {flag}" in capsys.readouterr().err


def test_image_size():
    """Test unpacking an image size tuple from a comma-separated string of integers."""
    for size in "1,2", "1, 2", "1 ,2", "1 , 2", "(1,2)", "'1,2'", '"1,2"', "'\"(1,2'\"":
        assert image_size(size) == (2, 1)


def test_image_size_wrong_number():
    """Check that we catch the wrong number of values passed."""
    for size in "1", "1, 2, 3":
        with pytest.raises(ValueError, match=r"values to unpack \(expected 2"):
            image_size(size)


def test_image_size_mangled_input():
    """Check that we catch errant commas."""
    for size in "", "1,", ",1", ",", "1,,2", "1,2,", "1.,2.", "a,b":
        with pytest.raises(
            ValueError, match=r"invalid literal for int\(\) with base 10:"
        ):
            image_size(size)


def test_image_size_negative():
    """Check that we catch negative values."""
    for size in "-1,0", "-1, 1", "1, -1", "0, -1":
        with pytest.raises(ValueError, match="Image dimensions must not be negative."):
            image_size(size)


def test_image_size_nonpositive():
    """Check that we catch the case of an image size (0, 0)."""
    error = "At least one image dimension must be positive."
    for size in "1,0", "0, 1":
        with pytest.raises(ValueError, match=error):
            image_size(size)


def test_image_output_parser_optional():
    """
    Test the parser for handling the output file path and output image shape.

    Check that none of the arguments belonging to this parser are mandatory.
    """
    args = image_output_parser.parse_args([])
    assert args.output_file is None
    assert args.force is False
    assert args.image_size is None


def test_image_output_parser_output():
    """Check that the output/image size parser's output file argument does its stuff."""
    for flag in "-o", "--output-file":
        assert image_output_parser.parse_args([flag, "test"]).output_file == "test"


def test_image_output_parser_force():
    """Check that the output/image size parser's --force flag stores true."""
    for flag in "-f", "--force":
        assert image_output_parser.parse_args([flag]).force is True


def test_image_output_parser_image_size():
    """Check the normal expected behaviour for the output parser image size argument."""
    for flag in "-s", "--image-size":
        assert image_output_parser.parse_args([flag, "1,2"]).image_size == (2, 1)


def test_image_output_parser_malformed_image_size(capsys):
    """Check that mal-formed image size values are rejected with a help message."""
    for size in "1", "1,2,", "a,b", ",", "1.,2.":
        with pytest.raises(SystemExit, match="2"):
            image_output_parser.parse_args(["-s", size])
        assert (
            f"error: argument -s/--image-size: invalid image_size value: '{size}'"
            in capsys.readouterr().err
        )


def test_image_output_parser_no_help(capsys):
    """Check that the output/image size parser parser does not introduce a help flag."""
    for flag in "-h", "--help":
        with pytest.raises(SystemExit, match="2"):
            image_output_parser.parse_args([flag])
        assert f"error: unrecognized arguments: {flag}" in capsys.readouterr().err


def test_units_of_time():
    """Test the utility for applying a default unit to a quantity."""
    # Check that a dimensionless input quantity defaults to seconds.
    assert units_of_time(1) == ureg.Quantity(1, "s")
    # Check standard expected input.
    assert units_of_time("2ms") == ureg.Quantity(2, "ms")


def test_units_of_time_invalid():
    """Check that we catch invalid units in the units_of_time function."""
    error = (
        r"Cannot convert from '3 meter' \(\[length\]\) to 'a quantity of' \(\[time\]\)"
    )
    with pytest.raises(pint.errors.DimensionalityError, match=error):
        units_of_time("3m")


def test_units_of_time_undefined():
    """
    Check that we catch undefined units in the units_of_time function.

    Check that the UndefinedUnitError (which is a subclass of AttributeError) is
    re-raised as a ValueError, so that argparse recognises that this is a bad argument.
    """
    error = "'foo' is not defined in the unit registry"
    with pytest.raises(ValueError, match=error):
        units_of_time("foo")


def test_units_of_time_nonpositive():
    """Catch non-positive values in the units_of_time function."""
    for duration in "0", "0s", "-1", "-s", "-1s":
        with pytest.raises(ValueError, match="Time quantity must be positive."):
            units_of_time(duration)


def test_positive_int():
    """Test the utility for checking an integer value is positive."""
    for value in 1, "1", 1.1, True:
        assert positive_int(value) == 1


def test_positive_int_nonpositive():
    """Check that non-positive values passed to positive_int raise a ValueError."""
    for value in -1, "-1", 0, "0", 0.1, False:
        msg = f"The value {value} does not cast to a positive integer."
        with pytest.raises(ValueError, match=msg):
            positive_int(value)


def test_exposure_parser_exposure_time():
    """Test the behaviour of the exposure time/image number parser's exposure flag."""
    for flag in "-e", "--exposure-time":
        args = exposure_parser.parse_args([flag, ".1"])
        assert args.exposure_time == pint.Quantity(100, "ms")
        assert args.num_images is None
    for exposure in ".1s", "ms", "100µs":
        args = exposure_parser.parse_args(["-e", exposure])
        assert args.exposure_time == pint.Quantity(exposure)


def test_exposure_parser_invalid_exposure_time(capsys):
    """Check that mal-formed exposure time values are rejected with a help message."""
    for e in "foo", "1m", "0", "0s", "-1":
        with pytest.raises(SystemExit, match="2"):
            exposure_parser.parse_args(["-e", e])
        assert (
            f"error: argument -e/--exposure-time: invalid units_of_time value: '{e}'"
            in capsys.readouterr().err
        )


def test_exposure_parser_image_number():
    """Test the behaviour of the exposure time/image number parser's images flag."""
    for flag in "-n", "--num-images":
        args = exposure_parser.parse_args([flag, "100"])
        assert args.exposure_time is None
        assert args.num_images == 100


def test_exposure_parser_invalid_image_number(capsys):
    """Check that mal-formed image number values are rejected with a help message."""
    for number in "100.", "a", "100s", "0", "-1":
        with pytest.raises(SystemExit, match="2"):
            exposure_parser.parse_args(["-n", number])
        msg = f"error: argument -n/--num-images: invalid positive_int value: '{number}'"
        assert msg in capsys.readouterr().err


def test_exposure_parser_mandatory(capsys):
    """Check that either the exposure time or the number of images is required."""
    with pytest.raises(SystemExit, match="2"):
        exposure_parser.parse_args([])
    assert (
        "error: one of the arguments -e/--exposure-time -n/--num-images is required"
        in capsys.readouterr().err
    )


def test_exposure_time_mutually_exclusive(capsys):
    """Check that the exposure time and image number args are mutually exclusive."""
    with pytest.raises(SystemExit, match="2"):
        exposure_parser.parse_args(["-e", ".1", "-n", "100"])
    assert (
        "error: argument -n/--num-images: not allowed with argument -e/--exposure-time"
        in capsys.readouterr().err
    )
    with pytest.raises(SystemExit, match="2"):
        exposure_parser.parse_args(["-n", "100", "-e", ".1"])
    assert (
        "error: argument -e/--exposure-time: not allowed with argument -n/--num-images"
        in capsys.readouterr().err
    )


def test_exposure_time_no_help(capsys):
    """Check that this parser does not introduce a help flag."""
    for flag in "-h", "--help":
        with pytest.raises(SystemExit, match="2"):
            exposure_parser.parse_args(["-e", ".1", flag])
        assert f"error: unrecognized arguments: {flag}" in capsys.readouterr().err
