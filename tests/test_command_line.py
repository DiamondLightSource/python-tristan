import argparse
from pathlib import Path

import h5py
import pint
import pytest

from tristan import __version__
from tristan.command_line import (
    _find_input_file_name,
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


def test_units_of_time():
    """Test the utility for applying a default unit to a quantity."""
    # Check that a dimensionless input quantity defaults to seconds.
    assert units_of_time(1) == ureg.Quantity(1, "s")
    # Check standard expected input.
    assert units_of_time("2ms") == ureg.Quantity(2, "ms")

    # Check that we catch invalid units.
    error = (
        r"Cannot convert from '3 meter' \(\[length\]\) to 'a quantity of' \(\[time\]\)"
    )
    with pytest.raises(pint.errors.DimensionalityError, match=error):
        units_of_time("3m")

    # Check that we catch undefined units and re-raise the UndefinedUnitError (which
    # is a subclass of AttributeError) as a ValueError, so that argparse recognises
    # that this is a bad argument.
    error = "'foo' is not defined in the unit registry"
    with pytest.raises(ValueError, match=error):
        units_of_time("foo")

    # Catch non-positive values.
    for duration in "0", "0s", "-1", "-s", "-1s":
        with pytest.raises(ValueError, match="Time quantity must be positive."):
            units_of_time(duration)


def test_positive_int():
    """Test the utility for checking an integer value is positive."""
    # Check standard expected behaviour.
    for value in 1, "1", 1.1, True:
        assert positive_int(value) == 1

    # Check that non-positive values raise a ValueError.
    for value in -1, "-1", 0, "0", 0.1, False:
        msg = f"The value {value} does not cast to a positive integer."
        with pytest.raises(ValueError, match=msg):
            positive_int(value)


@pytest.mark.parametrize("stem", ("dummy_meta", "dummy_1", "dummy_0001"))
@pytest.mark.parametrize("directory", (".", "/", "~", "test_dir"))
def test_find_input_file_name(directory, stem):
    """Test the determination of input file names."""
    in_file = "/".join([directory, stem + ".h5"])
    expected_dir = Path(directory).expanduser().resolve()
    assert _find_input_file_name(in_file) == (expected_dir, "dummy")


def test_find_input_file_name_by_directory(tmp_path):
    """Test that the input file name can be found from its parent directory."""
    with h5py.File(tmp_path / "dummy_meta.h5", "w"):
        pass
    assert _find_input_file_name(tmp_path) == (tmp_path, "dummy")


def test_find_input_file_name_unexpected():
    """Test that a malformed input file name raises an error."""
    in_file = "dummy_invalid.h5"
    error = (
        f"Input file name did not have the expected format '<name>_meta.h5':\n"
        f"\t.*{in_file}"
    )
    with pytest.raises(SystemExit, match=error):
        _find_input_file_name(in_file)


def test_find_file_name_empty_directory(tmp_path):
    """
    Test that finding an input file in an empty directory raises an appropriate error.
    """
    error = (
        "Could not find a single unique '<filename>_meta.h5' file in the "
        "specified directory."
    )
    with pytest.raises(SystemExit, match=error):
        _find_input_file_name(tmp_path)


def test_check_output_file(tmp_path):
    """Test the function for checking for a valid output file name."""
    assert check_output_file() is None
    assert check_output_file(out_file="test.ext") == Path("test.ext").resolve()
    assert check_output_file(root="test") == Path("test_output.h5").resolve()
    assert (
        check_output_file(root="test", suffix="other")
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


def test_version_parser(capsys):
    """Test that the version parser gives the correct behaviour."""
    for flag in "--version", "-V":
        with pytest.raises(SystemExit, match="0"):
            version_parser.parse_args([flag])
        assert f"Tristan tools {__version__}" in capsys.readouterr().out

    # Check that the version flag is not mandatory.
    assert version_parser.parse_args([]) == argparse.Namespace()

    # Check that this parser does not introduce a help flag.
    with pytest.raises(SystemExit, match="2"):
        version_parser.parse_args(["-h"])
    assert "error: unrecognized arguments: -h" in capsys.readouterr().err


def test_input_file_action():
    """Test the custom argparse action for parsing the input file path."""
    action = _InputFileAction(option_strings=(), dest="")
    namespace = argparse.Namespace()
    directory = "some/dummy/path/to"
    root = "file_name"
    action(argparse.ArgumentParser(), namespace, f"{directory}" f"/{root}_meta.h5")
    assert namespace.data_dir == Path(directory).resolve()
    assert namespace.root == root


def test_input_parser(capsys):
    """Test the parser for handling the input file path."""
    # Check normal expected behaviour.
    directory = "some/dummy/path/to"
    root = "file_name"
    args = input_parser.parse_args([f"{directory}/{root}_meta.h5"])
    assert args.data_dir == Path(directory).resolve()
    assert args.root == root

    # Check that the input file path is a mandatory argument.
    with pytest.raises(SystemExit, match="2"):
        input_parser.parse_args([])
    error = capsys.readouterr().err
    assert "error: the following arguments are required: input-file" in error

    # Check that an undefined directory defaults to the current working directory,
    # and that a file name that doesn't match the expected format is caught with a
    # help message.
    error = (
        "Input file name did not have the expected format '<name>_meta.h5':\n\t"
        f"{Path.cwd() / 'test.h5'}"
    )
    with pytest.raises(SystemExit, match=error):
        input_parser.parse_args(["test.h5"])

    # Check that this parser does not introduce a help flag.
    for flag in "-h", "--help":
        with pytest.raises(SystemExit, match="2"):
            input_parser.parse_args([f"{directory}/{root}_meta.h5", flag])
        assert f"error: unrecognized arguments: {flag}" in capsys.readouterr().err


def test_image_size():
    """Test unpacking an image size tuple from a comma-separated string of integers."""
    # Check normal expected behaviour.
    for size in ("1,2", "1, 2", "1 ,2", "1 , 2", "(1,2)", "'1,2'", '"1,2"', "'\"(1,2"):
        assert image_size(size) == (2, 1)

    # Check that we catch the wrong number of values passed.
    for size in "1", "1, 2, 3":
        with pytest.raises(ValueError, match=r"values to unpack \(expected 2"):
            image_size(size)

    # Check that we catch errant commas.
    for size in "", "1,", ",1", ",", "1,,2", "1,2,", "1.,2.", "a,b":
        with pytest.raises(
            ValueError, match=r"invalid literal for int\(\) with base 10:"
        ):
            image_size(size)

    # Check that we catch negative values.
    for size in "-1,0", "-1, 1", "1, -1", "0, -1":
        with pytest.raises(ValueError, match="Image dimensions must not be negative."):
            image_size(size)

    # Check that we catch the case of an image size (0, 0).
    error = "At least one image dimension must be positive."
    for size in "1,0", "0, 1":
        with pytest.raises(ValueError, match=error):
            image_size(size)


def test_output_file_parser(capsys):
    """Test the parser for handling the output file path and output image shape."""
    # Check that none of the arguments belonging to this parser are mandatory.
    args = image_output_parser.parse_args([])
    assert args.output_file is None
    assert args.force is False
    assert args.image_size is None

    # Check that the output file argument does its stuff.
    for flag in "-o", "--output-file":
        assert image_output_parser.parse_args([flag, "test"]).output_file == "test"

    # Check that the --force flag stores true.
    for flag in "-f", "--force":
        assert image_output_parser.parse_args([flag]).force is True

    # Check the normal expected behaviour for the image size argument.
    for flag in "-s", "--image-size":
        assert image_output_parser.parse_args([flag, "1,2"]).image_size == (2, 1)
    # Check that mal-formed image size values are rejected with a help message.
    for size in "1", "1,2,", "a,b", ",", "1.,2.":
        with pytest.raises(SystemExit, match="2"):
            image_output_parser.parse_args(["-s", size])
        assert (
            f"error: argument -s/--image-size: invalid image_size value: '{size}'"
            in capsys.readouterr().err
        )

    # Check that this parser does not introduce a help flag.
    for flag in "-h", "--help":
        with pytest.raises(SystemExit, match="2"):
            image_output_parser.parse_args([flag])
        assert f"error: unrecognized arguments: {flag}" in capsys.readouterr().err


def test_exposure_parser(capsys):
    """Test the parser for determining the exposure time for binning into images."""
    # Check the normal expected behaviour of the --exposure-time flag.
    for flag in "-e", "--exposure-time":
        args = exposure_parser.parse_args([flag, ".1"])
        assert args.exposure_time == pint.Quantity(100, "ms")
        assert args.num_images is None
    for exposure in ".1s", "ms", "100µs":
        args = exposure_parser.parse_args(["-e", exposure])
        assert args.exposure_time == pint.Quantity(exposure)

    # Check that mal-formed exposure time values are rejected with a help message.
    for e in "foo", "1m", "0", "0s", "-1":
        with pytest.raises(SystemExit, match="2"):
            exposure_parser.parse_args(["-e", e])
        assert (
            f"error: argument -e/--exposure-time: invalid units_of_time value: '{e}'"
            in capsys.readouterr().err
        )

    # Check the normal expected behaviour of the --num-images flag.
    for flag in "-n", "--num-images":
        args = exposure_parser.parse_args([flag, "100"])
        assert args.exposure_time is None
        assert args.num_images == 100

    # Check that mal-formed image number values are rejected with a help message.
    for number in "100.", "a", "100s", "0", "-1":
        with pytest.raises(SystemExit, match="2"):
            exposure_parser.parse_args(["-n", number])
        msg = f"error: argument -n/--num-images: invalid positive_int value: '{number}'"
        assert msg in capsys.readouterr().err

    # Check that one of the exposure time and number of images arguments is required.
    with pytest.raises(SystemExit, match="2"):
        exposure_parser.parse_args([])
    assert (
        "error: one of the arguments -e/--exposure-time -n/--num-images is required"
        in capsys.readouterr().err
    )

    # Check that the exposure time and image number arguments are mutually exclusive.
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

    # Check that this parser does not introduce a help flag.
    for flag in "-h", "--help":
        with pytest.raises(SystemExit, match="2"):
            exposure_parser.parse_args(["-e", ".1", flag])
        assert f"error: unrecognized arguments: {flag}" in capsys.readouterr().err
