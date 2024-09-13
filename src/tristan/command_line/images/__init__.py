"""Aggregate the events from a LATRD Tristan data collection into one or more images."""

from __future__ import annotations

# Some platforms seem to raise OSError: [Errno -101] NetCDF: HDF error:
# '/path/to/HDF5/file.h5'
# Importing netCDF4 before h5py here seems to fix it.  ¯\_(ツ)_/¯
import netCDF4  # noqa F401

import sys
from pathlib import Path

import h5py
import pint

from ... import clock_frequency
from ...data import seconds


def determine_image_size(nexus_file: Path) -> tuple[int, int]:
    """Find the image size from metadata."""
    recommend = "Please specify the detector dimensions in (x, y) with '--image-size'."
    try:
        with h5py.File(nexus_file) as f:
            # For the sake of some functions like zarr.create, ensure that the image
            # dimensions are definitely tuple[int, int], not tuple[np.int64,
            # np.int64] or anything else.
            y, x = f["entry/instrument/detector/module/data_size"][()].astype(int)
            return y, x
    except (FileNotFoundError, OSError):
        sys.exit(f"Cannot find NeXus file:\n\t{nexus_file}\n{recommend}")
    except KeyError:
        sys.exit(f"Input NeXus file does not contain image size metadata.\n{recommend}")


def exposure(
    start: int,
    end: int,
    exposure_time: pint.Quantity | None = None,
    num_images: int | None = None,
) -> tuple[pint.Quantity, int, int]:
    """
    Find the exposure time or number of images.

    From a start time and an end time, either derive an exposure time from the
    number of images, or derive the number of images from the exposure time.

    Args:
        start:          Start time in clock cycles.
        end:            End time in clock cycles.
        exposure_time:  Exposure time in any unit of time (optional).
        num_images:     Number of images (optional).

    Returns:
        The exposure time in seconds, the exposure time in clock cycles, the number
        of images.
    """
    if exposure_time:
        exposure_time = exposure_time.to_base_units().to_compact()
        exposure_cycles = (exposure_time * clock_frequency).to_base_units().magnitude
        num_images = int((end - start) // exposure_cycles)
    else:
        # Because they are expected to be mutually exclusive, if there is no
        # exposure_time, there must be a num_images.
        exposure_cycles = int((end - start) / num_images)
        exposure_time = seconds(exposure_cycles)

    return exposure_time, exposure_cycles, num_images
