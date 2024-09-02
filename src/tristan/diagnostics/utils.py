"""General utilities for the diagnostic tools."""
from __future__ import annotations

import glob
import logging
from enum import Enum
from pathlib import Path

import h5py
import numpy as np

from ..data import (
    cue_id_key,
    cue_time_key,
    event_location_key,
    shutter_close,
    shutter_open,
)

# Define a logger
logger = logging.getLogger("TristanDiagnostics.Utils")

# Some constants
TIME_RES = 1.5625e-9  # timing resolution fine
DIV = np.uint32(0x2000)


class FileChecker(str, Enum):
    CUES = "cues"
    EVENTS = "events"


class TristanConfig(str, Enum):
    T1M = "1M"
    T2M = "2M"
    T10M = "10M"

    @staticmethod
    def config_opts():
        return list(map(lambda d: d.value, TristanConfig))


TRISTAN_CONFIG = {"10M": (2, 5), "2M": (1, 2), "1M": (1, 1)}  # (H, V) -.> (fast, slow)

# Tristan 10M specs
MODULE_SIZE = (515, 2069)  # slow, fast
GAP_SIZE = (117, 45)  # slow, fast
IMAGE_SIZE_10M = (3043, 4183)  # slow, fast


def get_full_file_list(filename_template: str | Path) -> list[Path]:
    """Given a template filename, including directory, get a list of all the files\
    using that template.

    Args:
        filename_template(str | Path): Template to look up in the directory.

    Returns:
        file_list(list[Path]): A list of all the files found matching the template.
    """
    if not isinstance(filename_template, Path):
        filename_template = Path(filename_template)
    file_list = [
        Path(f).expanduser().resolve()
        for f in sorted(glob.glob(filename_template.as_posix()))
    ]
    return file_list


def define_modules(det_config: TristanConfig = "10M") -> dict[str, tuple]:
    """Define the start and end pixel of each module in the Tristan detector.

    Args:
        det_config (TristanConfig, optional): Specify how many physical modules make up the Tristan\
            detector currently in use. Available configurations: 1M, 2M, 10M.\
            Defaults to "10M".

    Returns:
        dict[str, tuple]: Start and end pixel value of each module - which are defined\
            by a (x,y) tuple. For example a Tristan 1M will return \
            {"0": ([0, 515], [0, 2069])}
    """
    if det_config not in TristanConfig.config_opts():
        logger.error(f"Detector configuration {det_config} unknown.")
        raise ValueError(
            f"Detector configuration unknown. Please pass one of {TristanConfig.config_opts()}."
        )
    modules = TRISTAN_CONFIG[det_config]
    mod = {}
    n = 0
    for _y in range(modules[0]):
        for _x in range(modules[1]):
            int_x = [
                _x * (MODULE_SIZE[0] + GAP_SIZE[0]),
                _x * (MODULE_SIZE[0] + GAP_SIZE[0]) + MODULE_SIZE[0],
            ]
            int_y = [
                _y * (MODULE_SIZE[1] + GAP_SIZE[1]),
                _y * (MODULE_SIZE[1] + GAP_SIZE[1]) + MODULE_SIZE[1],
            ]
            mod[str(n)] = (int_x, int_y)
            # mod[(_x, _y)] = (int_x, int_y)
            n += 1
    return mod


def module_cooordinates(det_config: TristanConfig = "10M") -> dict[str, tuple]:
    """ Create a conversion table between module number and its location on the detector.

    Args:
        det_config(TristanConfig, optional): Specify how many physical modules make up the Tristan\
            detector currently in use. Available configurations: 1M, 2M, 10M.\
            Defaults to "10M".

    Returns:
        dict[str, tuple]: effectively a conversion table mapping the module number to its\
        location on the detector. For example a Trisstan 1M will return \
        {"0": (0, 0)}
    """
    if det_config not in TristanConfig.config_opts():
        logger.error(f"Detector configuration {det_config} unknown.")
        raise ValueError(
            f"Detector configuration unknown. Please pass one of {TristanConfig.config_opts()}."
        )
    modules = TRISTAN_CONFIG[det_config]
    table = {}
    n = 0
    for _y in range(modules[0]):
        for _x in range(modules[1]):
            table[str(n)] = (_x, _y)
            n += 1
    return table


def _check_for_cues(filename: Path) -> bool:
    try:
        with h5py.File(filename) as fh:
            _ = fh[cue_id_key][1]
            _ = fh[cue_time_key][1]
        return True
    except IndexError:
        return False


def assign_files_to_modules(
    filelist: list[Path],
    det_config: TristanConfig = "10M",
    check_for: FileChecker = "events",
):
    """ Assign each file to the correct module after having checked that it has valid events/cues in it.
    While the files should be in order i.e. for module 0 we'll have file numbers 000001-000010, for module 1 \
    file numbers 000011-000020 and so on, when checking for events an additional check will be done on the \
    event id when assigning to each module.

    Args:
        filelist (list[Path]): List of input tristan files.
        det_config (TristanConfig, optional): Specify how many physical modules make up the Tristan \
            detector currently in use. Available configurations: 1M, 2M, 10M.\
            Defaults to "10M".
        check_for (FileChecker, optional): Specify whether to check for valid events or cues. \
            Defaults to "cues".

    Returns:
        files_per_module(dict[str, list(Path)]): List of files assigned to each module.
    """
    MOD = define_modules(det_config)
    files_per_module = {k: [] for k in MOD.keys()}
    broken_files = []
    match check_for:
        case FileChecker.EVENTS:
            for filename in filelist:
                try:
                    with h5py.File(filename) as fh:
                        x, y = divmod(fh[event_location_key][1], DIV)
                        # Assign file to correct module depending on coordinates.
                        # Should be irrelevant as they go in order but still good to check for now
                        for k, v in MOD.items():
                            if v[1][0] <= x <= v[1][1]:
                                if v[0][0] <= y <= v[0][1]:
                                    files_per_module[k].append(filename)
                except IndexError:
                    broken_files.append(filename)
        case FileChecker.CUES:
            # NOTE This is to check for triggers when dealing with dark field collections which
            # sometimes have datasets with no events.
            for filename in filelist:
                has_cues = _check_for_cues(filename)
                if has_cues:
                    # Assign to module based on filename
                    filenum = int(filename.stem.split("_")[-1]) - 1
                    mod_num = str(filenum // int(det_config.strip("M")))
                    files_per_module[mod_num].append(filename)
                else:
                    broken_files.append(filename)
    return files_per_module, broken_files


def find_shutter_times(filelist):
    """Find shutter open and close timestamps."""
    sh_open = []
    sh_close = []
    for filename in filelist:
        with h5py.File(filename) as fh:
            cues = fh[cue_id_key][()]
            cues_time = fh[cue_time_key]
            op_idx = np.where(cues == shutter_open)[0]
            cl_idx = np.where(cues == shutter_close)[0]
            if len(op_idx) == 1:
                sh_open.append(cues_time[op_idx[0]] * TIME_RES)
            if len(cl_idx) == 1:
                sh_close.append(cues_time[cl_idx[0]] * TIME_RES)
    return sh_open[0], sh_close[0]
