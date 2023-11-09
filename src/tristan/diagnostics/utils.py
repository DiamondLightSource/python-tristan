"""General utilities for the diagnostic tools."""
from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Literal, get_args

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

# Tristan 10M specs
TConfig = Literal["1M", "2M", "10M"]
tristan_config = {"10M": (2, 5), "2M": (1, 2), "1M": (1, 1)}  # (H, V) -.> (fast, slow)
mod_size = (515, 2069)  # slow, fast
gap_size = (117, 45)  # slow, fast
image_size = (3043, 4183)  # slow, fast


def get_full_file_list(filename_template: str | Path) -> list(Path):
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


def define_modules(det_config: TConfig = "10M") -> dict[str, tuple]:
    """Define the start and end pixel of each module in the Tristan detector.

    Args:
        det_config (TConfig, optional): Specify how many physical modules make up the Tristan\
            detector currently in use. Available configurations: 1M, 2M, 10M.\
            Defaults to "10M".

    Returns:
        dict[str, tuple]: Start and end pixel value of each module - which are defined\
            by a (x,y) tuple. For example a Tristan 1M will return \
            {"0": ([0, 515], [0, 2069])}
    """
    config_opts = get_args(TConfig)
    if det_config not in config_opts:
        logger.error(f"Detector configuration {det_config} unknown.")
        raise ValueError(
            f"Detector configuration unknown. Please pass one of {config_opts}."
        )
    modules = tristan_config[det_config]
    mod = {}
    n = 0
    for _y in range(modules[0]):
        for _x in range(modules[1]):
            int_x = [
                _x * (mod_size[0] + gap_size[0]),
                _x * (mod_size[0] + gap_size[0]) + mod_size[0],
            ]
            int_y = [
                _y * (mod_size[1] + gap_size[1]),
                _y * (mod_size[1] + gap_size[1]) + mod_size[1],
            ]
            mod[str(n)] = (int_x, int_y)
            # mod[(_x, _y)] = (int_x, int_y)
            n += 1
    return mod


def module_cooordinates(det_config: TConfig = "10M") -> dict[str, tuple]:
    """ Create a conversion table between module number and its location on the detector.

    Args:
        det_config(TConfig, optional): Specify how many physical modules make up the Tristan\
            detector currently in use. Available configurations: 1M, 2M, 10M.\
            Defaults to "10M".

    Returns:
        dict[str, tuple]: effectively a conversion table mapping the module number to its\
        location on the detector. For example a Trisstan 1M will return \
        {"0": (0, 0)}
    """
    config_opts = get_args(TConfig)
    if det_config not in config_opts:
        logger.error(f"Detector configuration {det_config} unknown.")
        raise ValueError(
            f"Detector configuration unknown. Please pass one of {config_opts}."
        )
    modules = tristan_config[det_config]
    table = {}
    n = 0
    for _y in range(modules[0]):
        for _x in range(modules[1]):
            table[str(n)] = (_x, _y)
            n += 1
    return table


def assign_files_to_modules(filelist: list[Path | str], det_config: TConfig = "10M"):
    MOD = define_modules(det_config)
    files_per_module = {k: [] for k in MOD.keys()}
    broken_files = []
    for filename in filelist:
        with h5py.File(filename) as fh:
            try:
                x, y = divmod(fh[event_location_key][1], DIV)
                for k, v in MOD.items():
                    if v[1][0] <= x <= v[1][1]:
                        if v[0][0] <= y <= v[0][1]:
                            files_per_module[k].append(filename)
            except IndexError:
                broken_files.append(filename)
    return files_per_module, broken_files


def find_shutter_times(filelist):
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
