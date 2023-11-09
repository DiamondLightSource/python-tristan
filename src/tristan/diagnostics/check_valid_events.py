"""
Run a quick check to diagnose possible asynchronicity between the shutter timestamps and events timestamps.
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import time
from pathlib import Path

import h5py
import numpy as np

from ..command_line import version_parser
from ..data import event_time_key
from . import diagnostics_log as log
from .utils import TIME_RES, find_shutter_times, get_full_file_list

epilog_message = """
This program checks that there are events recorded after the shutter open signal in the data files.\n
The results are written to a filename_VALIDEVENTSCHECK.log.
"""

# Define parser
usage = "%(prog)s /path/to/data/dir filename_root [options]"
parser = argparse.ArgumentParser(
    usage=usage,
    formatter_class=argparse.RawTextHelpFormatter,
    description=__doc__,
    epilog=epilog_message,
    parents=[version_parser],
)
parser.add_argument("visitpath", type=str, help="Visit directory.")
parser.add_argument("filename", type=str, help="Root filename.")
parser.add_argument(
    "-s",
    "--shutters",
    type=float,
    nargs=2,
    help="Shutter open and close timestamps if known, passed in the order (open, close). \
        These values can be found by running find-trigger-signals. \
            If not passed, the program will look for them inside the data files.",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    help="Output directory to save results. If not passed, the script will default to current working directory.",
)
parser.add_argument(
    "-n",
    "--nproc",
    type=int,
    help="The number of processes to use.",
)

# Define a logger
logger = logging.getLogger("TristanDiagnostics.ValidEventsCheck")


def setup_logging(wdir, filestem):
    logfile = wdir / (filestem + "_VALIDEVENTSCHECK.log")
    log.config(logfile.as_posix())


def event_timestamp_check(tristanlist):
    mod_number, filelist, sh_open, sh_close = tristanlist

    valid = False  # Assume False and overwrite later
    min_timestamp = 0
    max_timestamp = 0
    for filename in filelist:
        with h5py.File(filename) as fh:
            event_time = fh[event_time_key]
            # Use chunking here
            shape = event_time.shape[0]
            chunk_size = event_time.chunks[0]
            chunk_num = (
                shape // chunk_size
                if shape % chunk_size == 0
                else (shape // chunk_size) + 1
            )
            for j in range(chunk_num):
                t = event_time[j * chunk_size : (j + 1) * chunk_size] * TIME_RES
                Tmin = np.min(t)
                Tmax = np.max(t)
                if Tmax > max_timestamp:
                    max_timestamp = Tmax
                if Tmin < min_timestamp:
                    min_timestamp = Tmin

    if max_timestamp > sh_open:
        if min_timestamp < sh_close:
            valid = True

    T = {
        f"Module {mod_number}": {
            "Valid events": valid,
            "Min timestamp": min_timestamp,
            "Max timestamp": max_timestamp,
        }
    }
    return T


def main(args):
    filepath = Path(args.visitpath).expanduser().resolve()
    base = args.filename + f"_{6*'[0-9]'}.h5"

    filename_template = filepath / base
    file_list = get_full_file_list(filename_template)

    L = [file_list[i : i + 10] for i in range(0, len(file_list), 10)]

    if args.output:
        savedir = Path(args.output).expanduser().resolve()
        savedir.mkdir(exist_ok=True)
    else:
        savedir = Path.cwd()

    setup_logging(savedir, args.filename)

    # Log some info
    logger.info("Check for valid events between shutter times.")
    logger.info(f"Current working directory: {savedir}")
    logger.info(f"Collection directory: {filepath}")
    logger.info(f"Filename root: {args.filename}")

    logger.info(
        f"Tristan{len(L)}M collection. Found {len(file_list)} files in directory.\n"
    )

    if args.nproc:
        nproc = args.nproc
    else:
        nproc = mp.cpu_count() - 1

    if not args.shutters:
        logger.info(
            "No shutter timestamps parsed, searching for them in the datafiles."
        )
        sh_open, sh_close = find_shutter_times(L[0])
    else:
        sh_open, sh_close = args.shutters

    logger.info(f"Interval: {sh_open} {sh_close}")

    logger.info(f"Start Pool with {nproc} processes.")
    tristanlist = [(n, l, sh_open, sh_close) for n, l in enumerate(L)]
    with mp.Pool(processes=nproc) as pool:
        func = event_timestamp_check
        res = pool.map(func, tristanlist)

    for el in res:
        for k, v in el.items():
            logger.info(f"--- {k} ---")
            if v["Valid events"] is False:
                logger.warning("WARNING! No valid events found in this module!")
            else:
                logger.info("Valid events found!")
            logger.info(f"Max timestamp found: {v['Max timestamp']}")
            logger.info(f"Min timestamp found: {v['Min timestamp']}")
    logger.info("\n")


def cli():
    tic = time.time()
    args = parser.parse_args()
    main(args)
    toc = time.time()
    logger.debug(f"Total time taken: {toc - tic:4f} s.")
    logger.info("~~~ EOF ~~~")
