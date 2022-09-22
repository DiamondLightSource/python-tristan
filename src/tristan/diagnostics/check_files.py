"""
Check that all files from all detector modules contain valid data.
"""
import argparse
import glob
import logging
import time
from pathlib import Path

import h5py

from ..command_line import version_parser
from . import DIV, define_modules
from . import diagnostics_log as log

# Define a logger
logger = logging.getLogger("TristanDiagnostics.ModuleCheck")

# Define parser
parser = argparse.ArgumentParser(description=__doc__, parents=[version_parser])
parser.add_argument("visitpath", type=str, help="Visit directory.")
parser.add_argument("filename", type=str, help="Root filename.")
parser.add_argument(
    "-l",
    "--list",
    action="store_true",
    help="Print out the list of files corresponding to each module.",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    help="Output directory to save results/log file. If not passed, the script will default to current working directory.",
)


def main(args):
    filepath = Path(args.visitpath).expanduser().resolve()
    base = args.filename + f"_{6*'[0-9]'}.h5"

    if args.output:
        savedir = Path(args.output).expanduser().resolve()
        savedir.mkdir(exist_ok=True)
    else:
        savedir = Path.cwd()

    logfile = savedir / (filepath.stem + "_MODULECHECK.log")
    log.config(logfile.as_posix())

    logger.info("Quick data check for Tristan 10M modules.")
    logger.info(f"Collection directory: {filepath}")
    logger.info(f"Filename root: {args.filename}")

    filename_template = filepath / base
    file_list = [
        Path(f).expanduser().resolve()
        for f in sorted(glob.glob(filename_template.as_posix()))
    ]
    logger.info(f"Found {len(file_list)} files in directory.")

    MOD = define_modules()
    logger.info("Assigning each data file to correct module.\n")
    split = {k: [] for k in MOD.keys()}
    broken = []
    for filename in file_list:
        with h5py.File(filename, "r") as fh:
            try:
                # Note: checking item of index 1 because for broken files there will just be one item in "event_id" set to 0.
                x, y = divmod(fh["event_id"][1], DIV)
                for k, v in MOD.items():
                    if x >= v[1][0] and x <= v[1][1]:
                        if y >= v[0][0] and y <= v[0][1]:
                            split[k].append(filename.name)
            except IndexError:
                broken.append(filename.name)

    num = 0
    for k, v in split.items():
        logger.info(f"--- Module {num} ---")
        logger.info(f"Position of detector: {k}")
        logger.info(f"Number of files found for this module: {len(v)}")
        num += 1
        if args.list:
            for f in v:
                logger.info(f"{f}")
        logger.info("\n")

    if len(broken) == 0.0:
        logger.info("No broken files found.")
    else:
        logger.warning(f"{len(broken)} broken files found.")
        logger.info("The following files all have missing data:")
        for i in broken:
            logger.info(f"{i}")


def cli():
    tic = time.process_time()
    args = parser.parse_args()
    main(args)
    toc = time.process_time()
    logger.debug(f"Total time taken: {toc - tic:.4f} s.")
