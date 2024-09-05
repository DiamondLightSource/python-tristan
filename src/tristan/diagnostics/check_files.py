"""
Check that all files from all detector modules contain valid data.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from ..command_line import version_parser
from . import diagnostics_log as log
from .utils import (
    TristanConfig,
    assign_files_to_modules,
    get_filename_template,
    get_full_file_list,
    module_cooordinates,
)

epilog_message = """
This program runs through all the files written for a Tristan collection and checks that they contain events, \
as well as that they are assigned to the correct module.\n
The results are written to a filename_MODULECHECK.log.
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
    "-l",
    "--list",
    action="store_true",
    help="Print out the list of files corresponding to each module.",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    help="""
    Output directory to save results/log file.
    If not passed, the script will default to current working directory.
    """,
)
parser.add_argument(
    "-m",
    "--num-modules",
    choices=["1M", "2M", "10M"],
    default="10M",
    type=str,
    help="Number of detector modules.",
)

# Define a logger
logger = logging.getLogger("TristanDiagnostics.ModuleCheck")


def setup_logging(wdir: Path, filestem: str):
    logfile = wdir / (filestem + "_MODULECHECK.log")
    log.config(logfile.as_posix())


def run_file_check(
    filepath: Path,
    filename_root: str,
    outdir: Path | None = None,
    det_config: TristanConfig = "10M",
    print_list: bool = False,
):
    filename_template = get_filename_template(filepath, filename_root)

    # Current working directory
    if outdir:
        outdir.mkdir(exist_ok=True, parents=True)
    else:
        outdir = Path.cwd()

    setup_logging(outdir, filepath.stem)

    logger.info(f"Quick data check for Tristan {det_config} modules.")
    logger.info(f"Collection directory: {filepath}")
    logger.info(f"Filename root: {filename_root}")

    file_list = get_full_file_list(filename_template)
    logger.info(f"Found {len(file_list)} files in directory.")

    mod_coord = module_cooordinates(det_config)
    logger.info("Assigning each data file to correct module.\n")
    split, broken = assign_files_to_modules(file_list, det_config)
    split = {k: [val.name for val in v] for k, v in split.items()}
    broken = [b.name for b in broken]

    for k, v in split.items():
        logger.info(f"--- Module {k} ---")
        logger.info(f"Position on detector: {mod_coord[k]}")
        logger.info(f"Number of files found for this module: {len(v)}")
        if print_list:
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
    tic = time.time()
    args = parser.parse_args()

    filepath = Path(args.visitpath).expanduser().resolve()
    wdir = Path(args.output).expanduser().resolve() if args.output else None

    run_file_check(filepath, args.filename, wdir, args.num_modules, args.list)
    # main(args)
    toc = time.time()
    logger.debug(f"Total time taken: {toc - tic:.4f} s.")
