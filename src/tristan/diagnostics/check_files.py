"""
Check that all modules contain valid data.
"""

import argparse
import logging
import time

from ..command_line import version_parser
from . import diagnostics_log as log

# from . import (
#     modules,
#     mod_size,
#     gap_size,
#     image_size,
#     define_modules,
# )

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
    help="Output directory to save results. If not passed, the script will default to current working directory.",
)


def main(args):
    log.config()
    print(args)


def cli():
    tic = time.process_time()
    args = parser.parse_args()
    main(args)
    toc = time.process_time()
    logger.debug(f"Total time taken: {toc - tic:.4f} s.")
