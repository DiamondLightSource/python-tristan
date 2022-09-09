"""
Check that all modules contain valid data.
"""
import argparse
import logging

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


def main():
    log.config()


def cli():
    main()
