from __future__ import annotations

import argparse

from .. import (
    exposure_parser,
    gate_parser,
    image_output_parser,
    input_parser,
    interval_parser,
    trigger_parser,
    triggers,
    version_parser,
)
from . import (
    gated_images_cli,
    multiple_images_cli,
    multiple_sequences_cli,
    pump_probe_cli,
    single_image_cli,
)

parser = argparse.ArgumentParser(description=__doc__, parents=[version_parser])
subparsers = parser.add_subparsers(
    help="Choose the manner in which to create images.",
    required=True,
    dest="sub-command",
)

parser_single = subparsers.add_parser(
    "single",
    aliases=["1"],
    description=single_image_cli.__doc__,
    parents=[version_parser, input_parser, image_output_parser],
)
parser_single.set_defaults(func=single_image_cli.main)

parser_multiple = subparsers.add_parser(
    "multiple",
    aliases=["multi"],
    description=multiple_images_cli.__doc__,
    parents=[version_parser, input_parser, image_output_parser, exposure_parser],
)
parser_multiple.add_argument(
    "-a",
    "--align-trigger",
    help="Align the start and end time of images such that the first trigger signal of "
    "the chosen type is matched up with an image start time.  Useful for examining "
    "effects in the data before and after a single trigger pulse.",
    choices=triggers.keys(),
)
parser_multiple.set_defaults(func=multiple_images_cli.main)

parser_pump_probe = subparsers.add_parser(
    "pump-probe",
    aliases=["pp"],
    description=pump_probe_cli.__doc__,
    parents=[
        version_parser,
        input_parser,
        image_output_parser,
        trigger_parser,
        exposure_parser,
    ],
)
parser_pump_probe.set_defaults(func=pump_probe_cli.main)

parser_multiple_sequences = subparsers.add_parser(
    "sequences",
    aliases=["sweeps"],
    description=multiple_sequences_cli.__doc__,
    parents=[
        version_parser,
        input_parser,
        image_output_parser,
        trigger_parser,
        exposure_parser,
        interval_parser,
    ],
)
parser_multiple_sequences.set_defaults(func=multiple_sequences_cli.main)

parser_serial = subparsers.add_parser(
    "serial",
    description=gated_images_cli.__doc__,
    parents=[version_parser, input_parser, image_output_parser, gate_parser],
)
parser_serial.set_defaults(func=gated_images_cli.main)


def main(args=None):
    """Perform the image binning with a user-specified sub-command."""
    args = parser.parse_args(args)
    args.func(args)
