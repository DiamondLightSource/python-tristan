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
    description=(
        "Aggregate all the events from a LATRD Tristan data collection "
        "into a single image."
    ),
    parents=[version_parser, input_parser, image_output_parser],
)
parser_single.set_defaults(func=single_image_cli.main)

parser_multiple = subparsers.add_parser(
    "multiple",
    aliases=["multi"],
    description=(
        "Bin the events from a LATRD Tristan data collection into multiple images."
    ),
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
    description="Bin events into images representing different pump-probe delays.\n\n"
    "With LATRD data from a pump-probe experiment, where the pump signal has a fairly "
    "constant repeat rate, bin events into a stack of images spanning the range of "
    "pump-probe delay times, from shortest to longest.",
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
    description="Bin events into several sequences of images, each corresponding to "
    "a different pump-probe delay time interval.\n\n"
    "With LATRD data from a pump-probe experiment, where the pump signal has a fairly "
    "constant repeat rate, the recorded events are separated into groups corresponding "
    "to the time elapsed since the most recent pump trigger signal.  Each group is "
    "binned into a sequence of chronological images.  Each sequence is saved to a "
    "separate output file, numbered from the shortest pump-probe delay to the longest.",
    parents=[
        version_parser,
        input_parser,
        image_output_parser,
        trigger_parser,
        exposure_parser,
        interval_parser,
    ],
)
parser_multiple_sequences.set_defaults(func=multiple_sequences_cli)

parser_serial = subparsers.add_parser(
    "serial",
    description="Bin events into images, gated with trigger signals.\n\n"
    "Events will be binned into as many images as there are gate signals, one image "
    "per gate.  Each 'gate-open' signal is taken as the start of an exposure and the "
    "next 'gate-close' signal is taken as the end of the exposure.",
    parents=[version_parser, input_parser, image_output_parser, gate_parser],
)
parser_serial.set_defaults(func=gated_images_cli)


def main(args=None):
    """Perform the image binning with a user-specified sub-command."""
    args = parser.parse_args(args)
    args.func(args)
