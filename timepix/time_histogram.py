#!/usr/bin/env python
# coding: utf-8

"""Create a time histogram of all events on the detector and plot it in SVG format."""

import argparse
import os
import sys
from typing import List, Optional, Sequence

import h5py
import numpy as np
import pint
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from timepix import (
    clock_frequency,
    cue_id_key,
    cue_times,
    event_time_key,
    first_cue_time,
    fullpath,
    seconds,
    shutter_close,
    shutter_open,
    ttl_rising,
)

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity


def make_figure(data_file: h5py.File, events_group: str, exposure_time: int) -> Figure:
    start_time = first_cue_time(data_file, shutter_open, events_group=events_group)
    end_time = first_cue_time(data_file, shutter_close, events_group=events_group)

    # Get only those laser pulses occurring between shutter open and close.
    laser_pulse_times = cue_times(data_file, ttl_rising, events_group=events_group)
    laser_pulse_times = laser_pulse_times[
        (start_time <= laser_pulse_times) & (laser_pulse_times <= end_time)
    ]

    figure, _ = plot_histogram(
        data_file[events_group + event_time_key],
        start_time,
        end_time,
        exposure_time,
        laser_pulse_times,
    )

    return figure


def plot_histogram(
    events: Sequence[int],
    start: int,
    end: int,
    exposure_time: int,
    pulses: Sequence[int] = None,
) -> (Figure, Axes):
    import matplotlib

    matplotlib.use("svg")

    from matplotlib import pyplot as plt

    # If there are any laser pulses, align a bin edge with the first one.
    if pulses:
        pre_trigger_bins = np.arange(pulses[0] - exposure_time, start, -exposure_time)
        post_trigger_bins = np.arange(pulses[0], end, exposure_time)
        bins = np.concatenate([pre_trigger_bins[::-1], post_trigger_bins])
        start = pulses[0]
        bins = seconds(bins, start)
    else:
        bins = np.arange(0, end - start + exposure_time, exposure_time)
        pulses = np.array([])

    fig, ax = plt.subplots()

    # Plot the laser pulses as vertical red lines.
    for pulse in seconds(pulses, start):
        ax.axvline(pulse, color="r")

    # Plot the histogram.
    ax.hist(seconds(events, start), bins)

    ax.set_title(rf"Exposure time: {seconds(exposure_time)}$\,$s")
    ax.set_ylabel("Number of events")
    if pulses:
        ax.set_xlabel("Time from first laser pulse (seconds)")
    else:
        ax.set_xlabel("Time from detector shutter open (seconds)")

    return fig, ax


class _ClockCyclesAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # values = [int(value * clock_frequency) for value in values]
        # Parse human-readable input.
        values = Q_(values)
        if values.u == ureg.Unit(""):
            values *= ureg.Unit("s")
        values *= clock_frequency * ureg.Unit("Hz")
        values = int(values.to_base_units().m)
        setattr(namespace, self.dest, values)


def determine_output_file(infile: str, outfile: Optional[str] = None) -> str:
    infile = fullpath(infile)

    if outfile:
        outfile = fullpath(outfile)
        outfile, ext = os.path.splitext(outfile)
        if ext not in [".svg", ""]:
            print(
                f"You have specified the invalid output file extension '{ext}'.\n"
                "\tThis will be replaced with '.svg'."
            )
    else:
        visit_dir = os.getenv("VISITDIR")
        visit_dir = os.path.abspath(visit_dir) if visit_dir else visit_dir
        if not visit_dir:
            sys.exit(
                "The output file path could not be determined automatically "
                "from the visit directory.\n"
                "\tPlease specify an output file path with the '-o' option."
            )
        elif visit_dir not in infile:
            sys.exit(
                "The input file you have specified does not belong to the current "
                "visit on beamline I19.\n"
                "\tPlease specify an output file path with the '-o' option."
            )
        else:
            outfile = os.path.join(
                visit_dir, "processing", os.path.relpath(infile, visit_dir)
            )
            outfile, ext = os.path.splitext(outfile)
            exposure_annotation = args.exposure_time / clock_frequency * ureg.Unit("s")
            outfile += f"_{exposure_annotation.to_compact():~.0f}".replace(" ", "")

    return outfile


def _parse_args(arguments: List[str] = None) -> argparse.Namespace:
    default_exposure = 0.1  # s

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file",
        help="NeXus-like data file containing Tristan-standard event data.",
        metavar="input-file",
    )
    parser.add_argument(
        "-e",
        "--exposure-time",
        metavar="time",
        # nargs="+",
        # default=[int(default_exposure * clock_frequency)],
        default=int(default_exposure * clock_frequency),
        type=float,
        help=f"The size of each time bin in seconds.  "
        f"Defaults to {Q_(default_exposure, 's').to_compact():~.0f}.",
        action=_ClockCyclesAction,
    )
    parser.add_argument(
        "-o",
        "--output-file",
        metavar="filename",
        type=str,
        help="Output file name for the plotted histogram.",
    )

    return parser.parse_args(arguments)


if __name__ == "__main__":
    args = _parse_args()

    output_file = determine_output_file(args.input_file, args.output_file)

    try:
        with h5py.File(args.input_file, "r") as data:
            try:
                data[cue_id_key]
            except KeyError:
                try:
                    group = "/entry/data/data/"
                    _ = data[group + cue_id_key]
                except KeyError:
                    sys.exit(
                        "The input data appear to be invalid.\n"
                        "\tTristan-standard event data cannot be found."
                    )
            else:
                group = None

            fig = make_figure(data, group, args.exposure_time)
            fig.savefig(output_file)
            print(f"Histogram plot saved to\n\t{output_file}.svg")

    except OSError:
        sys.exit(f"Error, input file does not exist: {args.input_file}")
