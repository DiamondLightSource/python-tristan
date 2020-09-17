#!/usr/bin/env python
# coding: utf-8

"""Create a time histogram of all events on the detector and plot it in SVG format."""

import argparse
import os
import sys
from typing import Optional, Sequence

import h5py
import numpy as np
import pint
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from timepix import (
    clock_frequency,
    coordinates,
    cue_id_key,
    cue_times,
    event_location_key,
    event_time_key,
    first_cue_time,
    fullpath,
    seconds,
    shutter_close,
    shutter_open,
    size_key,
    ttl_rising,
)

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity


def select_roi(data_file, events_group, selection, labels) -> np.ndarray:
    size = data_file.get(size_key)
    x1, x2 = selection[:2].sort()
    y1, y2 = selection[2:].sort()
    roi = [x1, x2, y1, y2]
    if size:
        tests = np.repeat([range(0, size[0]), range(0, size[1])], 2)
        if not all(value in bounds for value, bounds in zip(roi, tests)):
            one_by_one = zip(roi, [np.repeat(size, 2)], labels)
            for item, top, label in one_by_one:
                if item < 0:
                    print(f"{label} is too small, it falls outside the image.")
                    size = False
                if item >= top:
                    print(f"{label} is too large, it falls outside the image.")
                    size = False
            sys.exit("The image size is " + " × ".join(size))

    else:
        print(
            "Warning:  The input data file does not contain information "
            "about the detector size.  Your region of interest bounds "
            "cannot be checked for consistency."
        )

    x, y = coordinates(data_file[events_group + event_location_key])
    index = np.argnonzero(x1 <= x & x <= x2 & y1 <= y & y <= y2)

    if not index.size:
        sys.exit("The region of interest contains no events.")

    return index


def make_figure(
    data_file: h5py.File,
    events_group: str,
    exposure_time: int,
    selection: np.ndarray = Ellipsis,
) -> Figure:
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
        selection,
    )

    return figure


def plot_histogram(
    events: Sequence[int],
    start: int,
    end: int,
    exposure_time: int,
    pulses: Sequence[int] = None,
    selection: np.ndarray = Ellipsis,
) -> (Figure, Axes):
    import matplotlib

    matplotlib.use("svg")

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()

    # If there are any laser pulses, align a bin edge with the first one.
    if np.array(pulses).size:
        pre_trigger_bins = np.arange(pulses[0] - exposure_time, start, -exposure_time)
        post_trigger_bins = np.arange(pulses[0], end, exposure_time)
        bins = np.concatenate([pre_trigger_bins[::-1], post_trigger_bins])
        start = pulses[0]
        bins = seconds(bins, start)
        ax.set_xlabel("Time from first laser pulse (seconds)")
    else:
        bins = np.arange(0, end - start, exposure_time)
        bins = seconds(bins)
        pulses = np.array([])
        ax.set_xlabel("Time from detector shutter open (seconds)")

    # Plot the laser pulses as vertical red lines.
    for pulse in seconds(pulses, start):
        ax.axvline(pulse, color="r")

    # Plot the histogram.
    ax.hist(seconds(events[selection].astype(int), start), bins)

    ax.set_title(f"Exposure time: {Q_(seconds(exposure_time), 's').to_compact():~.0f}")
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 3))
    ax.set_ylabel("Number of events")

    return fig, ax


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


class _Formatter(argparse.RawTextHelpFormatter):
    pass


default_exposure = 0.1  # s

_parser = argparse.ArgumentParser(description=__doc__, formatter_class=_Formatter)
_parser.add_argument(
    "input_file",
    help="HDF5 data file containing Tristan-standard event data.\n"
    "Typically this is either the virtual data set (VDS) '.h5' file written "
    "by the detector or the NeXus-like '.nxs' file written by the data "
    "acquisition system.",
    metavar="input-file",
)
_parser.add_argument(
    "-e",
    "--exposure-time",
    metavar="time",
    # nargs="+",
    # default=[int(default_exposure * clock_frequency)],
    default=int(default_exposure * clock_frequency),
    help="The size of each time bin in seconds.\n"
    f"Defaults to {Q_(default_exposure, 's').to_compact():~.0f}.",
    action=_ClockCyclesAction,
)
_parser.add_argument(
    "-o",
    "--output-file",
    metavar="filename",
    help="Output file name for the plotted histogram.",
)
select = _parser.add_argument(
    "-s",
    "--selection",
    "--roi",
    metavar=("x₁", "x₂", "y₁", "y₂"),
    type=int,
    nargs=4,
    help="Bounds of a region of interest.\n"
    "Perform the histogram only on data falling on pixels with\n"
    "\tx-position: x₁,₂ ≤ x ≤ x₂,₁;\n"
    "\ty-position: y₁,₂ ≤ y ≤ y₂,₁.\n"
    "Values will automatically be rearranged in size order so you need only make "
    "sure to pass the x-coordinates before the y-coordinates.",
)


if __name__ == "__main__":
    args = _parser.parse_args()

    output_file = determine_output_file(args.input_file, args.output_file)

    try:
        with h5py.File(args.input_file, "r") as data:
            if data.get(cue_id_key):
                group = ""
            elif data.get("/entry/data/data/" + cue_id_key):
                group = "/entry/data/data/"
            else:
                sys.exit(
                    "The input data appear to be invalid.\n"
                    "Tristan-standard event data cannot be found in the "
                    "expected locations."
                )

            if args.selection:
                index = select_roi(data, group, args.selection, select.metavar)
            else:
                index = Ellipsis

            fig = make_figure(data, group, args.exposure_time, index)
            fig.savefig(output_file)
            print(f"Histogram plot saved to\n\t{output_file}.svg")

    except OSError:
        sys.exit(f"Error, input file does not exist: {args.input_file}")
