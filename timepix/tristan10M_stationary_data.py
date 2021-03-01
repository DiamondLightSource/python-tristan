#!/usr/bin/env python3
"""
Convert Tristan10M stationary data from pumpprobe experiment into images.
"""

import argparse
import logging
import os

# import bitshuffle.h5
import h5py
import numpy as np

from timepix import coordinates, shutter_close, shutter_open, ttl_rising

# import time

# from nexgen.CopyNexusStructure import copy_nexus_from_timepix

# Initialize a logger
logger = logging.getLogger("Tristan10M_stat")

# Some checks on input and output file format
class CheckInputFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not values.endswith(".nxs"):
            parser.error("Please pass the NeXus-like .nxs file containing events data.")

        setattr(namespace, self.dest, values)


class CheckOutputFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        values, ext = os.path.splitext(values)
        if ext != ".h5":
            print(
                f"You specified an invalid file extension {ext} for the output "
                "image file.\n"
                f"The output images will be saved to {values}.h5 instead."
            )

        setattr(namespace, self.dest, f"{values}.h5")


# Parse command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "event_data",
    help="Input event mode data from timepix detector, Nexus file",
    action=CheckInputFile,
)
parser.add_argument(
    "num_bins",
    type=int,
    nargs="?",
    default=10,
    help="Number of bins after each trigger",
)
parser.add_argument(
    "image_file",
    action=CheckOutputFile,
    help="Output file to save reconstructed images, HDF5 file",
)

# Some useful functions
def discover_shutter_times(cues, cues_t):
    "Finds out the timestamps of the shutter open and close signals."
    _open = cues_t[np.flatnonzero(cues[()] == shutter_open)]
    _close = cues_t[np.flatnonzero(cues[()] == shutter_close)]
    assert len(_open) == len(_close)
    assert np.all(_open) and np.all(_close), "Trigger signals do not match"
    return _open[0].astype(int), _close[0].astype(int)


def discover_trigger_times(cues, cues_t, sh_open, sh_close):
    "Finds out the timestamps of ttl_rising triggers in between the shutter open and close timestamps."
    ttl_up = cues_t[np.flatnonzero(cues[()] == ttl_rising)]
    # All values are repeated once per module
    ttl_up = np.unique(ttl_up)
    ttl_up = ttl_up[(ttl_up > sh_open) & (ttl_up < sh_close)]
    return ttl_up.astype(int)


def calculate_time_per_frame(trigger_array, bins):
    diff = np.array([])
    for i in range(1, trigger_array.size):
        res = (trigger_array[i] - trigger_array[i - 1]) // bins
        diff = np.append(diff, res)
    assert np.all(diff), "Trigger timestamps are not evenly spaced"
    diff = np.unique(diff)
    return diff[0].astype(int)
    # res = (time_end - time_start) // bins
    # return int(res)


def get_valid_data(pos, t, sh_open, sh_close):
    pos = coordinates(pos)
    xyt = np.array([pos[0], pos[1], t])

    # Just a double check in case of out of order data (or last bins)
    xyt = xyt[:, (t > sh_open) & (t < sh_close)]
    return xyt


def make_histogram(xyt, img_shape, T, nbins):
    img_start = int(xyt[2].min() // T)
    # img_end = int(xyt[2].max() // T)
    img_count = nbins
    # img_count = img_end - img_start

    # Bin the events into images
    time_bounds = T * (img_start + np.array([0, img_count]))
    hist_ranges = ((0, img_shape[0]), (0, img_shape[1]), time_bounds)
    images, edges = np.histogramdd(
        xyt.T, range=hist_ranges, bins=img_shape + [img_count]
    )
    images = np.moveaxis(images, 2, 0)
    images = images.astype(np.uint32, copy=False)

    return images


def write_to_file(dset, images):
    start_dset = dset[:, :, :]
    dset[:, :, :] = start_dset + images


def create_images(event_data: h5py.File, num_bins, image_file: h5py.File):
    # Define logger
    logdir = os.path.dirname(image_file.filename)
    logging.basicConfig(
        filename=os.path.join(logdir, "binning.log"),
        format="%(message)s",
        level="DEBUG",
    )
    # logger.info()

    image_size = list(event_data["entry/instrument/detector/module/data_size"][()])
    # Data is to be found in vds (external link in "entry/data/data")
    data = event_data["entry/data/data"]
    # Look at cue messages
    cues = data["cue_id"]
    cues_time = data["cue_timestamp_zero"]
    # Get shutter times information
    _open, _close = discover_shutter_times(cues, cues_time)
    # Find triggers
    ttl_up = discover_trigger_times(cues, cues_time, _open, _close)
    # num_triggers = ttl_up.size

    # Events datasets
    event_id = data["event_id"]
    event_time = data["event_time_offset"]
    num_events = event_time.len()

    # Create output image dataset
    # block_size = 0
    dset = image_file.create_dataset(
        "data",
        shape=(num_bins, image_size[0], image_size[1]),
        dtype="i4",
        chunks=(1, image_size[0], image_size[1]),
        compression="lzf",
        # compression=bitshuffle.h5.H5FILTER,
        # compression_opts=(block_size, bitshuffle.h5.H5_COMPRESS_LZ4),
    )

    logger.info("Start binning")
    step = 200000  # to be used as "chunking"
    if num_events % step == 0:
        count = num_events // step
    else:
        count = (num_events // step) + 1
    # Find time per frame
    T = calculate_time_per_frame(ttl_up, num_bins)
    for n in range(count):
        pos = event_id[n * step : (n + 1) * step]
        t = event_time[n * step : (n + 1) * step]
        pos = coordinates(pos)
        xyt = np.array([pos[0], pos[1], t])
        xyt = xyt[:, (t > _open) & (t < _close)]
        for j in range(1, ttl_up.size):
            t_i = ttl_up[j - 1]
            t_f = ttl_up[j]
            xyt_tmp = xyt[:, (t > t_i) & (t < t_f)]
            if xyt_tmp.size == 0:
                continue
            print(j)
            images = make_histogram(xyt_tmp, image_size, T, num_bins)
            write_to_file(dset, images)
            print("images written")
        # TODO need to handle ttl[-1] to shutter_close

    # for n in range(ttl_up.size - 1):
    #    t_i = ttl_up[n]
    #    t_f = ttl_up[n + 1]
    #    # it should always be the same anyway
    #    time_per_frame = calculate_time_per_frame(t_i, t_f, num_bins)
    # histogram between these timestamps
    # copy metadata


if __name__ == "__main__":
    args = parser.parse_args()
    with h5py.File(args.event_data, "r") as f, h5py.File(args.image_file, "x") as g:
        create_images(f, args.num_bins, g)
