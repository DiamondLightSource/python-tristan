#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
.
"""

import argparse

import h5py
import numpy as np
from make_nxs import CopyNexusStructure

# Trigger messages
shutter_open = 0x840
shutter_close = 0x880
ttl_rising = 0x8E9

# Detector clock frequency
clock_freq = int(6.4e8)


# Parse command line arguments
def timepix_parser():
    parser = argparse.ArgumentParser(
        description="Generates images from event-mode data."
    )
    parser.add_argument(
        "event_data",
        nargs=1,
        help="Input event mode data from timepix detector, Nexus file",
    )
    parser.add_argument(
        "exposure_time",
        type=float,
        nargs="?",
        default=0.1,
        help="Image exposure time, float",
    )
    parser.add_argument(
        "step",
        type=int,
        nargs="?",
        default=100000,
        help="Because data is not chunked, define a step to load this many events at "
        "a time.",
    )
    parser.add_argument(
        "image_file",
        nargs=1,
        help="Output file to save reconstructed images, HDF5 file",
    )

    args = parser.parse_args()
    # TBD: make image_file optional, give warning and just not write file if nothing
    # is passed

    Timepix2MImageConverter(
        args.event_data, args.exposure_time, args.step, args.image_file
    ).run()


class Timepix2MImageConverter(object):
    """
    .
    """

    def __init__(self, event_data, exposure_time, step, image_file):
        self._nxs = event_data[0]
        # Check that the NeXus file has been passed
        assert self._nxs.endswith(".nxs"), "Please pass the NeXus file"

        # Open input file
        self._fin = h5py.File(self._nxs, "r")
        self._data = self._fin["entry/data/data"]

        # Get cue messages information
        self._cues = self._data["cue_id"]
        self._cues_time = self._data["cue_timestamp_zero"]
        self._num_cues = self._cues_time.len()

        # Position (id) and timestamp of events
        self._d_pos = self._data["event_id"]
        self._d_time = self._data["event_time_offset"]
        self._num_events = self._d_time.len()

        # Shape of the detector
        self._shape = list(self._fin["entry/instrument/detector/module/data_size"][()])

        # Exposure time
        self._exposure_time = exposure_time
        self._expT = int(self._exposure_time * clock_freq)

        # Define a step (in lieu of chunking)
        self._step = step

        # Open output file
        self._fout = h5py.File(image_file[0].split(".")[0] + ".h5", "x")

        # Discover shutter times
        self._open, self._close = self._discover_shutter_times()

        # Discover ttl_rising trigger times
        self._ttl = self._discover_trigger_times()
        # assert (
        #     len(self._ttl) > 0
        # ), "No ttl_rising trigger messages found in either module."

    def _discover_shutter_times(self):
        """Finds out the timestamps of the shutter open and close signals."""
        cues = self._cues
        cues_t = self._cues_time
        # There should be only one open and one close trigger.
        # Since we have 2 modules, we should have 2 triggers with the same timestamp.
        _open = cues_t[np.flatnonzero(cues[()] == shutter_open)]
        _close = cues_t[np.flatnonzero(cues[()] == shutter_close)]
        assert len(_open) == len(_close)
        assert len(_open) == 2, str(len(_open))
        assert np.all(_open) and np.all(_close), "Trigger signals do not match"
        return _open[0].astype(int), _close[0].astype(int)

    def _discover_trigger_times(self):
        """ Finds out the timestamps of ttl_rising triggers."""
        cues = self._cues
        cues_t = self._cues_time
        ttl_up = cues_t[np.flatnonzero(cues[()] == ttl_rising)]
        # All values are repeated twice (once per module)
        ttl_up = np.unique(ttl_up)
        return ttl_up.astype(int)

    def trigger_time_difference(self):
        """ Find the maximum and minimum time between triggers."""
        pass

    def write_to_file(self, dset, img):
        """ Write images to HDF5 file. """
        n = img.shape[0]
        dset.resize(dset.shape[0] + n, axis=0)
        dset[-n:] = img

        self._fout.flush()

    def get_data(self, _pos, _time):
        """ Get event position and time."""
        x = _pos & 0x1FFF
        y = _pos >> 13
        t = _time

        xyt = np.array([x, y, t])
        # Remove anything that falls before or after shutter signal
        xyt = xyt[:, (_time > self._open) & (_time < self._close)]
        return xyt

    def get_time_interval(self, ttl_up):
        """
        Get interval around laser pulse that we want to observe

        0.8s before pulse to 4s after.
        """
        inizio = ttl_up - (0.8 * clock_freq)
        fine = ttl_up + (4.0 * clock_freq)
        # Check that it doesn't overlap with shutter open and shutter close signal
        # In that case, set those as limits
        if self._open > inizio:
            inizio = self._open
        if self._close < fine:
            fine = self._close
        return inizio[0].astype(int), fine[0].astype(int)

    def make_histogram(self, xyt):
        expT = self._expT
        img_shape = self._shape

        img_start = int(xyt[2].min() // expT)
        img_end = -(-int(xyt[2].max()) // expT)
        img_count = img_end - img_start

        # Bin the events into images
        time_bounds = expT * (img_start + np.array([0, img_count]))
        hist_ranges = ((0, img_shape[0]), (0, img_shape[1]), time_bounds)
        images, edges = np.histogramdd(
            xyt.T, range=hist_ranges, bins=img_shape + [img_count]
        )
        images = np.moveaxis(images, 2, 0)
        images = images.astype(np.uint32, copy=False)

        return images, edges

    def create_images(self):
        """ Generate images from events """
        # step = self._step

        ttl_up = self._ttl

        pos_dset = self._d_pos
        time_dset = self._d_time

        #if self._num_events < step:
        # TBD: later add calc to go through all data at the same time
        #    break
        if self._num_events % step == 0:
            count = self._num_events // step
        else:
            count = (self._num_events // step) + 1

        # There should only be one laser pulse in this one experiment

        if len(ttl_up) == 1:
            # Create dataset in output file
            dset = self._fout.create_dataset(
                "laserpulse_0",
                shape=(0, self._shape[0], self._shape[1]),
                maxshape=(None, self._shape[0], self._shape[1]),
                dtype="i4",
                chunks=(1, self._shape[0], self._shape[1]),
                compression="lzf",
            )
            # Get interval of interest around pulse
            t_i, t_f = self.get_time_interval(ttl_up[0])
            # Bin images
            _pos = pos_dset[()]
            _time = time_dset[()]
            xyt = self.get_data(_pos, _time)
            xyt = xyt[:, (xyt[2] > t_i) & (xyt[2] < t_f)]
            img = self.make_histogram(xyt)
            img, bins = self.make_histogram(xyt)
            self.write_to_file(dset, img)
        elif len(ttl_up) == 0:
            print("WARNING: No laser pulse")
            # Create dataset in output file
            dset = self._fout.create_dataset(
                "laserpulse_0",
                shape=(0, self._shape[0], self._shape[1]),
                maxshape=(None, self._shape[0], self._shape[1]),
                dtype="i4",
                chunks=(1, self._shape[0], self._shape[1]),
                compression="lzf",
            )
            _pos = pos_dset[()]
            _time = time_dset[()]
            xyt = self.get_data(_pos, _time)
            img = self.make_histogram(xyt)
            self.write_to_file(dset, img)
        # return bins

        # else:
        # there is more than one pulse
        # check actual time difference between pulses

    def run(self):
        self.create_images()
        # bins = self.create_images()
        # Copy metadata
        CopyNexusStructure(self._fout.filename, self._nxs).write()
        # CopyNexusStructure(self._fout.filename, self._nxs, bins).write()
        # Close everything
        self._fout.close()
        self._fin.close()


if __name__ == "__main__":
    timepix_parser()
