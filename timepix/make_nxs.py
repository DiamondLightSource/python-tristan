#!/usr/bin/env python3

import sys

import h5py
import numpy
from h5py import AttributeManager


def _get_attributes(obj, names, values):
    for n, v in zip(names, values):
        if type(v) is str:
            v = numpy.string_(v)
        AttributeManager.create(obj, name=n, data=v)


class CopyNexusStructure(object):
    """
    Class to copy nexus tree from one file to another
    """

    def __init__(self, h5_out, h5_in):
        self._fin = h5py.File(h5_in, "r")
        self._nxs = h5py.File(h5_out.split(".")[0] + ".nxs", "x")
        self._fout = h5py.File(h5_out, "r")

    def write(self):
        # Create first level with attributes
        nxentry = self._nxs.create_group("entry")
        _get_attributes(nxentry, ("NX_class",), ("NXentry",))

        # Copy all of the nexus tree as it is except for /entry/data
        for k in self._fin["entry"].keys():
            if k == "data":
                continue
            self._fin["entry"].copy(k, nxentry)

        # Write NXdata group
        nxdata = nxentry.create_group("data")
        # Axes
        _ax = None
        for k in self._fin["entry/data"].keys():
            if "data" in k:
                continue
            if "event" in k:
                continue
            if "cue" in k:
                continue
            self._fin["entry/data"].copy(k, nxdata)
            _ax = k
        _get_attributes(nxdata, ("NX_class", "axes", "signal"), ("NXdata", _ax, "data"))

        # Add link to data
        nxdata["data"] = h5py.ExternalLink(self._fout.filename, "/")

        # Close everything
        self._fin.close()
        self._fout.close()
        self._nxs.close()


if __name__ == "__main__":
    CopyNexusStructure(sys.argv[1], sys.argv[2]).write()
