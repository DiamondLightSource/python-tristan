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


def copy_nexus_structure(h5_out: h5py.File, h5_in: h5py.File, nxs: h5py.File):
    # Create first level with attributes
    nxentry = nxs.create_group("entry")
    _get_attributes(nxentry, ("NX_class",), ("NXentry",))

    # Copy all of the nexus tree as it is except for /entry/data
    for k in h5_in["entry"].keys():
        if k == "data":
            continue
        h5_in["entry"].copy(k, nxentry)

    # Write NXdata group
    nxdata = nxentry.create_group("data")
    # Axes
    _ax = None
    for k in h5_in["entry/data"].keys():
        if "data" in k:
            continue
        if "event" in k:
            continue
        if "cue" in k:
            continue
        h5_in["entry/data"].copy(k, nxdata)
        _ax = k
    _get_attributes(nxdata, ("NX_class", "axes", "signal"), ("NXdata", _ax, "data"))

    # Add link to data
    data = nxdata.create_group("data")
    for k in h5_out.keys():
        data[k] = h5py.ExternalLink(h5_out.filename, k)
    # nxdata["data"] = h5py.ExternalLink(h5_out.filename, "/")

    # Close everything


if __name__ == "__main__":
    copy_nexus_structure(*sys.argv[1:])
