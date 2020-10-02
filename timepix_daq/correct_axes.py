#!/usr/bin/env python
"""
Just a small utility to correct the phi/omega blunder in the nexus file.
Not tested on rotation data.
"""

import os
import sys

import h5py
import numpy
from h5py import AttributeManager


def get_attributes(obj, names, values):
    for n, v in zip(names, values):
        if type(v) is str:
            v = numpy.string_(v)
        AttributeManager.create(obj, name=n, data=v)


def run(filename):
    # Start by renaming the old file and use the filename for the new one
    name, ext = os.path.splitext(filename)
    old_file = name + "_old" + ext
    os.rename(filename, old_file)
    # Get name of vds file
    vds_file = name + "_vds.h5"
    with h5py.File(old_file, "r") as f, h5py.File(filename, "x") as g:
        nxentry = g.create_group("entry")
        get_attributes(nxentry, ("NX_class",), ("NXentry",))

        # Keep all but "entry/data" and "entry/sample" as is
        for k in f["entry"].keys():
            if k == "data" or k == "sample":
                continue
            f["entry"].copy(k, nxentry)

        # Deal with "entry/data"
        nxdata = nxentry.create_group("data")
        # External link
        try:
            with h5py.File(vds_file, "r") as vds:
                nxdata["data"] = h5py.ExternalLink(vds.filename, "/")
        except OSError:
            print(
                f"No {vds_file} file was found in directory. External link to data will not be written."
            )
        # Axes correction
        for k in f["entry/data"].keys():
            if k == "omega":
                ax = "phi"
                f["entry/data"].copy(k, nxdata, name="phi")
                get_attributes(
                    nxdata["phi"],
                    ("depends_on",),
                    ("/entry/sample/transformations/kappa",),
                )
            elif k == "phi":
                ax = "omega"
                f["entry/data"].copy(k, nxdata, name="omega")
                get_attributes(nxdata["omega"], ("depends_on",), (".",))

        get_attributes(nxdata, ("NX_class", "axes", "signal"), ("NXdata", ax, "data"))

        # Deal with "entry/sample"
        f["entry"].copy(
            "sample", nxentry, shallow=True
        )  # Copy just first level and attributes
        for k in f["entry/sample"]:
            s = "entry/sample/" + k
            if isinstance(f[s], h5py.Dataset):
                continue
            # Also the dependencies need to be switched!
            for k1 in f[s].keys():
                if "omega" in k:
                    if ax == "phi":
                        nxentry["sample/sample_phi/phi"] = nxdata["phi"]
                    else:
                        f[s].copy(k1, nxentry["sample/sample_phi"], name="phi")
                        get_attributes(
                            nxentry["sample/sample_phi/phi"],
                            ("depends_on",),
                            ("/entry/sample/transformations/kappa",),
                        )
                elif "phi" in k:
                    if ax == "omega":
                        nxentry["sample/sample_omega/omega"] = nxdata["omega"]
                    else:
                        f[s].copy(k1, nxentry["sample/sample_omega"], name="omega")
                        get_attributes(
                            nxentry["sample/sample_omega/omega"],
                            ("depends_on",),
                            (".",),
                        )
                else:
                    f[s].copy(k1, nxentry["sample/" + k])


if __name__ == "__main__":
    run(sys.argv[1])
