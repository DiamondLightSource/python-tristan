#!/usr/bin/env python

import sys

import h5py


def run(nxs):
    z = nxs["entry/instrument/detector_z/det_z"]
    for k, v in z.attrs.items():
        if k == "vector":
            z.attrs.create(k, [0, 0, 1])
    del nxs["entry/instrument/transformations/det_z"]
    del nxs["entry/sample/transformations/det_z"]
    nxs["entry/instrument/transformations/det_z"] = nxs[
        "entry/instrument/detector_z/det_z"
    ]
    nxs["entry/sample/transformations/det_z"] = nxs["entry/instrument/detector_z/det_z"]
    print("Done!")


if __name__ == "__main__":
    with h5py.File(sys.argv[1], "r+") as fh:
        run(fh)
