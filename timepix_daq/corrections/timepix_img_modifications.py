import sys

import h5py
import numpy as np


def apply_modifications(nxs):
    # Modify scan axis, turn tuple into list
    phi = np.array([0.1 * p for p in range(-850, 650)])
    da = nxs["/entry/sample/sample_phi/phi"].attrs.items()
    del nxs["/entry/sample/sample_phi/phi"]
    nxs["/entry/sample/sample_phi/phi"] = phi
    for key, value in da:
        if key == "vector":
            value = [-1, 0, 0]
        nxs["/entry/sample/sample_phi/phi"].attrs.create(key, value)
    del nxs["/entry/data/phi"]
    nxs["/entry/data/phi"] = nxs["/entry/sample/sample_phi/phi"]
    del nxs["/entry/sample/transformations/phi"]
    nxs["/entry/sample/transformations/phi"] = nxs["/entry/sample/sample_phi/phi"]
    # Modify kappa vector (it's an attribute)
    kappa = nxs["entry/sample/sample_kappa/kappa"]
    for k, v in kappa.attrs.items():
        if k == "vector":
            kappa.attrs.create(k, [-0.766414, -0.642347, 0.0])
    del nxs["entry/sample/transformations/kappa"]
    nxs["entry/sample/transformations/kappa"] = nxs["entry/sample/sample_kappa/kappa"]
    # Modify omega vector
    omega = nxs["entry/sample/sample_omega/omega"]
    for k, v in omega.attrs.items():
        if k == "vector":
            omega.attrs.create(k, [-1, 0, 0])
    del nxs["entry/sample/transformations/omega"]
    nxs["entry/sample/transformations/omega"] = nxs["entry/sample/sample_omega/omega"]

    # Modify fast and slow axes (also an attribute)
    fast = nxs["entry/instrument/detector/module/fast_pixel_direction"]
    for k, v in fast.attrs.items():
        if k == "vector":
            fast.attrs.create(k, [0, -1, 0])
    slow = nxs["entry/instrument/detector/module/slow_pixel_direction"]
    for k, v in slow.attrs.items():
        if k == "vector":
            slow.attrs.create(k, [-1, 0, 0])
    # Recalculate module_offset
    x_pix = nxs["entry/instrument/detector/module/fast_pixel_direction"][()]
    y_pix = nxs["entry/instrument/detector/module/slow_pixel_direction"][()]
    beam_x = nxs["entry/instrument/detector/beam_center_x"][()]
    beam_y = nxs["entry/instrument/detector/beam_center_y"][()]
    x_scaled = beam_x * x_pix
    y_scaled = beam_y * y_pix
    det_origin = x_scaled * np.array([0, -1, 0]) + y_scaled * np.array([-1, 0, 0])
    det_origin = list(-det_origin)
    offset = nxs["entry/instrument/detector/module/module_offset"]
    for k, v in offset.attrs.items():
        if k == "offset":
            offset.attrs.create(k, det_origin)

    # Det_x and 2theta
    z = nxs["entry/instrument/detector_z/det_z"]
    for k, v in z.attrs.items():
        if k == "vector":
            z.attrs.create(k, [0, 0, 1])
    del nxs["/entry/instrument/transformations/det_z"]
    nxs["/entry/instrument/transformations/det_z"] = nxs[
        "entry/instrument/detector_z/det_z"
    ]
    t = nxs["entry/instrument/twotheta/twotheta"]
    for k, v in t.attrs.items():
        if k == "vector":
            t.attrs.create(k, [-1, 0, 0])
    del nxs["/entry/instrument/transformations/two_theta"]
    nxs["/entry/instrument/transformations/two_theta"] = nxs[
        "entry/instrument/twotheta/twotheta"
    ]

    print("All done!")


if __name__ == "__main__":
    with h5py.File(sys.argv[1], "r+") as fh:
        apply_modifications(fh)
