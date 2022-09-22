"""General utilities for the diagnostic tools."""

import numpy as np

# Some constants
timing_resolution_fine = 1.5625e-9

DIV = np.uint32(0x2000)

# Tristan 10M specs
modules = (2, 5)  # (H, V) -.> (fast, close)
mod_size = (515, 2069)  # slow, fast
gap_size = (117, 45)  # slow, fast
image_size = (3043, 4183)  # slow, fast


def define_modules():
    """_summary_

    Returns:
        _type_: _description_
    """
    mod = {}
    for _y in range(modules[0]):
        for _x in range(modules[1]):
            int_x = [
                _x * (mod_size[0] + gap_size[0]),
                _x * (mod_size[0] + gap_size[0]) + mod_size[0],
            ]
            int_y = [
                _y * (mod_size[1] + gap_size[1]),
                _y * (mod_size[1] + gap_size[1]) + mod_size[1],
            ]
            mod[(_x, _y)] = (int_x, int_y)
    return mod
