"""General utilities for the diagnostic tools."""

import numpy as np

# Some constants
timing_resolution_fine = 1.5625e-9

DIV = np.uint32(0x2000)

# Tristan 10M specs
tristan_modules = {"10M": (2, 5), "2M": (1, 2), "1M": (1, 1)}  # (H, V) -.> (fast, slow)
mod_size = (515, 2069)  # slow, fast
gap_size = (117, 45)  # slow, fast
image_size = (3043, 4183)  # slow, fast


def define_modules(num_modules: str = "10M") -> dict:
    """Define the start and end pixel of each module in the Tristan detector.

    Args:
        num_modules (str, optional): Specify how many modules make up the Tristan detector currently in use. Defaults to "10M".

    Returns:
        dict[tuple, tuple]: Start and end pixel value of each module - which are defined by a (x,y) tuple.
            For example a Tristan 1M will return {(0,0): {"10M": (2, 5), "2M": (1, 2), "1M": (1,1)}}
    """
    modules = tristan_modules[num_modules]
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
