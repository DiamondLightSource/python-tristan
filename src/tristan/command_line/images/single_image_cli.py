import sys
from operator import mul

import h5py
import numpy as np
from dask import array as da
from dask.diagnostics import ProgressBar
from hdf5plugin import Bitshuffle
from nexgen.nxs_copy import copy_tristan_nexus

from tristan.binning import find_start_end
from tristan.command_line import check_output_file, data_files
from tristan.command_line.images import determine_image_size
from tristan.data import (
    cue_keys,
    event_location_key,
    event_time_key,
    latrd_data,
    pixel_index,
    valid_events,
)


def main(args):
    """Utility for making a single image from event-mode data."""
    write_mode = "w" if args.force else "x"
    output_file = check_output_file(
        args.output_file, args.stem, "single_image", args.force
    )
    input_nexus = args.data_dir / f"{args.stem}.nxs"
    if input_nexus.exists():
        try:
            # Write output NeXus file if we have an input NeXus file.
            output_nexus = copy_tristan_nexus.single_image_nexus(
                output_file, input_nexus, write_mode=write_mode
            )
        except FileExistsError:
            sys.exit(
                f"This output file already exists:\n\t"
                f"{output_file.with_suffix('.nxs')}\n"
                "Use '-f' to override, "
                "or specify a different output file path with '-o'."
            )
    else:
        output_nexus = None
        print(
            "Could not find a NeXus file containing experiment metadata.\n"
            "Resorting to writing raw image data without accompanying metadata."
        )

    image_size = args.image_size or determine_image_size(input_nexus)

    raw_files = data_files(args.data_dir, args.stem)

    print("Finding detector shutter open and close times.")
    with latrd_data(raw_files, keys=cue_keys) as data, ProgressBar():
        start, end = find_start_end(data)

    print("Binning events into a single image.")
    with latrd_data(raw_files, keys=(event_location_key, event_time_key)) as data:
        data = valid_events(data, start, end)
        data[event_location_key] = pixel_index(data[event_location_key], image_size)
        image = da.bincount(data[event_location_key], minlength=mul(*image_size))
        image = image.astype(np.uint32).reshape(1, *image_size)

        with ProgressBar(), h5py.File(output_file, write_mode) as f:
            data_set = f.require_dataset(
                "data",
                shape=image.shape,
                dtype=image.dtype,
                chunks=image.chunksize,
                **Bitshuffle(),
            )
            image.store(data_set)

    print(f"Image written to\n\t{output_nexus or output_file}")
