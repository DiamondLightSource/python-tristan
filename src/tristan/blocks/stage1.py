"""Extract information about modules and triggers."""
from __future__ import annotations

from pathlib import Path

# from ..data import (
#     cue_id_key,
#     cue_time_key,
#     event_location_key,
#     lvds_falling,
#     lvds_rising,
#     shutter_close,
#     shutter_open,
#     sync_falling,
#     sync_rising,
#     ttl_rising,
# )
# from ..diagnostics.utils import (
#     define_modules,
#     find_shutter_times,
#     find_trigger_timestamps,
#     gap_size,
#     image_size,
#     mod_size,
# )


def run_trigger_lookup():
    pass


def extract_tristan_info(input_file: Path | str, output_json: Path | str):
    D = {}
    print(D)

    if not isinstance(input_file, Path):
        input_file = Path(input_file).expanduser().resolve()
