"""
Monitor a directory for new serial tristan nexus files and kick off diagnostic tool.
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path

from . import diagnostics_log as log
from .check_files import run_file_check
from .find_trigger_intervals import run_trigger_lookup

usage = "%(prog)s /path/to/dir/to/monitor nexus_filename [options]"

SSX_NXS_PATTERN = re.compile(r"(.*)_w(?:\d+).nxs")
DIR_TIMEOUT = 30
FILE_TIMEOUT = 600

# Define a logger object
logger = logging.getLogger("TristanDiagnostics.Monitor")


class TristanCollectionMonitor:
    def __init__(self, root_visit: Path, collection_num: str):
        self.root_visit = root_visit
        self.num = collection_num
        self.known_files = []

    def scan_visit_for_collection(self) -> Path | None:
        """Look for new tristan directory using collection number"""
        for el in self.root_visit.iterdir():
            if self.num in el.name and "trs" in el.name:
                return self.root_visit / el
        return None

    def scan_collection_for_nexus_file(self, collection_dir: Path) -> str | None:
        """Wait for all the nexus files in"""
        for filename in collection_dir.iterdir():
            m = SSX_NXS_PATTERN.fullmatch(filename.name)
            if m:
                if filename in self.known_files:
                    continue
                self.known_files.append(filename)
                return filename.name
        return None


def run_monitor(args, start_time):
    log.config()  # Just stream handler
    root_visit = Path(args.visit_dir).expanduser().resolve()
    outdir = args.outdir if args.outdir else root_visit / "processing/trigger_check"
    checked_files = []
    monitor = TristanCollectionMonitor(root_visit, args.num)
    try:
        while time.time() - start_time < DIR_TIMEOUT:
            collection_dir = monitor.scan_visit_for_collection()
            if collection_dir:
                print(f"Found {collection_dir}")
                time.sleep(1)
                break
            print("Waiting for collection directory to appear.")
            time.sleep(1)
        if not collection_dir:
            print(f"Giving up waiting for {collection_dir} after {DIR_TIMEOUT} seconds")
            return

        t0 = time.time()
        while time.time() - t0 < FILE_TIMEOUT:
            # Start looking for nexus files
            new_nxs = monitor.scan_collection_for_nexus_file(collection_dir)
            if new_nxs and (collection_dir / new_nxs) not in checked_files:
                print(f"Found new nexus file {new_nxs}")
                filename_root = new_nxs.stem.replace(f"_w%0{3}d", "")
                print("kicking off file check")
                run_file_check(
                    collection_dir,
                    filename_root,
                    outdir,
                    "10M",
                )
                print("Kicking off tristan triggering")
                # NOTE this filename here changed for reasons
                run_trigger_lookup(
                    collection_dir,
                    filename_root,
                    outdir,
                    "10M",
                    "standard",
                    new_nxs,
                )
                checked_files.append(new_nxs)
            print("Waiting for new nexus files")
            time.sleep(5)
    except KeyboardInterrupt:
        print("Exiting program, bye!")
        sys.exit()


def cli():
    parser = argparse.ArgumentParser(usage=usage, description=__doc__)
    parser.add_argument("visit_dir", type=str, help="The visit directory")
    parser.add_argument("num", type=str, help="Collection number")
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        help="""
        Output directory for tristan tools.
        If not passed, it will default to the current working directory.
        """,
    )
    parser.add_argument(
        "-e",
        "--expt",
        type=str,
        choices=["standard", "ssx"],
        default="standard",
        help="Specify the type of collection. Defaults to standard.",
    )
    args = parser.parse_args()
    start_time = time.time()
    run_monitor(args, start_time)
