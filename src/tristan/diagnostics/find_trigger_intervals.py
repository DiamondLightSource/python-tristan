"""
Run a quick check on trigger signals recorded in a Tristan collection.
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import time
from pathlib import Path

import h5py
import numpy as np

from ..command_line import version_parser
from ..data import (  # ttl_falling,
    cue_id_key,
    cue_time_key,
    lvds_falling,
    lvds_rising,
    shutter_close,
    shutter_open,
    sync_falling,
    sync_rising,
    ttl_rising,
)
from . import diagnostics_log as log
from .utils import TIME_RES, assign_files_to_modules, get_full_file_list

epilog_message = """
This program looks for shutter open and close signals and checks their timestamps.\n
Additionally, it calculates the time interval between rising and falling edge of each trigger in a Tristan collection:\n
    - TTL and LVDS for a standard time-resolved collection\n
    - TTL, LVDS and SYNC for time-resolved serial crystallography collection.\n
The results are written to a filename_TRIGGERCHECK.log.
"""

# Define parser
usage = "%(prog)s /path/to/data/dir filename_root [options]"
parser = argparse.ArgumentParser(
    usage=usage,
    formatter_class=argparse.RawTextHelpFormatter,
    description=__doc__,
    epilog=epilog_message,
    parents=[version_parser],
)
parser.add_argument("visitpath", type=str, help="Visit directory")
parser.add_argument("filename", type=str, help="Filename")
parser.add_argument(
    "-e",
    "--expt",
    type=str,
    choices=["standard", "ssx"],
    default="standard",
    help="Specify the type of collection. Defaults to standard.",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    help="""
    Output directory to save results
    If not passed, the script will default to current working directory.
    """,
)
parser.add_argument(
    "-n",
    "--nproc",
    type=int,
    help="The number of processes to use.",
)
parser.add_argument(
    "-trig",
    "--triggers",
    type=str,
    nargs="+",
    default="all",
    help="""
    Specify which triggers to look for.
    If not passed, will look at all the available ones for the experiment type.
    """,
)
parser.add_argument(
    "-m",
    "--num-modules",
    choices=["1M", "2M", "10M"],
    default="10M",
    type=str,
    help="Number of detector modules.",
)
parser.add_argument(
    "-nxs", "--nexus", type=str, help="Nexus filename if different from filename.nxs."
)


# Define a logger object
logger = logging.getLogger("TristanDiagnostics.TriggerTimes")


def setup_logging(wdir, filestem):
    logfile = wdir / (filestem + "_TRIGGERCHECK.log")
    log.config(logfile.as_posix())


def trigger_lookup(tristanlist):
    mod_number, filelist, expt = tristanlist
    sh_open = []
    sh_close = []
    ttl_re = []
    lvds_re = []
    lvds_fe = []
    sync_re = []
    sync_fe = []
    for filename in filelist:
        with h5py.File(filename) as fh:
            cues = fh[cue_id_key][()]
            cues_time = fh[cue_time_key]
            # Look for shutters
            op_idx = np.where(cues == shutter_open)[0]
            cl_idx = np.where(cues == shutter_close)[0]
            if len(op_idx) > 0:
                for i in range(len(op_idx)):
                    sh_open.append(cues_time[op_idx[i]] * TIME_RES)
            if len(cl_idx) > 0:
                for i in range(len(cl_idx)):
                    sh_close.append(cues_time[cl_idx[i]] * TIME_RES)
            # Look for lvds
            lvds_up_idx = np.where(cues == lvds_rising)[0]
            lvds_down_idx = np.where(cues == lvds_falling)[0]
            if len(lvds_up_idx) > 0:
                for i in range(len(lvds_up_idx)):
                    lvds_re.append(cues_time[lvds_up_idx[i]] * TIME_RES)
            if len(lvds_down_idx) > 0:
                for i in range(len(lvds_down_idx)):
                    lvds_fe.append(cues_time[lvds_down_idx[i]] * TIME_RES)
            # Look for ttl
            ttl_idx = np.where(cues == ttl_rising)[0]
            if len(ttl_idx) > 0:
                for i in range(len(ttl_idx)):
                    ttl_re.append(cues_time[ttl_idx[i]] * TIME_RES)
            if expt == "ssx":
                # Look for sync
                sync_up_idx = np.where(cues == sync_rising)[0]
                sync_down_idx = np.where(cues == sync_falling)[0]
                if len(sync_up_idx) > 0:
                    for i in range(len(sync_up_idx)):
                        sync_re.append(cues_time[sync_up_idx[i]] * TIME_RES)
                if len(sync_down_idx) > 0:
                    for i in range(len(sync_down_idx)):
                        sync_fe.append(cues_time[sync_down_idx[i]] * TIME_RES)

    D = {
        f"Module {mod_number}": {
            "num_files": len(filelist),
            "Shutter open": sh_open,
            "Shutter close": sh_close,
            "LVDS re": sorted(lvds_re),
            "LVDS fe": sorted(lvds_fe),
            "TTL re": sorted(ttl_re),
            "SYNC re": sorted(sync_re),
            "SYNC fe": sorted(sync_fe),
        }
    }
    return D


def log_full_summary(res: list[dict], expt: str):
    for el in res:
        for k, v in el.items():
            logger.info(f"--- {k} ---")
            if v["num_files"] == 0:
                logger.warning(
                    """
                    WARNING! There are no files for this module.
                    """
                )
                break
            shutters = [v["Shutter open"], v["Shutter close"]]
            logger.info("SHUTTERS")
            if len(shutters[0]) > 0 and len(shutters[1]) > 0:
                logger.info(f"Shutter open timestamp: {shutters[0][0]:.4f}.")
                logger.info(f"Shutter close timestamp: {shutters[1][0]:.4f}.")
                diff0 = shutters[1][0] - shutters[0][0]
                logger.info(
                    f"Total time between shutter opening and closing: {diff0:.4f} s."
                )
            elif len(shutters[0]) == 0 or len(shutters[1]) == 0:
                logger.warning("Missing shutter information!")
                logger.warning(
                    f"Number of shutter open timestamps found: {len(shutters[0])}."
                )
                logger.warning(
                    f"Number of shutter close timestamps found: {len(shutters[1])}."
                )
            logger.info("LVDS")
            if len(v["LVDS re"]) > 0 and len(v["LVDS fe"]) > 0:
                logger.info(
                    f"Found {len(v['LVDS re'])} rising edges and {len(v['LVDS fe'])} falling edges."
                )
                logger.info(f"LVDS rising edge timestamp: {v['LVDS re'][0]:.4f}.")
                logger.info(f"LVDS falling edge timestamp: {v['LVDS fe'][0]:.4f}.")
                diff1 = [b - a for a, b in zip(v["LVDS re"], v["LVDS fe"])]
                logger.info(
                    f"Time difference between re and fe signal: {diff1[0]:.4f} s."
                )
            else:
                logger.warning("Missing LVDS triggers!")
                logger.warning(
                    f"Number of LVDS re timestamps found: {len(v['LVDS re'])}"
                )
                logger.warning(
                    f"Number of LVDS fe timestamps found: {len(v['LVDS fe'])}"
                )
            logger.info("TTL")
            logger.info(f"Found {len(v['TTL re'])} rising edges.\n")
            if len(v["TTL re"]) > 0:
                logger.info(f"First TTL rising edge timestamp: {v['TTL re'][0]:.4f} .")
                logger.info(f"Last TTL rising edge timestamp: {v['TTL re'][-1]:.4f} .")
                d = [i - v["TTL re"][n - 1] for n, i in enumerate(v["TTL re"])][1:]
                d_avg = np.average(d) if len(d) else np.nan
                logger.info(
                    f"Average time interval between one TTL re and the next: {d_avg:.4} s"
                )
                if len(v["LVDS re"]) > 0:
                    diff2 = v["TTL re"][0] - v["LVDS re"][0]
                    logger.info(
                        f"Time difference between first TTL and LVDS re: {diff2:.4f} s."
                    )
                    before = np.where(v["TTL re"] < v["LVDS re"][0])[0]
                    if len(before) > 0:
                        logger.info(
                            f"{len(before)} TTL triggers found before LVDS rising edge."
                        )
                        new_re = [
                            el for n, el in enumerate(v["TTL re"]) if n not in before
                        ]
                        diff2_1 = new_re[0] - v["LVDS re"][0]
                        logger.info(
                            f"Time difference between LVDS re and first TTL after it: {diff2_1:.4f} s."
                        )
                if len(v["LVDS fe"]) > 0:
                    diff3 = v["LVDS fe"][0] - v["TTL re"][-1]
                    logger.info(
                        f"Time difference between LVDS fe and last TTL: {diff3:.4f} s."
                    )
                    after = np.where(v["TTL re"] > v["LVDS fe"][0])[0]
                    if len(after) > 0:
                        logger.info(
                            f"{len(after)} TTL triggers found before LVDS falling edge."
                        )
                        new_re = [
                            el for n, el in enumerate(v["TTL re"]) if n not in after
                        ]
                        diff3_1 = v["LVDS fe"][0] - new_re[-1]
                        logger.info(
                            f"Time difference between LVDS re and last TTL after it: {diff3_1:.4f} s."
                        )
            else:
                logger.warning("No TTL triggers found!")
            # If SSX, print out SYNC info.
            if expt == "ssx":
                logger.info("SYNC")
                if len(v["SYNC re"]) > 0 and len(v["SYNC fe"]) > 0:
                    logger.info(
                        f"Found {len(v['SYNC re'])} rising edges and {len(v['SYNC fe'])} falling edges."
                    )
                    logger.info(
                        f"First SYNC rising edge timestamp: {v['SYNC re'][0]:.4f}."
                    )
                    logger.info(
                        f"Last SYNC falling edge timestamp: {v['SYNC fe'][-1]:.4f}."
                    )
                    if v["SYNC re"][0] < shutters[0][0]:
                        logger.warning(
                            "First SYNC rising edge was recorded before the shutter open signal! \n"
                            f"Timestamp difference: {shutters[0][0] - v['SYNC re'][0]} s."
                        )
                    if v["SYNC fe"][-1] > shutters[1][0]:
                        logger.warning(
                            "Last SYNC falling edge was recorded after the shutter close signal! \n"
                            f"Timestamp difference: {v['SYNC fe'][-1] - shutters[1][0]} s."
                        )
                    diff4 = [b - a for a, b in zip(v["SYNC re"], v["SYNC fe"])]
                    avg4 = np.average(diff4) if len(diff4) else np.nan
                    logger.info(
                        f"Average time interval between SYNC re and fe: {avg4:.4f} s."
                    )
                    if len(v["TTL re"]) == len(v["SYNC re"]):
                        diff5 = [b - a for a, b in zip(v["TTL re"], v["SYNC re"])]
                        avg5 = np.average(diff5) if len(diff5) else np.nan
                        logger.info(
                            f"Average time interval between TTL re and SYNC re: {avg5:.4f} s."
                        )
                    else:
                        logger.error(
                            "The number of TTL re and SYNC re doesn't match. Impossible to accurately calculate time difference."
                        )
                elif len(v["SYNC re"]) == 0:
                    logger.warning("No SYNC rising edges found!")
                    if len(v["SYNC fe"]) == 0:
                        logger.warning("No SYNC falling edges found!")
                elif len(v["SYNC fe"]) == 0:
                    logger.warning("No SYNC falling edges found!")


def log_only_requested_trigger_info(res: list[dict], trigger_request: list[str]):
    for el in res:
        for k, v in el.items():
            logger.info(f"--- {k} ---")
            if v["num_files"] == 0:
                logger.warning(
                    """
                    WARNING! There are no files for this module.
                    """
                )
                break

            if "LVDS" in trigger_request:
                logger.info("LVDS")
                if len(v["LVDS re"]) > 0 and len(v["LVDS fe"]) > 0:
                    logger.info(
                        f"Found {len(v['LVDS re'])} rising edges and {len(v['LVDS fe'])} falling edges."
                    )
                    logger.info(f"LVDS rising edge timestamp: {v['LVDS re'][0]:.4f}.")
                    logger.info(f"LVDS falling edge timestamp: {v['LVDS fe'][0]:.4f}.")
                else:
                    logger.warning("Missing LVDS triggers!")
                    if len(v["LVDS re"]) == 0:
                        logger.warning("No LVDS rising edges found!")
                    if len(v["LVDS fe"]) == 0:
                        logger.warning("No LVDS falling edges found!")
            if "TTL" in trigger_request:
                logger.info("TTL")
                logger.info(f"Found {len(v['TTL re'])} rising edges.\n")
                if len(v["TTL re"]) > 0:
                    logger.info(
                        f"First TTL rising edge timestamp: {v['TTL re'][0]:.4f} ."
                    )
                    logger.info(
                        f"Last TTL rising edge timestamp: {v['TTL re'][-1]:.4f} ."
                    )
                else:
                    logger.warning("No TTL triggers found!")
            if "SYNC" in trigger_request:
                logger.info("SYNC")
                if len(v["SYNC re"]) > 0 and len(v["SYNC fe"]) > 0:
                    logger.info(
                        f"Found {len(v['SYNC re'])} rising edges and {len(v['SYNC fe'])} falling edges."
                    )
                    logger.info(
                        f"First SYNC rising edge timestamp: {v['SYNC re'][0]:.4f}."
                    )
                    logger.info(
                        f"Last SYNC falling edge timestamp: {v['SYNC fe'][-1]:.4f}."
                    )
                else:
                    logger.warning("Missing SYNC triggers!")
                    if len(v["SYNC re"]) == 0:
                        logger.warning("No SYNC rising edges found!")
                    if len(v["SYNC fe"]) == 0:
                        logger.warning("No SYNC falling edges found!")


def main(args):
    filepath = Path(args.visitpath).expanduser().resolve()
    base = args.filename + f"_{6*'[0-9]'}.h5"

    # Current working directory
    if args.output:
        wdir = Path(args.output).expanduser().resolve()
        wdir.mkdir(exist_ok=True)
    else:
        wdir = Path.cwd()

    # Define stream handler
    setup_logging(wdir, filepath.stem)

    # Start logging
    logger.info(f"Current working directory: {wdir}")
    logger.info(f"Collection directory: {filepath}")
    logger.info(f"Filename root: {args.filename}")
    filename_template = filepath / base
    file_list = get_full_file_list(filename_template)
    logger.info(f"Found {len(file_list)} files in directory.\n")

    logger.info(
        f"Look for triggers in cue messages for a Tristan{args.num_modules} {args.expt} collection."
    )

    nxsfile = (
        filepath / (args.filename + ".nxs") if not args.nexus else filepath / args.nexus
    )
    if nxsfile in filepath.iterdir():
        with h5py.File(nxsfile) as nxs:
            count_time = nxs["/entry/instrument/detector/count_time"][()]
        logger.info(
            f"Total collection time recorded in NeXus file ({nxsfile.name}): {count_time} s.\n"
        )

    if args.nproc:
        nproc = args.nproc
    else:
        nproc = mp.cpu_count() - 1

    L, _ = assign_files_to_modules(file_list, args.num_modules, "cues")
    tristanlist = [l + (args.expt,) for l in list(L.items())]  # noqa: E741

    logger.info(f"Start Pool with {nproc} processes.")
    with mp.Pool(processes=nproc) as pool:
        res = pool.map(trigger_lookup, tristanlist)
    logger.info("\n")

    logger.info("----- SUMMARY -----")
    if args.triggers == "all":
        log_full_summary(res, args.expt)
    else:
        log_only_requested_trigger_info(res, args.triggers)
    logger.info("\n")


def cli():
    tic = time.time()
    args = parser.parse_args()
    main(args)
    toc = time.time()
    logger.debug(f"Total time taken: {toc - tic:4f} s.")
    logger.info("~~~ EOF ~~~")
