"""
Run a quick check on trigger signals recorded in a Tristan collection.

Look for shutter open and close signals and check their timestamps.
Calculate the time interval between rising and falling edge of each trigger.
"""
import argparse
import glob
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
from . import timing_resolution_fine

# Define a logger object
logger = logging.getLogger("TristanDiagnostics.TriggerTimes")

# Define parser
parser = argparse.ArgumentParser(description=__doc__, parents=[version_parser])
parser.add_argument("visitpath", type=str, help="Visit directory")
parser.add_argument("filename", type=str, help="Filename")
parser.add_argument(
    "-e",
    "--expt",
    type=str,
    choices=["standard", "ssx"],
    default="standard",
    help="Specify the type of collection.",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    help="Output directory to save results. If not passed, the script will default to current working directory.",
)
parser.add_argument(
    "-n",
    "--nproc",
    type=int,
    # default=1,
    help="",
)


def trigger_lookup(tristanlist):
    mod_number, filelist = tristanlist
    sh_open = []
    sh_close = []
    ttl_re = []
    lvds_re = []
    lvds_fe = []
    for filename in filelist:
        with h5py.File(filename, "r") as fh:
            cues = fh[cue_id_key][()]
            cues_time = fh[cue_time_key]
            # Look for shutters
            op_idx = np.where(cues == shutter_open)[0]
            cl_idx = np.where(cues == shutter_close)[0]
            if len(op_idx) > 0:
                for i in range(len(op_idx)):
                    sh_open.append(cues_time[op_idx[i]] * timing_resolution_fine)
            if len(cl_idx) > 0:
                for i in range(len(cl_idx)):
                    sh_close.append(cues_time[cl_idx[i]] * timing_resolution_fine)
            # Look for lvds
            lvds_up_idx = np.where(cues == lvds_rising)[0]
            lvds_down_idx = np.where(cues == lvds_falling)[0]
            if len(lvds_up_idx) > 0:
                for i in range(len(lvds_up_idx)):
                    lvds_re.append(cues_time[lvds_up_idx[i]] * timing_resolution_fine)
            if len(lvds_down_idx) > 0:
                for i in range(len(lvds_down_idx)):
                    lvds_fe.append(cues_time[lvds_down_idx[i]] * timing_resolution_fine)
            # Look for ttl
            ttl_idx = np.where(cues == ttl_rising)[0]
            if len(ttl_idx) > 0:
                for i in range(len(ttl_idx)):
                    ttl_re.append(cues_time[ttl_idx[i]] * timing_resolution_fine)

    D = {
        f"Module {mod_number}": {
            "Shutter open": sh_open,
            "Shutter close": sh_close,
            "LVDS re": lvds_re,
            "LVDS fe": lvds_fe,
            "TTL re": ttl_re,
        }
    }
    return D


def trigger_lookup_ssx(tristanlist):
    mod_number, filelist = tristanlist
    sh_open = []
    sh_close = []
    ttl_re = []
    lvds_re = []
    lvds_fe = []
    sync_re = []
    sync_fe = []
    for filename in filelist:
        with h5py.File(filename, "r") as fh:
            cues = fh[cue_id_key][()]
            cues_time = fh[cue_time_key]
            # Look for shutters
            op_idx = np.where(cues == shutter_open)[0]
            cl_idx = np.where(cues == shutter_close)[0]
            if len(op_idx) > 0:
                for i in range(len(op_idx)):
                    sh_open.append(cues_time[op_idx[i]] * timing_resolution_fine)
            if len(cl_idx) > 0:
                for i in range(len(cl_idx)):
                    sh_close.append(cues_time[cl_idx[i]] * timing_resolution_fine)
            # Look for lvds
            lvds_up_idx = np.where(cues == lvds_rising)[0]
            lvds_down_idx = np.where(cues == lvds_falling)[0]
            if len(lvds_up_idx) > 0:
                for i in range(len(lvds_up_idx)):
                    lvds_re.append(cues_time[lvds_up_idx[i]] * timing_resolution_fine)
            if len(lvds_down_idx) > 0:
                for i in range(len(lvds_down_idx)):
                    lvds_fe.append(cues_time[lvds_down_idx[i]] * timing_resolution_fine)
            # Look for ttl
            ttl_idx = np.where(cues == ttl_rising)[0]
            if len(ttl_idx) > 0:
                for i in range(len(ttl_idx)):
                    ttl_re.append(cues_time[ttl_idx[i]] * timing_resolution_fine)
            # Look for sync
            sync_up_idx = np.where(cues == sync_rising)
            sync_down_idx = np.where(cues == sync_falling)
            if len(sync_up_idx) > 0:
                for i in range(len(sync_up_idx)):
                    sync_re.append(cues_time[sync_up_idx[i]] * timing_resolution_fine)
            if len(sync_down_idx) > 0:
                for i in range(len(sync_down_idx)):
                    sync_fe.append(cues_time[sync_down_idx[i]] * timing_resolution_fine)

    D = {
        f"Module {mod_number}": {
            "Shutter open": sh_open,
            "Shutter close": sh_close,
            "LVDS re": lvds_re,
            "LVDS fe": lvds_fe,
            "TTL re": ttl_re,
            "SYNC re": sync_re,
            "SYNC fe": sync_fe,
        }
    }
    return D


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
    logfile = wdir / (filepath.stem + "_TRIGGERCHECK.log")
    log.config(logfile.as_posix())

    # Start logging
    logger.info(f"Current working directory: {wdir}")
    logger.info(
        f"Look for triggers in cue messages for a Tristan10M {args.expt} collection."
    )
    logger.info(f"Collection directory: {filepath}")
    logger.info(f"Filename root: {args.filename}")
    filename_template = filepath / base
    file_list = [
        Path(f).expanduser().resolve()
        for f in sorted(glob.glob(filename_template.as_posix()))
    ]
    logger.info(f"Found {len(file_list)} files in directory.\n")
    # For now let's just go with the usual assumption that files are coherently divided.
    # TODO what if they're not?!?!
    L = [file_list[i : i + 10] for i in range(0, len(file_list), 10)]

    nxsfile = filepath / (args.filename + ".nxs")
    if nxsfile in filepath.iterdir():
        with h5py.File(nxsfile, "r") as nxs:
            count_time = nxs["/entry/instrument/detector/count_time"][()]
        logger.info(f"Total collection time recorded in NeXus file: {count_time} s.\n")

    if args.nproc:
        nproc = args.nproc
    else:
        nproc = mp.cpu_count() - 1
    # if args.nproc >= mp.cpu_count():
    #     nproc = mp.cpu_count() -1
    # else:
    #     nproc = args.nproc

    tristanlist = [(n, l) for n, l in enumerate(L)]

    logger.info(f"Start Pool with {nproc} processes.")
    with mp.Pool(processes=nproc) as pool:
        func = trigger_lookup if args.expt == "standard" else trigger_lookup_ssx
        res = pool.map(func, tristanlist)
    logger.info("\n")

    logger.info("----- SUMMARY -----")
    for el in res:
        for k, v in el.items():
            shutters = [v["Shutter open"], v["Shutter close"]]
            logger.info(f"--- {k} ---")
            logger.info("SHUTTERS")
            if len(shutters[0]) > 0 and len(shutters[1]) > 0:
                logger.info(f"Shutter open timestamp: {shutters[0][0]:.4f}")
                logger.info(f"Shutter close timestamp: {shutters[1][0]:.4f}")
                diff0 = shutters[1][0] - shutters[0][0]
                logger.info(
                    f"Total time between shutter opening and closing: {diff0:.4f} s."
                )
            elif len(shutters[0]) == 0 or len(shutters[1]) == 0:
                logger.warning("Missing shutter information!")
                logger.warning(
                    f"Number of shutter open timestamps found: {len(shutters[0])}"
                )
                logger.warning(
                    f"Number of shutter close timestamps found: {len(shutters[1])}"
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
                if len(v["LVDS fe"]) > 0:
                    diff3 = v["LVDS fe"][0] - v["TTL re"][-1]
                    logger.info(
                        f"Time difference between LVDS fe and last TTL: {diff3:.4f} s."
                    )
            else:
                logger.warning("No TTL triggers found!")
            # If SSX, print out SYNC info.
            if args.expt == "ssx":
                logger.info("SYNC")
                if len(v["SYNC re"]) > 0 and len(v["SYNC fe"]) > 0:
                    logger.info(
                        f"Found {len(v['SYNC re'])} rising edges and {len(v['SYNC fe'])} falling edges."
                    )
                    diff4 = [b - a for a, b in zip(v["SYNC re"], v["SYNC fe"])]
                    avg4 = np.average(diff4) if len(diff4) else np.nan
                    logger.info(
                        f"Average time interval between SYNC re and fe: {avg4:.4f} s"
                    )
                    if len(v["TTL re"]) == len(v["SYNC re"]):
                        diff5 = [b - a for a, b in zip(v["TTL re"], v["SYNC re"])]
                        avg5 = np.average(diff5) if len(diff5) else np.nan
                        logger.info(
                            f"Average time interval between TTL re and SYNC re: {avg5:.4f} s"
                        )
                    else:
                        logger.error(
                            "The number of TTL re and SYNC re doesn't match. Impossible to accurately calculate time difference."
                        )
                elif len(v["SYNC re"]) == 0:
                    logger.warning("No SYNC rising edges found!")
                elif len(v["SYNC fe"]) == 0:
                    logger.warning("No SYNC rising edges found!")
        logger.info("\n")


def cli():
    tic = time.process_time()
    args = parser.parse_args()
    main(args)
    toc = time.process_time()
    logger.debug(f"Total time taken: {toc - tic:4f} s.")
    logger.info("~~~ EOF ~~~")
