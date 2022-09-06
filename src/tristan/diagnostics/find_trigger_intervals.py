"""
Calculate the interval between rising and falling edge of trigger signal.
"""
import argparse
import glob
import logging
import sys
import time
from pathlib import Path

import h5py
import numpy as np

# Define a logger
logger = logging.getLogger("TriggerIntervalTime")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(message)s")  # %(levelname)s
# Deifne stream handler
CH = logging.StreamHandler(sys.stdout)
CH.setLevel(logging.DEBUG)
CH.setFormatter(formatter)
logger.addHandler(CH)

# Define parser
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("visitpath", type=str, help="Visit directory")
parser.add_argument("filename", type=str, help="Filename")

# Some constants
timing_resolution_fine = 1.5625e-9
shutter_open_header = 0x0840
shutter_close_header = 0x0880
trigger_ttl_re_header = 0x08E9  # just recording RE
trigger_ttl_fe_header = 0x08C9
trigger_lvds_re_header = 0x08EA
trigger_lvds_fe_header = 0x08CA
# trigger_sync_re_header = 0x08EC
# trigger_sync_fe_header = 0x08CC

# Detector dimensions
x_mod = 515
y_mod = 2069
x_gap = 117
y_gap = 45
n_mod = (2, 5)  # (H, V)
y_pix = 3043
x_pix = 4183


def define_module_dim():
    mod = {}
    for _y in range(n_mod[0]):
        for _x in range(n_mod[1]):
            int_x = [_x * (x_mod + x_gap), _x * (x_mod + x_gap) + x_mod]
            int_y = [_y * (y_mod + y_gap), _y * (y_mod + y_gap) + y_mod]
        mod[(_y, _x)] = (int_y, int_x)
    return mod


def main(args):
    logger.info("Look for triggers in cue messages.")
    filepath = Path(args.visitpath).expanduser().resolve()
    base = args.filename + f"_{6*'[0-9]'}.h5"
    logger.info(f"Collection directory: {filepath}")
    logger.info(f"Filename root: {args.filename}")
    filename_template = filepath / base
    file_list = [
        Path(f).expanduser().resolve()
        for f in sorted(glob.glob(filename_template.as_posix()))
    ]
    logger.info(f"Found {len(file_list)} files in directory.\n")
    # modules = define_module_dim()
    # L = split_modules(file_list, modules)

    # TODO, for test it works, but in a real collection which
    # files are in which module is anyone's guess.
    L = [file_list[i : i + 10] for i in range(0, len(file_list), 10)]

    SH_sorted = {}
    LVDS_sorted = {}
    TTL_sorted = {}
    # SYNC_sorted = {}
    for n, l in enumerate(L):
        sh_open = []
        sh_close = []
        ttl_re = []
        lvds_re = []
        lvds_fe = []
        # sync_re = []
        # sync_fe = []
        for f in l:
            with h5py.File(f, "r") as fh:
                cues = fh["cue_id"][()]
                cues_time = fh["cue_timestamp_zero"][()]
                # Look for shutter signal
                op_idx = np.where(cues == shutter_open_header)[0]
                cl_idx = np.where(cues == shutter_close_header)[0]
                if len(op_idx) > 0:
                    for i in range(len(op_idx)):
                        sh_open.append(cues_time[op_idx[i]] * timing_resolution_fine)
                if len(cl_idx) > 0:
                    for i in range(len(cl_idx)):
                        sh_close.append(cues_time[cl_idx[i]] * timing_resolution_fine)
                # Look for TTL
                ttl_idx = np.where(cues == trigger_ttl_re_header)[0]
                if len(ttl_idx) > 0:
                    for i in range(len(ttl_idx)):
                        ttl_re.append(cues_time[ttl_idx[i]] * timing_resolution_fine)
                # Look for LVDS
                lvds_up_idx = np.where(cues == trigger_lvds_re_header)[0]
                lvds_down_idx = np.where(cues == trigger_lvds_fe_header)[0]
                if len(lvds_up_idx) > 0:
                    for i in range(len(lvds_up_idx)):
                        lvds_re.append(
                            cues_time[lvds_up_idx[i]] * timing_resolution_fine
                        )
                if len(lvds_down_idx) > 0:
                    for i in range(len(lvds_down_idx)):
                        lvds_fe.append(
                            cues_time[lvds_down_idx[i]] * timing_resolution_fine
                        )
                # Look for SYNC
                # sync_up_idx = np.where(cues == trigger_sync_re_header)[0]
                # sync_down_idx = np.where(cues == trigger_sync_fe_header)[0]
                # if len(sync_up_idx) > 0:
                #     for i in range(len(sync_up_idx)):
                #         sync_re.append(cues_time[sync_up_idx[i]]*timing_resolution_fine)
                # if len(sync_down_idx) > 0:
                #     for i in range(len(sync_down_idx)):
                #         sync_fe.append(cues_time[sync_down_idx[i]]*timing_resolution_fine)
        SH_sorted[str(n)] = [sorted(sh_open), sorted(sh_close)]
        LVDS_sorted[str(n)] = [sorted(lvds_re), sorted(lvds_fe)]
        TTL_sorted[str(n)] = [sorted(ttl_re)]
        # SYNC_sorted[str(n)] = [sorted(sync_re), sorted(sync_fe)]

    for k in LVDS_sorted.keys():
        logger.info(f"--- Module {k} ---")
        logger.info("SHUTTERS")
        diff0 = [b - a for a, b in zip(SH_sorted[k][0], SH_sorted[k][1])]
        if diff0:
            logger.info(f"Total time shutter opening and closing: {diff0[0]:.4f} s.")
        logger.info("LVDS")
        logger.info(
            f"Found {len(LVDS_sorted[k][0])} rising edges and {len(LVDS_sorted[k][1])} falling edges."
        )
        diff1 = [b - a for a, b in zip(LVDS_sorted[k][0], LVDS_sorted[k][1])]
        if diff1:
            logger.info(f"Time difference between re and fe signal: {diff1[0]:.4f} s.")
        logger.info("TTL")
        logger.info(f"Found {len(TTL_sorted[k][0])} rising edges.\n")
        diff2 = TTL_sorted[k][0][0] - LVDS_sorted[k][0][0]
        logger.info(f"Time difference between first TTL and LVDS re: {diff2:.4f} s.")
        diff3 = LVDS_sorted[k][1][0] - TTL_sorted[k][0][-1]
        logger.info(f"Time difference between LVDS fe and last TTL: {diff3:.4f} s.")
        # print(LVDS_sorted[k][0][0], LVDS_sorted[k][1][0])

        # logger.info("SYNC")
        # logger.info(f"Found {len(SYNC_sorted[k][0])} rising edges and {len(SYNC_sorted[k][0])} falling edges.")

        # if len(TTL_sorted[k][0]) == len(SYNC_sorted[k][0]):
        #     if len(TTL_sorted[k][0]) > 0:
        #         diff2 = [b-a for a,b in zip(TTL_sorted[k][0], SYNC_sorted[k][0])]
        #         avg2 = np.average(diff2) if len(diff2) else np.nan
        #         logger.info(f"Average time interval between TTL re and SYNC re: {avg2:.4} s")
        # else:
        #     logger.error("The number of TTL re and SYNC re doesn't match. Impossible to accurately calculate time difference.")

        # if len(SYNC_sorted[k][0]) == len(SYNC_sorted[k][1]):
        #     if len(SYNC_sorted[k][0]) > 0:
        #         diff3 = [b-a for a,b in zip(SYNC_sorted[k][0], SYNC_sorted[k][1])]
        #         avg3 = np.average(diff3) if len(diff3) else np.nan
        #         logger.info(f"Average time interval between SYNC re and fe: {avg3:.4} s")
        # else:
        #     logger.error("The number of SYNC re and fe doesn't match.")
        logger.info("\n")

    # print(ttl_re[0], ttl_re[-1])
    # print(min(ttl_re), max(ttl_re))
    logger.info("Printing out LVDS timestamps")
    logger.info(f"LVDS rising edge: {lvds_re[0]:.4f}")
    logger.info(f"LVDS falling edge: {lvds_fe[0]:.4f}")


def cli():
    tic = time.process_time()
    args = parser.parse_args()
    main(args)
    toc = time.process_time()
    logger.debug(f"Total time taken: {toc - tic:4f} s.")
