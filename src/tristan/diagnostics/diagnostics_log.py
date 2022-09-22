"""
Logging configuration for diagnostics.
"""

import logging
import logging.config

logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "class": "logging.Formatter",
            "format": "%(message)s",
        }
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        }
    },
    "loggers": {
        "TristanDiagnostics": {
            "handlers": ["console"],
            "level": "DEBUG",
            "propagate": True,
        }
    },
}

logging.config.dictConfig(logging_config)


def config(logfile: str = None, write_mode: str = "a"):
    diag_logger = logging.getLogger("TristanDiagnostics")
    if logfile:
        fileFormatter = logging.Formatter(
            "%(asctime)s - %(levelname)s -- %(message)s",
            datefmt="%d-%m-%Y %I:%M:%S",
        )
        FH = logging.FileHandler(logfile, mode=write_mode, encoding="utf-8")
        FH.setLevel(logging.INFO)
        FH.setFormatter(fileFormatter)
        diag_logger.addHandler(FH)
