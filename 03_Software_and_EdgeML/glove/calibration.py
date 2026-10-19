"""
Calibration for the full program.

The calibration routine itself lives in Calibration_Routine/glove_calibration.py,
so it works the same whether you run that file or main.py. This file only adds
one extra check before trusting a saved calibration.
"""

import numpy as np

import settings
from settings import routine

run_calibration = routine.run_calibration       # the run_calibration() flowchart, unchanged
save_calibration = routine.save_calibration


def load_calibration(cal_file=settings.CAL_FILE):
    """
    Return the saved (flat, fist) readings, or None if there's no usable file.

    Uses Calibration_Routine's loader, then also rejects a finger whose range is
    too small, because normalize() would divide by (almost) zero.
    """
    calibration = routine.load_calibration(cal_file)
    if calibration is None:
        return None
    flat, fist = calibration
    if np.any(np.abs(fist - flat) < settings.MIN_RANGE):
        return None                                  # wouldn't pass calibration, so don't trust it
    return calibration
