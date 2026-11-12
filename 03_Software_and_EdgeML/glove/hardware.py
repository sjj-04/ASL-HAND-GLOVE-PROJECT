"""
The real glove hardware.

The ADC, wrist display and Sampler come from Calibration_Routine/glove_calibration.py.
This file adds what the full program needs on top: the Selection button, a spike
filter, and a display that skips repeated messages.
"""

import time

import numpy as np

import settings
from settings import routine

PI_LIBRARIES_FOUND = routine.PI_LIBRARIES_FOUND
MCP3208 = routine.MCP3208                            # the ADC, unchanged



def open_hardware():
    """Connect to the real glove. Returns the ADC and the wrist display."""
    return MCP3208(), WristDisplay()


class WristDisplay(routine.WristDisplay):
    """
    Calibration_Routine's wrist LCD, but it skips messages that are already on screen,
    so the practice loop doesn't redraw it 20 times per second.
    """
    def __init__(self):
        super().__init__()
        self.shown = None

    def message(self, title, detail=""):
        if (title, detail) != self.shown:
            self.shown = (title, detail)
            super().message(title, detail)


class Sampler(routine.Sampler):
    """Calibration_Routine's 200 Hz sampler, plus newest() for the startup check and practice loop."""

    def start(self):
        super().start()
        time.sleep(0.1)                              # let the first readings arrive

    def newest(self):
        """Each finger's newest reading: the median of the last 5 readings (25 ms).

        A median ignores one wild reading, such as a spike from a cracked
        sensor, where an average would be pulled toward it.
        """
        with self.lock:
            recent = np.array(list(self.window)[-settings.MEDIAN_SAMPLES:], dtype=float)
        if len(recent) == 0:
            raise RuntimeError("No sensor readings. Check the ADC wiring and that SPI is turned on.")
        return np.median(recent, axis=0)
