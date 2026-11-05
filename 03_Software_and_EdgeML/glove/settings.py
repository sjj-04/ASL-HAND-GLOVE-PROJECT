"""
Every number you might change while building and tuning the glove.
Section 3.4 of the Design Specification.

Calibration and the hardware classes live in Calibration_Routine/glove_calibration.py,
so the settings they use are set there and only copied here.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "Calibration_Routine"))   # lets glove/ use that folder
import glove_calibration as routine  # type: ignore  (VS Code can't follow the line above)

# Shared with Calibration_Routine: change these in glove_calibration.py ------
FINGERS = routine.FINGERS
CALIBRATION_BUTTON_PIN, BUTTON_PULL_UP = routine.CALIBRATION_BUTTON_PIN, routine.BUTTON_PULL_UP
SAMPLE_HZ, WINDOW, LOOP_HZ = routine.SAMPLE_HZ, routine.WINDOW, routine.LOOP_HZ
MIN_RANGE, MAX_CAL_TRIES, MESSAGE_SECONDS = routine.MIN_RANGE, routine.MAX_CAL_TRIES, routine.MESSAGE_SECONDS
CAL_FILE, MOCK_CAL_FILE = routine.CAL_FILE, routine.MOCK_CAL_FILE     # saved in Calibration_Routine/

# Wrist LCD wiring (Winstar WDG0151: two 64x64 halves), BCM GPIO numbers.
# glove_calibration.py lists the physical pin next to each one.
LCD_CS1_PIN, LCD_CS2_PIN = routine.LCD_CS1_PIN, routine.LCD_CS2_PIN      # /CS1 left half, /CS2 right half
LCD_CS_ACTIVE_LOW = routine.LCD_CS_ACTIVE_LOW                           # True: a low pin selects that half
LCD_DI_PIN, LCD_E_PIN, LCD_RST_PIN = routine.LCD_DI_PIN, routine.LCD_E_PIN, routine.LCD_RST_PIN
LCD_DATA_PINS = routine.LCD_DATA_PINS                                   # DB0-DB7, in that order

# Wiring ---------------------------------------------------------------------
SELECTION_BUTTON_PIN = 26      # BCM GPIO number (physical pin 37)
BUTTON_BOUNCE_SECONDS = 0.05   # ignore the extra contacts a button makes in the 50 ms after a press

# Wrist LCD ------------------------------------------------------------------
LCD_REFRESH_HZ = 10            # how often the live CH0-CH4 screen redraws (Test 2.6 needs at least 10)
STARTUP_IMAGE_SECONDS = 1.5    # how long the start-up image stays up (Tests 2.4 and 2.8 look for it)

# Sampling -------------------------------------------------------------------
MEDIAN_SAMPLES = 5             # readings in the spike filter (25 ms)

# Start-up check -------------------------------------------------------------
ADC_MAX = 4095                 # the largest 12-bit reading (5 V)
FAULT_MARGIN = 50              # this close to 0 or 4095 means an unplugged or shorted sensor

# Sign feedback --------------------------------------------------------------
# How bent each finger should be for each sign, Thumb to Pinky (0.0 straight, 1.0 bent).
# PLACEHOLDERS: replace them with readings from an experienced signer wearing the glove.
REFERENCES = {
    "A": [0.1, 1.0, 1.0, 1.0, 1.0],    # fist with the thumb alongside
    "B": [0.9, 0.0, 0.0, 0.0, 0.0],    # flat fingers with the thumb folded in
    "C": [0.4, 0.5, 0.5, 0.5, 0.5],    # every finger curved halfway
}
ACCEPT_CONFIDENCE = 0.85       # how sure recognize() must be to accept a sign
ADJUST_TOLERANCE = 0.25        # a finger this far from the reference is "out of place"
HOLD_SECONDS = 1.0             # a correct sign must be held this long to count
SUCCESS_SECONDS = 1.5          # how long CORRECT stays on screen

# Lesson ---------------------------------------------------------------------
LESSON = ["A", "B", "C"]       # the order the lesson goes through the signs