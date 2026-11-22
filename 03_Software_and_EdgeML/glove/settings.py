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
    "D": [0.6, 0.0, 0.9, 0.9, 0.9],    # index up, the other fingers curled to touch the thumb
    "I": [0.8, 1.0, 1.0, 1.0, 0.0],    # pinky up, thumb folded across the curled fingers
    "V": [0.8, 0.0, 0.0, 1.0, 1.0],    # index and middle up, thumb holding ring and pinky down
    # J and Z are traced in the air, so these are the handshape held while tracing. They are only
    # recognized while the gyroscope says the hand is moving (MOTION_START); a still I stays I.
    "J": [0.8, 1.0, 1.0, 1.0, 0.0],    # I's handshape (pinky up), traced in a J
    "Z": [0.7, 0.0, 1.0, 1.0, 1.0],    # index pointing, the others curled, traced in a Z
}
ACCEPT_CONFIDENCE = 0.80       # free practice: how sure recognize() must be to name a sign
MARGIN = 0.10                  # free practice: the best sign must beat the second-best by this much
ACCEPT_SCORE = 90              # lesson: match score (0-100) needed to accept the target (TC-DELIV-03 1.1)
ADJUST_TOLERANCE = 0.20        # a finger more than this far from the reference is "out of place"
HOLD_SECONDS = 0.5             # a correct sign must be held this long to count (TC-DELIV-03 1.1)
FAULT_RECOVERY_SECONDS = 1.0   # after a sensor fault, readings must be valid this long before scoring resumes
SUCCESS_SECONDS = 1.5          # how long CORRECT stays on screen

# Neural network (asl_model.py) ----------------------------------------------
# Train it with collect_data.py then train_model.py (see TRAINING.md). With the gyroscope it can
# learn all 26 letters, J and Z included. Without it, the 24 still letters (A to Y except J).
USE_MODEL = False              # True: free practice reads letters with the trained network instead of REFERENCES.
                               # Turn on only after train_model.py shows it beats REFERENCES. If the model file
                               # is missing, the glove falls back to REFERENCES and says so in the terminal.
MODEL_FILE = Path(__file__).with_name("models") / "asl_model.npz"
MODEL_MAX_DISTANCE = 0.15      # show "--" if the hand is further than this (average finger error) from every
                               # recorded example of the letter the network picked
MODEL_SMOOTHING_FRAMES = 5     # average the network over this many readings (5 = 0.25 s) so letters don't flicker
ACTIVE_LETTERS = None          # None: every letter. ["A", "B", "C"] for the presentation: any other letter shows "--"

# One letter at a time --------------------------------------------------------
SHOW_LOOK_ALIKE_GROUPS = False # False: free practice always shows ONE letter, the most likely one.
                               # True: letters the model can't tell apart show together, e.g. "U/V".

# Gyroscope / IMU (MPU-6050 or MPU-6500 on I2C: SDA GPIO2 pin 3, SCL GPIO3 pin 5) -----------
USE_IMU = True                 # False: flex sensors only. If the IMU isn't found the glove carries on
                               # without it and says so in the terminal (J and Z need it).
IMU_I2C_BUS = 1                # /dev/i2c-1, the Pi's GPIO2/GPIO3 bus
IMU_ADDRESS = 0x68             # 0x68 with AD0 to GND (most boards), 0x69 with AD0 to 3.3 V
IMU_SAMPLE_HZ = 100            # readings per second, in their own background thread
IMU_AXES = ("+x", "+y", "+z")  # which IMU axis points (1) along the fingers, (2) toward the thumb,
                               # (3) out of the back of the hand. Check with  python3 imu.py
TILT_SECONDS = 0.1             # tilt = the average of this much accelerometer data
MOTION_SECONDS = 0.5           # motion = the average turning speed over this long
MOTION_FULL_SCALE_DPS = 250    # turning this fast (degrees per second) counts as motion 1.0
MOTION_START = 0.12            # motion above this (about 30 deg/s) = the hand is moving: J and Z can be
                               # read, still letters can't. Below it, only still letters can be read.
MOTION_HOLD_SECONDS = 0.3      # J or Z must be read for this long while tracing before it is shown

# Touchscreen (the Winstar's 4-wire resistive panel) -------------------------
USE_TOUCHSCREEN = True         # True: main.py starts on the home screen ([Free] [Lesson] [Recalibrate]).
                               # False: straight into free practice, as before.
TOUCH_XP_PIN, TOUCH_XM_PIN = 4, 14     # X+ (physical 7) and X- (physical 8). GPIO14/15 are the serial
TOUCH_YP_PIN, TOUCH_YM_PIN = 15, 18    # Y+ (physical 10) and Y- (physical 12). port: turn it off
TOUCH_XP_CHANNEL, TOUCH_YP_CHANNEL = 5, 6   # X+ also goes to MCP3208 CH5, Y+ to CH6
TOUCH_X_MIN, TOUCH_X_MAX = 300, 2400   # raw touch readings at the left/right edges: run
TOUCH_Y_MIN, TOUCH_Y_MAX = 300, 2400   # python3 main.py --touch-test and copy what it prints
TOUCH_SWAP_XY = False          # True if moving your finger sideways changes the Y value

# Training data --------------------------------------------------------------
# Everyone who records letters with collect_data.py --person <name>. Use these exact names:
# a typo ("zoin") would count as a different person and split that person's recordings.
TEAM = ["zion", "amad", "milly", "sleytters", "chris"]

# Lesson ---------------------------------------------------------------------
LESSON = ["A", "B", "C"]       # the order the lesson goes through the signs