#!/usr/bin/env python3
"""
glove_core.py - ASL Learning Tool, core glove program
Design Specifications - Appendix B

This program follows the flowcharts in Section 6 (Figures 6-1 and 6-2):
  1. Startup     - open the ADC and the wrist display, check all five sensors
  2. Calibrate   - record a flat hand and a closed fist, save them to a file
  3. Sample      - read the five flex sensors 200 times per second
  4. Normalize   - scale each finger from 0.0 (straight) to 1.0 (fully bent)
  5. Recognize   - work out which supported sign the hand is forming
  6. Evaluate    - compare the recognized sign with the lesson's target
  7. Feedback    - show CORRECT, ADJUST: [finger] or TRY AGAIN on the wrist OLED
  8. Success     - after an accepted match, tell the lesson so it can advance

With no lesson target, the glove only shows the sign it recognizes
(independent use) and never labels an attempt "correct".

Step 5 is a simple stand-in that picks the closest reference handshape.
The trained neural network will replace the body of recognize() later.

Run on the glove:        python3 glove_core.py
Run without any hardware: python3 glove_core.py --mock --seconds 14
"""

import argparse
import json
import random
import threading
import time
from collections import deque

import numpy as np

# ---------------------------------------------------------------------------
# Settings. Section 3.4 of the Design Specifications.
# ---------------------------------------------------------------------------
FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]   # ADC channels CH0-CH4
SAMPLE_HZ = 200          # sensor readings per second (one set every 5 ms)
DISPLAY_HZ = 20          # evaluations and display updates per second
WINDOW = 100             # samples kept in memory = the most recent 0.5 second
MEDIAN_SAMPLES = 5       # readings used by the median filter (25 ms)

SPI_BUS, SPI_DEVICE = 0, 0   # SPI0 with chip select CE0 (GPIO8, pin 24)
SPI_HZ = 250_000             # SPI clock: 250 kHz
OLED_ADDRESS = 0x3C          # I2C address of the wrist OLED
CAL_PIN, SELECT_PIN = 17, 27 # Calibration and Selection pushbuttons (GPIO numbers)

ADC_MAX = 4095           # largest 12-bit reading (5 V)
FAULT_MARGIN = 50        # a reading this close to 0 or 4095 means a wiring fault
MIN_RANGE = 200          # a finger must change by this many counts between poses
MAX_CAL_TRIES = 3        # calibration attempts before a fault is reported
CAL_FILE = "calibration.json"

# Reference handshapes: how bent each finger should be (thumb to little),
# from 0.0 (straight) to 1.0 (fully bent).
# PLACEHOLDER VALUES. Replace with recordings verified by an experienced signer.
REFERENCES = {
    "A": [0.1, 1.0, 1.0, 1.0, 1.0],    # fist with the thumb alongside
    "B": [0.9, 0.0, 0.0, 0.0, 0.0],    # flat fingers with the thumb folded in
    "C": [0.4, 0.5, 0.5, 0.5, 0.5],    # every finger curved halfway
}
LESSON = ["A", "B", "C"]     # the order the lesson presents its targets

ACCEPT_CONFIDENCE = 0.85     # how sure the recognizer must be to accept a sign
ADJUST_TOLERANCE = 0.25      # a finger this far from its reference gets named
HOLD_SECONDS = 1.0           # a correct sign must be held this long to count
SUCCESS_PAUSE = 1.5          # time "CORRECT" stays on the display


# ---------------------------------------------------------------------------
# Hardware: the MCP3208 analog-to-digital converter
# ---------------------------------------------------------------------------
class MCP3208:
    """Reads the 12-bit ADC over SPI."""

    def __init__(self):
        import spidev                      # only available on the Raspberry Pi
        self.spi = spidev.SpiDev()
        self.spi.open(SPI_BUS, SPI_DEVICE)
        self.spi.max_speed_hz = SPI_HZ
        self.spi.mode = 0                  # SPI mode 0, as the MCP3208 expects

    def read(self, channel):
        """Return one reading (0-4095) from one channel (0-7)."""
        # Build the 3-byte request that tells the chip which input to read.
        command = [0x06 | (channel >> 2), (channel & 0x03) << 6, 0x00]
        # Send the request. The chip sends 3 bytes back at the same time.
        reply = self.spi.xfer2(command)
        # The reading is 12 bits: the last 4 bits of the second byte
        # plus all 8 bits of the third byte. Join them into one number.
        return ((reply[1] & 0x0F) << 8) | reply[2]


class MockMCP3208:
    """Stands in for the ADC so the program can be tested with no glove.

    It pretends a hand is flat for 3 seconds and makes a fist for 3 seconds
    (so the calibration routine has both poses to record). It then works
    through a short script of handshapes that exercises every kind of feedback.
    """

    FLAT, FIST = 2250, 1500        # typical counts for a straight and a bent finger

    # (time the pose starts in seconds, how bent each finger is)
    SCRIPT = [
        (0.0,  [0.0, 0.0, 0.0, 0.0, 0.0]),     # flat hand (calibration)
        (3.0,  [1.0, 1.0, 1.0, 1.0, 1.0]),     # closed fist (calibration)
        (6.0,  [0.1, 0.2, 1.0, 1.0, 1.0]),     # "A" with the index finger left open
        (8.0,  REFERENCES["A"]),               # a correct "A"
        (11.0, [0.7, 0.2, 0.2, 0.2, 0.2]),     # a loose "B"
        (13.0, REFERENCES["B"]),               # a correct "B"
        (16.0, REFERENCES["C"]),               # a correct "C"
    ]

    def __init__(self):
        self.start = time.monotonic()

    def read(self, channel):
        t = time.monotonic() - self.start
        bends = self.SCRIPT[0][1]
        for start_time, pose in self.SCRIPT:   # find the pose for this moment
            if t >= start_time:
                bends = pose
        counts = self.FLAT + bends[channel] * (self.FIST - self.FLAT)
        return int(counts + random.gauss(0, 3))    # add a little sensor noise


# ---------------------------------------------------------------------------
# Hardware: the wrist OLED display
# ---------------------------------------------------------------------------
class WristOLED:
    """Draws two lines of text on the 128 x 64 I2C OLED."""

    def __init__(self):
        from luma.core.interface.serial import i2c
        from luma.core.render import canvas
        from luma.oled.device import ssd1306
        self.canvas = canvas
        self.device = ssd1306(i2c(port=1, address=OLED_ADDRESS))
        self.shown = None

    def message(self, line1, line2=""):
        """Show up to two lines of text."""
        if (line1, line2) == self.shown:
            return                         # already on the display: skip the redraw
        self.shown = (line1, line2)
        with self.canvas(self.device) as draw:
            draw.text((0, 16), line1, fill="white")
            draw.text((0, 36), line2, fill="white")


class MockOLED:
    """Prints to the terminal what the OLED would show."""

    def __init__(self):
        self.shown = None

    def message(self, line1, line2=""):
        if (line1, line2) == self.shown:
            return                         # print only when the message changes
        self.shown = (line1, line2)
        print(f"[OLED] {line1} | {line2}", flush=True)


# ---------------------------------------------------------------------------
# Step 3: Sample. Runs in the background so the display never delays a reading.
# ---------------------------------------------------------------------------
class Sampler(threading.Thread):
    """Reads all five fingers 200 times per second and keeps the last 0.5 s."""

    def __init__(self, adc):
        super().__init__(daemon=True)      # stops automatically when the program ends
        self.adc = adc
        self.window = deque(maxlen=WINDOW) # the oldest sample drops off automatically
        self.lock = threading.Lock()       # keeps the two threads from colliding
        self.running = True

    def run(self):
        period = 1.0 / SAMPLE_HZ           # 5 ms
        next_time = time.monotonic()       # next reading
        while self.running:
            #read every finger sensor once
            sample = [self.adc.read(ch) for ch in range(len(FINGERS))]
            # Save the readings. The lock stops other code from
            # using the list while we add to it.
            with self.lock:
                self.window.append(sample)
            # Wait until the next reading is due. Wait for a set
            # clock time instead of a flat 5 ms so the loop doesn't
            # slowly fall behind.
            next_time += period
            time.sleep(max(0.0, next_time - time.monotonic()))

    def latest(self):
        """Return the most recent reading of each finger."""
        with self.lock:
            return np.array(self.window[-1], dtype=float)

    def filtered(self):
        """Return each finger's median over the last few readings.

        A median ignores a single wild reading, such as a spike from a
        cracked sensor, where an average would be pulled toward it.
        """
        with self.lock:
            recent = list(self.window)[-MEDIAN_SAMPLES:]
        return np.median(np.array(recent, dtype=float), axis=0)

    def average(self):
        """Return each finger's average over the last 0.5 s (used by calibration)."""
        with self.lock:
            return np.mean(np.array(self.window, dtype=float), axis=0)


# ---------------------------------------------------------------------------
# Step 1: Startup check
# ---------------------------------------------------------------------------
def find_faulty_finger(readings):
    """Return the name of the first finger reading out of range, or None.

    A disconnected sensor reads near 0 and a shorted one reads near 4095.
    """
    for name, value in zip(FINGERS, readings):
        if value < FAULT_MARGIN or value > ADC_MAX - FAULT_MARGIN:
            return name
    return None


# ---------------------------------------------------------------------------
# Step 2: Calibration
# ---------------------------------------------------------------------------
def record_pose(sampler, display, prompt):
    """Show a prompt, wait for the user to hold the pose, and return the average reading."""
    display.message("CALIBRATING", prompt)
    time.sleep(2.0)                        # give the user time to get into the pose
    time.sleep(WINDOW / SAMPLE_HZ)         # wait until all saved readings are from this pose
    return sampler.average()


def run_calibration(sampler, display):
    """Record the flat-hand and closed-fist readings and save them.

    Returns (flat, fist) on success, or None after three failed attempts.
    """
    for attempt in range(1, MAX_CAL_TRIES + 1):
        flat = record_pose(sampler, display, "Hold hand FLAT")
        fist = record_pose(sampler, display, "Make a FIST")

        # Check how much each finger's reading changed between the two poses.
        # If any finger barely changed, something is wrong: the sensor
        # moved out of place or the user didn't hold the pose.
        change = np.abs(fist - flat)
        if np.all(change >= MIN_RANGE):
            # Every finger changed enough, so save the readings to a file.
            with open(CAL_FILE, "w") as f:
                json.dump({"flat": flat.tolist(), "fist": fist.tolist()}, f)
            display.message("Calibration", "saved")
            time.sleep(1.0) # leave the message up long enough to read
            return flat, fist

        # Find the finger that changed the least and tell the user to check it.
        stuck = FINGERS[int(np.argmin(change))]
        display.message(f"Check {stuck}", f"Try {attempt} of {MAX_CAL_TRIES}")
        time.sleep(2.0)

    display.message("FAULT", "Check wiring")
    return None


def load_calibration():
    """Return the saved (flat, fist) readings, or None if none are saved."""
    try:
        with open(CAL_FILE) as f:
            data = json.load(f)
        return np.array(data["flat"]), np.array(data["fist"])
    except (FileNotFoundError, KeyError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Step 4: Normalize
# ---------------------------------------------------------------------------
def normalize(raw, flat, fist):
    """Convert each finger's reading to a number from 0.0 (straight) to 1.0 (fully bent).

    Uses the flat and fist readings saved during calibration.
    Example: flat = 2250, fist = 1500, reading = 1875 gives 0.5 (half bent).
    Readings outside the calibrated range are set to 0.0 or 1.0.
    """
    return np.clip((raw - flat) / (fist - flat), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Step 5: Recognize
# ---------------------------------------------------------------------------
def recognize(bends):
     """Return the best-matching sign and a confidence score (0.0 to 1.0).

    TEMPORARY: this stands in for the neural network. It compares the
    hand to a stored example of each sign and picks the closest one.
    Confidence is 1.0 for an exact match and drops as the fingers move
    away from that example. The trained network will replace the code
    below and return the same two values.
    """
best_sign, best_distance = None, None
for sign, reference in REFERENCES.items():
        # Distance = average difference between the hand and this sign's
        # example, across all fingers. Smaller means a closer match.
        distance = float(np.mean(np.abs(bends - np.array(reference))))
        # Keep this sign if it is the closest one so far.
        if best_distance is None or distance < best_distance:
            best_sign, best_distance = sign, distance
    # Turn the distance into a confidence: closer match = higher score.
return best_sign, 1.0 - best_distance


# ---------------------------------------------------------------------------
# Step 6: Evaluate against the lesson's target
# ---------------------------------------------------------------------------
def evaluate(bends, target, recognized, confidence):
    """Grade one attempt at the target sign.

    Returns "CORRECT", "ADJUST: [finger]", or "TRY AGAIN".
    """    
    # Correct: the recognizer picked the target sign and is confident enough.
    if recognized == target and confidence >= ACCEPT_CONFIDENCE:
        return "CORRECT"

    # Not correct. Find the finger that is furthest from where it should be.
    errors = np.abs(bends - np.array(REFERENCES[target]))
    worst = int(np.argmax(errors))
    # If that finger is far enough off, tell the user which one to fix.

    if errors[worst] >= ADJUST_TOLERANCE:
        return f"ADJUST: {FINGERS[worst]}"

    # No single finger is far off, so ask the user to try again.
    return "TRY AGAIN"


# ---------------------------------------------------------------------------
# Step 8: The lesson (stand-in for the teaching application on the HDMI monitor)
# ---------------------------------------------------------------------------
class LessonManager:
    """Holds the active lesson target and receives the glove's success events.

    The teaching application will draw the prompt and the demonstration on
    the teaching monitor. This stand-in keeps track of the target and prints
    what the teaching screen would show.
    """

    def __init__(self, signs):
        self.signs = signs
        self.index = None                  # None = no active lesson target

    @property
    def target(self):
        """The sign being practiced, or None when no lesson is active."""
        return None if self.index is None else self.signs[self.index]

    def select_next(self):
        """Selection button: start the lesson, or skip to the next target."""
        self._advance()

    def report_success(self, sign):
        """Success event from the glove: mark the sign complete and move on."""
        print(f"[LESSON] {sign} complete", flush=True)
        self._advance()

    def _advance(self):
        self.index = 0 if self.index is None else self.index + 1
        if self.index >= len(self.signs):
            self.index = None              # lesson finished: back to independent use
            print("[LESSON] Lesson complete", flush=True)
        else:
            print(f"[LESSON] Form the sign: {self.target}", flush=True)


# ---------------------------------------------------------------------------
# Main program (Figure 6-1)
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ASL Learning Tool core program")
    parser.add_argument("--mock", action="store_true",
                        help="run with simulated hardware (no glove needed)")
    parser.add_argument("--seconds", type=float, default=None,
                        help="stop after this many seconds of practice")
    args = parser.parse_args()

    # The buttons only set a flag. The main loop below does the actual work,
    # so a button press can never interrupt a sensor reading.
    cal_pressed = threading.Event()
    select_pressed = threading.Event()

    # --- Step 1: Startup ---------------------------------------------------
    if args.mock:
        adc, display = MockMCP3208(), MockOLED()
        select_pressed.set()               # no buttons: start the lesson right away
    else:
        from gpiozero import Button
        adc, display = MCP3208(), WristOLED()
        # Each button connects its pin to ground; the Pi's internal pull-up
        # holds the pin at 3.3 V until it is pressed. 50 ms debounce.
        cal_button = Button(CAL_PIN, bounce_time=0.05)
        select_button = Button(SELECT_PIN, bounce_time=0.05)
        cal_button.when_pressed = cal_pressed.set
        select_button.when_pressed = select_pressed.set

    sampler = Sampler(adc)
    sampler.start()
    time.sleep(0.1)                        # let the first few samples arrive

    faulty = find_faulty_finger(sampler.latest())
    if faulty:
        display.message("SENSOR FAULT", f"Check {faulty}")
        return

    # --- Step 2: Calibration (automatic if nothing is saved) ---------------
    calibration = load_calibration()
    if calibration is None:
        calibration = run_calibration(sampler, display)
        if calibration is None:
            return                         # three failed attempts: stop here
    flat, fist = calibration
    display.message("READY")

    # --- Steps 3 to 8: the practice loop, 20 times per second --------------
    lesson = LessonManager(LESSON)
    correct_since = None                   # when the current correct hold began
    started = time.monotonic()
    try:
        while args.seconds is None or time.monotonic() - started < args.seconds:
            # Calibration button: recalibrate at any time.
            if cal_pressed.is_set():
                cal_pressed.clear()
                result = run_calibration(sampler, display)
                if result is not None:     # on a fault, keep the old calibration
                    flat, fist = result

            # Selection button: start the lesson or skip to the next target.
            if select_pressed.is_set():
                select_pressed.clear()
                lesson.select_next()
                correct_since = None

            # Steps 3 to 5: newest readings -> 0.0-1.0 bend values -> sign.
            bends = normalize(sampler.filtered(), flat, fist)
            recognized, confidence = recognize(bends)
            target = lesson.target

            if target is None:
                # Independent use: show the recognized sign, never "correct".
                shown = recognized if confidence >= ACCEPT_CONFIDENCE else "--"
                display.message("Free practice", f"SIGN: {shown}")
            else:
                # Steps 6 and 7: compare with the target and give feedback.
                result = evaluate(bends, target, recognized, confidence)
                now = time.monotonic()
                if result != "CORRECT":
                    correct_since = None
                    display.message(f"Target: {target}", result)
                elif correct_since is None:
                    correct_since = now    # correct: start timing the hold
                    display.message(f"Target: {target}", "HOLD STEADY")
                elif now - correct_since >= HOLD_SECONDS:
                    # Step 8: held long enough. Accept it and tell the lesson.
                    display.message(f"CORRECT: {target}")
                    lesson.report_success(target)
                    correct_since = None
                    time.sleep(SUCCESS_PAUSE)

            time.sleep(1.0 / DISPLAY_HZ)   # 20 evaluations per second
    except KeyboardInterrupt:
        pass                               # Ctrl+C ends the program cleanly
    finally:
        sampler.running = False


if __name__ == "__main__":
    main()