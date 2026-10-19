"""
A pretend glove, so the whole program and its tests run on any computer.
These are the mock classes planned in Appendix C (7.3.3) of the Design Specification.
"""

import random
import threading
import time

import numpy as np

from settings import FINGERS, MEDIAN_SAMPLES, REFERENCES, WINDOW

# Raw readings a typical hand gives, Thumb to Pinky. Straight fingers read HIGHER:
# bending raises a flex sensor's resistance, which lowers its divider's voltage.
FLAT_READINGS = [2250, 2300, 2280, 2260, 2240]
FIST_READINGS = [1750, 1500, 1450, 1480, 1550]
NOISE = 10                                       # each reading wobbles by up to +/- this, like a real sensor

FLAT, FIST = [0.0] * len(FINGERS), [1.0] * len(FINGERS)   # poses: how bent each finger is

# What the pretend hand does once a lesson starts: (seconds in, pose).
PRACTICE_SCRIPT = [
    (0.0,  FLAT),                        # target A: TRY AGAIN
    (2.0,  [0.1, 0.0, 1.0, 1.0, 1.0]),   # "A" with the index finger open: ADJUST: Index
    (4.0,  REFERENCES["A"]),             # HOLD STEADY, then CORRECT: A
    (7.0,  FLAT),                        # target B, thumb sticking out: ADJUST: Thumb
    (9.0,  REFERENCES["B"]),             # HOLD STEADY, then CORRECT: B
    (12.0, REFERENCES["C"]),             # CORRECT: C, then free practice shows SIGN: C
    (15.5, [0.6, 0.2, 0.9, 0.3, 0.7]),   # no clear sign: SIGN: --
]


def open_mock_glove(stuck_finger=None, unplugged_finger=None):
    """The pretend glove. Returns the same four parts as hardware.open_hardware()."""
    hand = PretendHand(stuck_finger, unplugged_finger)
    calibration_button, selection_button = MockButton(), MockButton()
    watch_keyboard(calibration_button, selection_button)
    print("Pretend glove: type c + Enter for the Calibration button, s + Enter for Selection. Ctrl+C quits.")
    return MockMCP3208(hand), MockOLED(hand), calibration_button, selection_button


class PretendHand:
    """A fake hand that reacts to the wrist display, like a real user would.

    During calibration it follows the FLAT and FIST prompts. Once a lesson
    starts, it acts out PRACTICE_SCRIPT so every kind of feedback appears.
    """

    def __init__(self, stuck_finger=None, unplugged_finger=None):
        self.stuck = FINGERS.index(stuck_finger) if stuck_finger else None              # never bends
        self.unplugged = FINGERS.index(unplugged_finger) if unplugged_finger else None  # reads near 0
        self.calibrating = False
        self.calibration_pose = FLAT
        self.script_start = None

    def follow(self, title, detail):
        """React to a new message on the wrist display."""
        self.calibrating = title == "CALIBRATING"
        if self.calibrating:
            self.calibration_pose = FIST if "FIST" in detail else FLAT
        if title.startswith("Target") and self.script_start is None:
            self.script_start = time.monotonic()        # the lesson has begun: start the script

    def pose(self):
        """How bent each finger is right now, Thumb to Pinky (0.0 straight, 1.0 bent)."""
        if self.calibrating:
            return self.calibration_pose
        if self.script_start is None:
            return FLAT
        elapsed = time.monotonic() - self.script_start
        return [pose for start, pose in PRACTICE_SCRIPT if elapsed >= start][-1]   # the latest pose that has begun

    def reading(self, channel):
        """The raw value (0-4095) this hand produces on one ADC channel right now."""
        if channel == self.unplugged:
            return random.randint(0, NOISE)
        bend = 0.0 if channel == self.stuck else self.pose()[channel]
        value = FLAT_READINGS[channel] + bend * (FIST_READINGS[channel] - FLAT_READINGS[channel])
        return round(value) + random.randint(-NOISE, NOISE)


class MockMCP3208:
    """Stands in for the ADC: returns the pretend hand's readings."""

    def __init__(self, hand):
        self.hand = hand

    def read(self, channel):
        return self.hand.reading(channel)

    def close(self):
        pass


class MockOLED:
    """Stands in for the wrist OLED: prints each message and shows it to the pretend hand."""

    def __init__(self, hand):
        self.hand = hand
        self.history = []                               # every message shown, so tests can check them
        self.shown = None

    def message(self, title, detail=""):
        if (title, detail) == self.shown:
            return                                      # already on screen
        self.shown = (title, detail)
        self.history.append(self.shown)
        print(f"[OLED] {title:<12} {detail}".rstrip())
        self.hand.follow(title, detail)


class MockButton:
    """Stands in for a gpiozero Button. main.py sets when_pressed the same way."""

    def __init__(self):
        self.when_pressed = None

    def press(self):
        if self.when_pressed:
            self.when_pressed()


def watch_keyboard(calibration_button, selection_button):
    """In the background, turn typed keys into button presses: c + Enter or s + Enter."""
    def watch():
        while True:
            try:
                key = input().strip().lower()
            except EOFError:                            # no keyboard to read from
                return
            if key == "c":
                calibration_button.press()
            elif key == "s":
                selection_button.press()

    threading.Thread(target=watch, daemon=True).start()


class InstantSampler:
    """Like hardware.Sampler, but reads the pretend hand on demand, so tests don't wait."""

    def __init__(self, adc):
        self.adc = adc

    def average(self):
        return np.mean([self._read_all() for _ in range(WINDOW)], axis=0)

    def newest(self):
        return np.median([self._read_all() for _ in range(MEDIAN_SAMPLES)], axis=0)

    def _read_all(self):
        return [self.adc.read(channel) for channel in range(len(FINGERS))]
