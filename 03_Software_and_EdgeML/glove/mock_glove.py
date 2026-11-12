"""
A pretend glove, so the whole program and its tests run on any computer.
These are the mock classes planned in Appendix C (7.3.3) of the Design Specification.
"""

import random
import threading
import time

import numpy as np

from settings import (FINGERS, LCD_CS1_PIN, LCD_CS2_PIN, LCD_CS_ACTIVE_LOW, LCD_DATA_PINS, LCD_DI_PIN,
                      LCD_E_PIN, LCD_RST_PIN, MEDIAN_SAMPLES, REFERENCES, WINDOW)

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
    """The pretend glove. Returns the same two parts as hardware.open_hardware()."""
    hand = PretendHand(stuck_finger, unplugged_finger)
    return MockMCP3208(hand), MockOLED(hand)


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
    """Stands in for the wrist LCD: prints each message and shows it to the pretend hand."""

    def __init__(self, hand):
        self.hand = hand
        self.history = []                               # every message shown, so tests can check them
        self.shown = None

    def message(self, title, detail=""):
        if (title, detail) == self.shown:
            return                                      # already on screen
        self.shown = (title, detail)
        self.history.append(self.shown)
        print(f"[LCD] {title:<12} {detail}".rstrip())
        self.hand.follow(title, detail)


class FakeLcdHalf:
    """One half of the wrist LCD (one KS0108 controller): a 64x64 area of the screen.

    Its memory is 8 pages x 64 columns. Each byte is one column of one page,
    8 pixels tall, with the top pixel in bit 0.
    """

    def __init__(self):
        self.ram = np.zeros((8, 64), dtype=np.uint8)
        self.page = 0                                   # which page the next byte goes to (0-7)
        self.column = 0                                 # which column the next byte goes to (0-63)
        self.on = False                                 # the screen starts off, like the real LCD
        self.start_line = 0                             # which memory row shows at the top of the screen

    def command(self, value):
        """Carry out one command byte. Checked from the most specific top bits down."""
        if value & 0xC0 == 0xC0:                        # 11xx xxxx: display start line
            self.start_line = value & 0x3F
        elif value & 0xF8 == 0xB8:                      # 1011 1xxx: set page
            self.page = value & 0x07
        elif value & 0xC0 == 0x40:                      # 01xx xxxx: set column
            self.column = value & 0x3F
        elif value & 0xFE == 0x3E:                      # 0011 111x: display on (1) or off (0)
            self.on = bool(value & 0x01)
        # Anything else is ignored, like the real chip.

    def data(self, value):
        """Store 8 pixels, then move one column right. Past column 63 it wraps to 0 on the same page."""
        self.ram[self.page, self.column] = value
        self.column = (self.column + 1) & 0x3F

    def reset(self):
        """What RST does: the display turns off and starts at line 0. The memory is NOT cleared."""
        self.on = False
        self.start_line = 0

    def pixels(self):
        """This half as a 64x64 array of True/False, as it looks on the glass."""
        if not self.on:
            return np.zeros((64, 64), dtype=bool)
        rows = np.unpackbits(self.ram[:, None, :], axis=1, bitorder="little").reshape(64, 64)
        return np.roll(rows, -self.start_line, axis=0).astype(bool)


class FakeGlcdBus:
    """Stands in for the lgpio library, and behind it the wrist LCD's 20 pins.

    The display code calls lgpio functions to set pins. This class keeps every
    pin's level and acts like the real LCD: when E goes from high to low, each
    half whose chip select is asserted reads the byte on DB0-DB7, as a command
    (D/I low) or as pixels (D/I high). A half that isn't selected ignores it,
    so a wrong-half bug shows up on a laptop the same way it would on the glove.

    Use it in a test the same way test_hardware.py swaps in its fake SPI:
        with mock.patch.object(settings.routine, "lgpio", FakeGlcdBus(), create=True):
    """

    def __init__(self):
        self.halves = [FakeLcdHalf(), FakeLcdHalf()]    # 0 = left (/CS1), 1 = right (/CS2)
        self.levels = {}                                # pin number -> 0 or 1, for every claimed pin
        self.groups = {}                                # first pin of a group -> all its pins, in bit order
        self.log = []                                   # every byte the LCD took: (half, is_data, value)

    # The lgpio functions the display code uses ---------------------------------

    def gpiochip_open(self, chip):
        return 0                                        # a handle; there is only one chip

    def gpiochip_close(self, handle):
        pass

    def gpio_claim_output(self, handle, gpio, level=0, lFlags=0):
        self._claim(gpio)
        self._set(gpio, level)

    def group_claim_output(self, handle, gpios, levels=(0,), lFlags=0):
        for gpio in gpios:
            self._claim(gpio)
        self.groups[gpios[0]] = list(gpios)
        for bit, gpio in enumerate(gpios):
            self._set(gpio, levels[bit] if bit < len(levels) else 0)

    def gpio_write(self, handle, gpio, level):
        if gpio not in self.levels:
            raise RuntimeError(f"GPIO {gpio} written before it was claimed")
        self._set(gpio, level)

    def group_write(self, handle, gpio, group_bits, group_mask=0xFF):
        for bit, pin in enumerate(self.groups[gpio]):   # bit 0 goes to the group's first pin, and so on
            if group_mask >> bit & 1:
                self._set(pin, group_bits >> bit & 1)

    def gpio_free(self, handle, gpio):
        self.levels.pop(gpio, None)

    def group_free(self, handle, gpio):
        for pin in self.groups.pop(gpio, []):
            self.levels.pop(pin, None)

    # What the LCD does when its pins change ------------------------------------

    def _claim(self, gpio):
        if gpio in self.levels:
            raise RuntimeError(f"GPIO busy: {gpio} is already claimed")   # what lgpio says too

    def _set(self, gpio, level):
        old, new = self.levels.get(gpio), 1 if level else 0
        self.levels[gpio] = new
        if gpio == LCD_E_PIN and old == 1 and new == 0:
            self._latch()                               # the LCD reads the byte as E goes low
        elif gpio == LCD_RST_PIN and new == 0 and old != 0:
            for half in self.halves:                    # RST pulled low resets both halves
                half.reset()

    def _latch(self):
        if self.levels.get(LCD_RST_PIN, 0) == 0:
            return                                      # held in reset: the LCD ignores everything
        value = sum(self.levels.get(pin, 0) << bit for bit, pin in enumerate(LCD_DATA_PINS))
        is_data = self.levels.get(LCD_DI_PIN, 0) == 1
        selected_level = 0 if LCD_CS_ACTIVE_LOW else 1
        for index, cs_pin in enumerate((LCD_CS1_PIN, LCD_CS2_PIN)):
            if self.levels.get(cs_pin) == selected_level:
                half = self.halves[index]
                half.data(value) if is_data else half.command(value)
                self.log.append((index, is_data, value))

    # For tests and --mock runs ----------------------------------------------------

    def pixels(self):
        """The whole screen as a 64x128 array of True/False (row, column), as it looks on the glass."""
        return np.hstack([half.pixels() for half in self.halves])

    def bytes_written(self, data_only=True):
        """How many bytes the LCD has taken: pixel bytes only, or commands too."""
        return sum(1 for _, is_data, _ in self.log if is_data or not data_only)

    def preview(self):
        """The screen as text: each character is 2 pixel rows (top, bottom), so 32 lines x 128 columns."""
        pixels = self.pixels()
        blocks = {(False, False): " ", (True, False): "\u2580", (False, True): "\u2584", (True, True): "\u2588"}
        lines = ["".join(blocks[(bool(top), bool(bottom))] for top, bottom in zip(pixels[row], pixels[row + 1]))
                 for row in range(0, 64, 2)]
        edge = "+" + "-" * 128 + "+"
        return "\n".join([edge] + [f"|{line}|" for line in lines] + [edge])


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