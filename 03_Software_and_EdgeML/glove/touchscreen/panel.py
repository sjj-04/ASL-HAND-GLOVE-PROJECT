"""
The Winstar's touch panel: a 4-wire resistive panel.

The panel is two see-through resistive sheets, X and Y. Pressing it makes them
touch at one spot. To find that spot:

  X position: put 3.3 V on X+ and 0 V on X-, so the X sheet has a voltage that
              rises from left to right. Leave the Y sheet floating and measure
              it at Y+: it picks up the voltage where you are pressing.
  Y position: the same with the sheets swapped (3.3 V on Y+, 0 V on Y-, measure at X+).
  Touched?:   0 V on Y-, a pull-up on X+. If nothing is pressed X+ stays high;
              pressing connects it to Y- and pulls it low.

The Pi can't measure voltages, so X+ and Y+ also go to spare MCP3208 channels.
The pins and channels are the TOUCH_ settings in settings.py.

TouchInput turns readings into taps. MockTouch does the same for clicks in
the laptop simulator. Both have the same get_tap() and clear().
"""

import threading
import time
from collections import deque

import settings as ts                 # the TOUCH_ pins and numbers are in settings.py
from .screens import HEIGHT, WIDTH

TOUCH_SETTLE_SECONDS = 0.0005  # wait after switching the panel's pins, before reading
TOUCH_SAMPLES = 3              # readings per axis; the median is used
TAP_STABLE_POLLS = 2           # a touch must be seen this many polls in a row to count
TAP_MAX_JITTER = 8             # ...and stay within this many pixels
RELEASE_POLLS = 2              # polls without a touch before the next tap can count


def to_pixel(raw_x, raw_y):
    """Raw panel readings -> screen pixel (x 0-127, y 0-63), using the numbers in touch_settings.
    If a direction is mirrored, swap that axis's MIN and MAX."""
    if ts.TOUCH_SWAP_XY:
        raw_x, raw_y = raw_y, raw_x
    x = (raw_x - ts.TOUCH_X_MIN) * (WIDTH - 1) / (ts.TOUCH_X_MAX - ts.TOUCH_X_MIN)
    y = (raw_y - ts.TOUCH_Y_MIN) * (HEIGHT - 1) / (ts.TOUCH_Y_MAX - ts.TOUCH_Y_MIN)
    return min(WIDTH - 1, max(0, round(x))), min(HEIGHT - 1, max(0, round(y)))


class ResistiveTouch:
    """Reads the panel through four GPIO pins and two MCP3208 channels."""

    def __init__(self, adc, lgpio):
        self.adc = adc                      # the glove's MCP3208 (the sensor sampler shares it)
        self.lg = lgpio
        self.chip = lgpio.gpiochip_open(0)
        self.claimed = set()
        for pin in (ts.TOUCH_XP_PIN, ts.TOUCH_XM_PIN, ts.TOUCH_YP_PIN, ts.TOUCH_YM_PIN):
            self._set(pin, None)            # every side floating to start with

    def _set(self, pin, mode):
        """mode: 1 or 0 = drive high or low, None = floating input, "pull-up" = input with pull-up."""
        if pin in self.claimed:
            self.lg.gpio_free(self.chip, pin)
        if mode is None:
            self.lg.gpio_claim_input(self.chip, pin)
        elif mode == "pull-up":
            self.lg.gpio_claim_input(self.chip, pin, self.lg.SET_PULL_UP)
        else:
            self.lg.gpio_claim_output(self.chip, pin, mode)
        self.claimed.add(pin)

    def touched(self):
        self._set(ts.TOUCH_XM_PIN, None)
        self._set(ts.TOUCH_YP_PIN, None)
        self._set(ts.TOUCH_YM_PIN, 0)
        self._set(ts.TOUCH_XP_PIN, "pull-up")
        time.sleep(TOUCH_SETTLE_SECONDS)
        return self.lg.gpio_read(self.chip, ts.TOUCH_XP_PIN) == 0

    def _axis(self, high, low, floating, channel):
        for pin in floating:
            self._set(pin, None)
        self._set(low, 0)
        self._set(high, 1)
        time.sleep(TOUCH_SETTLE_SECONDS)
        readings = sorted(self.adc.read(channel) for _ in range(TOUCH_SAMPLES))
        return readings[len(readings) // 2]                 # the median ignores one wild reading

    def read(self):
        """(raw_x, raw_y) while the panel is pressed, otherwise None."""
        if not self.touched():
            return None
        raw_x = self._axis(ts.TOUCH_XP_PIN, ts.TOUCH_XM_PIN, (ts.TOUCH_YP_PIN, ts.TOUCH_YM_PIN),
                           ts.TOUCH_YP_CHANNEL)
        raw_y = self._axis(ts.TOUCH_YP_PIN, ts.TOUCH_YM_PIN, (ts.TOUCH_XP_PIN, ts.TOUCH_XM_PIN),
                           ts.TOUCH_XP_CHANNEL)
        if not self.touched():                              # lifted while we were reading: throw it away
            return None
        return raw_x, raw_y

    def close(self):
        for pin in list(self.claimed):
            self.lg.gpio_free(self.chip, pin)
        self.claimed.clear()
        self.lg.gpiochip_close(self.chip)


class TapDetector:
    """Turns a stream of touch points (or None) into single taps.

    A tap counts once the touch has been seen TAP_STABLE_POLLS times in a row
    without moving more than TAP_MAX_JITTER pixels. Holding a finger down
    doesn't repeat it: the panel must be released first.
    """

    def __init__(self):
        self.points = []
        self.released = True
        self.untouched_polls = RELEASE_POLLS

    def update(self, point):
        if point is None:
            self.points = []
            self.untouched_polls += 1
            if self.untouched_polls >= RELEASE_POLLS:
                self.released = True
            return None
        self.untouched_polls = 0
        if not self.released:
            return None
        self.points.append(point)
        recent = self.points[-TAP_STABLE_POLLS:]
        if len(recent) < TAP_STABLE_POLLS:
            return None
        xs, ys = [p[0] for p in recent], [p[1] for p in recent]
        if max(xs) - min(xs) > TAP_MAX_JITTER or max(ys) - min(ys) > TAP_MAX_JITTER:
            self.points = recent[1:]                        # still moving: keep watching
            return None
        self.released, self.points = False, []
        return round(sum(xs) / len(xs)), round(sum(ys) / len(ys))

    def wait_for_release(self):
        self.released, self.points, self.untouched_polls = False, [], 0


class TouchInput:
    """The real panel, read once per pass of the main loop."""

    def __init__(self, panel):
        self.panel = panel
        self.detector = TapDetector()

    def get_tap(self):
        raw = self.panel.read()
        return self.detector.update(None if raw is None else to_pixel(*raw))

    def clear(self):
        """Forget anything touched during calibration; a finger still down must lift first."""
        self.detector.wait_for_release()


class MockTouch:
    """Taps from the laptop simulator (a mouse click) or a typed command."""

    def __init__(self):
        self.taps = deque()
        self.lock = threading.Lock()

    def tap(self, x, y):
        with self.lock:
            self.taps.append((x, y))

    def get_tap(self):
        with self.lock:
            return self.taps.popleft() if self.taps else None

    def clear(self):
        with self.lock:
            self.taps.clear()
