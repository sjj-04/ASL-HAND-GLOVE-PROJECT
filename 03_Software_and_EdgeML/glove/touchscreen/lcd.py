"""
Puts Screens on the Winstar, using the glove's own WristDisplay driver unchanged.

WristDisplay.message() rewrites all 1,024 bytes every time, which is too slow
for an animated progress bar on a Pi Zero. Like live_readings.py, this sends
only the bytes that changed since the last screen.

It also has a message(title, detail) method, so the glove's own
run_calibration() can draw on it: "CALIBRATING / Hold hand FLAT" becomes the
calibration screen with a moving progress bar, and every other message
("Calibration saved", "Check Ring", "FAULT") becomes a message screen.
"""

import threading
import time

import numpy as np

from . import screens

ANIMATION_SECONDS = 0.15       # how often the calibration progress bar moves


def pack(image):
    """The picture as the LCD stores it: 8 pages x 128 columns, top pixel in bit 0 (as WristDisplay does)."""
    pixels = np.array(image, dtype=np.uint8).reshape(8, 8, 128)
    return np.packbits(pixels, axis=1, bitorder="little")[:, 0, :]


class TouchLcd:

    def __init__(self, display, routine, on_message=None):
        """display: the glove's WristDisplay. routine: glove_calibration, for its pose timings.
        on_message: optional function(title, detail), used by the pretend hand in --sim."""
        self.display = display
        self.routine = routine
        self.on_message = on_message
        self.lock = threading.RLock()       # one screen update at a time (the progress bar has its own thread)
        self.on_lcd = None                  # what the LCD shows now; None = unknown, so send everything
        self.current = None                 # the Screen shown now (its buttons are what a tap hits)
        self.last_summary = None
        self.generation = 0                 # bumped to stop the progress-bar thread

    # What the app uses ------------------------------------------------------------

    def show(self, screen):
        self.stop_animation()
        self._tell_hand("screen", "")
        self._push(screen)

    # What run_calibration() uses ----------------------------------------------------

    def message(self, title, detail=""):
        self._tell_hand(title, detail)
        if title == "CALIBRATING":
            self._animate(detail, 2 if "FIST" in detail.upper() else 1)
        else:
            self.stop_animation()
            self._push(screens.message(title, detail))

    def stop_animation(self):
        with self.lock:
            self.generation += 1

    # Inside ------------------------------------------------------------------------

    def _tell_hand(self, title, detail):
        if self.on_message:
            self.on_message(title, detail)

    def _animate(self, prompt, step):
        """Show the calibration screen and fill its bar over the time the pose is held."""
        with self.lock:
            self.generation += 1
            generation = self.generation
            self._push(screens.calibrating(prompt, step, 0.0))
        duration = self.routine.POSE_SETTLE_SECONDS + self.routine.AVERAGE_SECONDS
        start = time.monotonic()

        def run():
            while True:
                time.sleep(ANIMATION_SECONDS)
                progress = (time.monotonic() - start) / duration if duration > 0 else 1.0
                with self.lock:
                    if generation != self.generation:
                        return                          # a new screen replaced this one
                    self._push(screens.calibrating(prompt, step, progress))
                if progress >= 1.0:
                    return

        threading.Thread(target=run, daemon=True).start()

    def _push(self, screen):
        with self.lock:
            self.current = screen
            if screen.summary != self.last_summary:
                print(f"[LCD] {screen.summary}")
                self.last_summary = screen.summary
            if self.display is None or getattr(self.display, "chip", None) is None:
                return                                  # no LCD: the terminal line above is all there is
            frame = pack(screen.image)
            changed = np.ones(frame.shape, dtype=bool) if self.on_lcd is None else frame != self.on_lcd
            for half in (0, 1):                         # 0: left half (/CS1), 1: right half (/CS2)
                first = half * 64
                if not changed[:, first:first + 64].any():
                    continue
                self.display.select(half)
                for page in range(8):
                    moved = np.flatnonzero(changed[page, first:first + 64])
                    if moved.size == 0:
                        continue
                    start, end = int(moved[0]), int(moved[-1])
                    self.display.write(0xB8 | page, data=False)      # go to this page
                    self.display.write(0x40 | start, data=False)     # and the first changed column
                    for byte in frame[page, first + start:first + end + 1]:
                        self.display.write(int(byte), data=True)     # the column moves right by itself
            self.on_lcd = frame
