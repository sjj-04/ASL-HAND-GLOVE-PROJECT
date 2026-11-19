#!/usr/bin/env python3
"""
Live CH0-CH4 sensor readings on the Winstar wrist LCD.

Shows each flex sensor's raw ADC reading (0-4095) on the Winstar about 10 times
per second, and prints the measured sampling and refresh rates in the terminal,
which are the numbers Test 2.6 asks for.

It uses the glove's own code exactly as it is. The MCP3208, the 200 Hz Sampler
with its 5-reading median filter and the WristDisplay driver are all imported
from the glove folder. Nothing in that folder is changed, and nothing is added
to it (not even a __pycache__). Keep this script outside the glove folder;

Stop main.py first, because only one program at a time can claim the LCD pins.

Run it from the glove folder (where main.py is):
    ~/glove-env/bin/python ~/live_readings.py              live readings
    ~/glove-env/bin/python ~/live_readings.py --pattern    Test 2.6 number pattern
    python3 ~/live_readings.py --mock                      any computer: pretend glove,
                                                           LCD drawn in the terminal
or from anywhere by adding  --glove /path/to/glove . Ctrl+C quits.
"""

import argparse
import math
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path

sys.dont_write_bytecode = True       # importing the glove code must not add __pycache__ to its folder

import numpy as np

# Screen layout: five rows on the 128x64 LCD, one per channel.
FONT_SIZE = 10                       # DejaVuSans-Bold, the same font the glove's other screens use
ROW_PITCH = 13                       # pixels from one row to the next
TOP = -1                             # first row's y; -1 leaves room for the "y" in Pinky at the bottom
NAME_X = 26                          # where the finger name starts, after "CH0"

# Test 2.6: "Write [0, 9, 10, 99, 100, 999, 1000, 4095] into each numeric field."
# The pattern goes up, then back down, so every change from a longer number to a
# shorter one is shown too. That is where leftover (residual) characters would appear.
TEST_VALUES = [0, 9, 10, 99, 100, 999, 1000, 4095]
PATTERN = TEST_VALUES + TEST_VALUES[-2:0:-1]     # 0 ... 4095, 1000 ... 9, then 0 again


def main():
    args = parse_args()
    sys.path.insert(0, str(find_glove_folder(args.glove)))
    import settings                  # the glove's own files, imported unchanged
    import hardware
    routine = settings.routine
    hz = args.hz or settings.LCD_REFRESH_HZ

    with ExitStack() as stack:
        if args.mock:
            adc, bus = open_mock_glove(stack, routine)
        elif hardware.PI_LIBRARIES_FOUND:
            adc, bus = hardware.MCP3208(), None
        else:
            raise SystemExit("Raspberry Pi libraries not found. Run this on the Pi, "
                             "or try the pretend glove:  python3 live_readings.py --mock")
        from PIL import Image, ImageDraw

        adc = CountingADC(adc)
        display = hardware.WristDisplay()      # resets the LCD and turns it on, as main.py does
        screen = LiveScreen(display, settings.FINGERS, routine.load_font(FONT_SIZE), Image, ImageDraw)
        sampler = hardware.Sampler(adc)
        sampler.start()                        # all five sensors, 200 times per second
        if display.chip is None:
            print("The LCD isn't responding, so the readings will only be shown here.")
        print(f"Live CH0-CH4 on the wrist LCD, target {hz:g} Hz"
              + (", Test 2.6 pattern" if args.pattern else "") + ". Ctrl+C quits.")
        try:
            run(sampler, adc, screen, hz, args.pattern, args.hold, bus)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            sampler.stop()
            adc.close()


def run(sampler, adc, screen, hz, pattern, hold, bus=None):
    """Redraw the LCD hz times per second until Ctrl+C. Prints the measured rates once per second."""
    period = 1.0 / hz
    start = next_frame = report_start = time.monotonic()
    samples_at_report, frames, slowest = adc.samples, 0, 0.0
    status, last_value, last_preview = "measuring...", None, 0.0

    while True:
        frame_start = time.monotonic()
        if pattern:
            value = PATTERN[int((frame_start - start) / hold) % len(PATTERN)]
            values = [value] * len(screen.fingers)
            if value != last_value and bus is None:
                print(f"[pattern] every field: {value}")
            last_value = value
        else:
            values = sampler.newest()          # each finger's median of the last 5 readings
        screen.show(values)
        frames += 1
        now = time.monotonic()
        slowest = max(slowest, now - frame_start)

        if now - report_start >= 1.0:          # once per second: the rates Test 2.6 records
            elapsed = now - report_start
            sampling_hz = (adc.samples - samples_at_report) / elapsed
            status = (f"sampling {sampling_hz:5.1f} Hz | LCD refresh {frames / elapsed:4.1f} Hz | "
                      f"slowest frame {slowest * 1000:3.0f} ms | "
                      + " ".join(f"CH{i} {v:4d}" for i, v in enumerate(screen.last_values)))
            if bus is None:
                print(status)
            report_start, samples_at_report, frames, slowest = now, adc.samples, 0, 0.0

        if bus is not None and now - last_preview >= 0.2:   # --mock: draw the pretend LCD in the terminal
            print("\033[H\033[J" + bus.preview() + "\n" + status, flush=True)
            last_preview = now

        next_frame += period                   # wait for a set clock time, like the Sampler does
        delay = next_frame - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            next_frame = time.monotonic()      # a frame ran long: carry on rather than rushing to catch up


class LiveScreen:
    """The live CH0-CH4 screen. Each redraw sends the LCD only the bytes that changed.

    WristDisplay.message() rewrites all 1,024 bytes of the screen, which is too slow
    to repeat 10 times per second on a Pi Zero. Between two frames of live readings
    only a few digits change, so this compares the new frame with the one already on
    the LCD and, for each page of each half, sends just the run of columns from the
    first changed byte to the last. The old digits are part of what gets rewritten,
    so a shorter number never leaves characters behind.
    """

    def __init__(self, display, fingers, font, Image, ImageDraw):
        self.display = display
        self.fingers = fingers
        self.font = font
        self.Image, self.ImageDraw = Image, ImageDraw
        self.background = Image.new("1", (128, 64))   # the labels, which never change, drawn once
        draw = ImageDraw.Draw(self.background)
        for row, finger in enumerate(fingers):
            y = TOP + row * ROW_PITCH
            draw.text((0, y), f"CH{row}", font=font, fill=1)
            draw.text((NAME_X, y), finger, font=font, fill=1)
        self.on_lcd = None             # unknown at first, so the first frame is sent in full
        self.last_values = [0] * len(fingers)

    def render(self, values):
        """Draw one frame. Returns it the way the LCD stores it: 8 pages x 128 columns, top pixel in bit 0."""
        image = self.background.copy()
        draw = self.ImageDraw.Draw(image)
        for row, value in enumerate(values):
            text = str(value)
            x = 127 - round(draw.textlength(text, font=self.font))   # right-aligned, so the digits line up
            draw.text((x, TOP + row * ROW_PITCH), text, font=self.font, fill=1)
        pixels = np.array(image, dtype=np.uint8).reshape(8, 8, 128)
        return np.packbits(pixels, axis=1, bitorder="little")[:, 0, :]   # same packing as WristDisplay

    def show(self, readings):
        values = [min(4095, max(0, int(round(float(v))))) for v in readings]
        self.last_values = values
        frame = self.render(values)
        if self.display.chip is None:
            return
        changed = np.ones(frame.shape, dtype=bool) if self.on_lcd is None else frame != self.on_lcd
        for half in (0, 1):                          # 0: left half (/CS1), 1: right half (/CS2)
            first_column = half * 64
            if not changed[:, first_column:first_column + 64].any():
                continue
            self.display.select(half)
            for page in range(8):
                moved = np.flatnonzero(changed[page, first_column:first_column + 64])
                if moved.size == 0:
                    continue
                start, end = int(moved[0]), int(moved[-1])
                self.display.write(0xB8 | page, data=False)      # go to this page
                self.display.write(0x40 | start, data=False)     # and to the first changed column
                for byte in frame[page, first_column + start:first_column + end + 1]:
                    self.display.write(int(byte), data=True)     # the column moves right by itself
        self.on_lcd = frame


class CountingADC:
    """Passes every read through to the real ADC and counts full samples, to measure the sampling rate."""

    def __init__(self, adc):
        self.adc = adc
        self.samples = 0

    def read(self, channel):
        if channel == 0:               # the Sampler reads CH0 first in every sample
            self.samples += 1
        return self.adc.read(channel)

    def close(self):
        self.adc.close()


def open_mock_glove(stack, routine):
    """--mock: a pretend hand that bends one finger at a time, and mock_glove's FakeGlcdBus as the LCD."""
    from unittest import mock
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise SystemExit("--mock needs Pillow to draw the screen:  pip install pillow")
    import mock_glove

    bus = mock_glove.FakeGlcdBus()               # acts like the LCD's 20 pins and both KS0108 halves
    for name, value in [("lgpio", bus), ("Image", Image), ("ImageDraw", ImageDraw), ("ImageFont", ImageFont)]:
        stack.enter_context(mock.patch.object(routine, name, value, create=True))

    class SweepingHand(mock_glove.PretendHand):
        """Bends Thumb, then Index ... then Pinky, 2 seconds each, so every channel moves in turn."""

        def __init__(self):
            super().__init__()
            self.start = time.monotonic()

        def pose(self):
            elapsed = time.monotonic() - self.start
            pose = [0.0] * len(mock_glove.FINGERS)
            pose[int(elapsed // 2) % len(pose)] = math.sin(math.pi * (elapsed % 2) / 2)
            return pose

    if os.name == "nt":
        os.system("")                            # lets the Windows console clear the screen between frames
    sys.stdout.reconfigure(encoding="utf-8")     # the preview is drawn with block characters
    return mock_glove.MockMCP3208(SweepingHand()), bus


def find_glove_folder(path):
    folder = Path(path).expanduser().resolve()
    if not ((folder / "settings.py").is_file() and (folder / "hardware.py").is_file()):
        raise SystemExit(f"Can't find the glove code in {folder}.\n"
                         "Run this from the glove folder (where main.py is), or add  --glove /path/to/glove")
    return folder


def parse_args():
    parser = argparse.ArgumentParser(description="Live CH0-CH4 readings on the wrist LCD")
    parser.add_argument("--glove", default=".", help="the glove folder (default: the current folder)")
    parser.add_argument("--pattern", action="store_true",
                        help="Test 2.6: show 0, 9, 10, 99, 100, 999, 1000, 4095 and back down in every field")
    parser.add_argument("--hold", type=float, default=2.0,
                        help="with --pattern: seconds each number stays up (default 2)")
    parser.add_argument("--hz", type=float,
                        help="LCD redraws per second (default: LCD_REFRESH_HZ in settings.py)")
    parser.add_argument("--mock", action="store_true",
                        help="use a pretend glove and draw the LCD in the terminal (no Pi needed)")
    args = parser.parse_args()
    if args.hold <= 0 or (args.hz is not None and args.hz <= 0):
        parser.error("--hold and --hz must be greater than 0")
    return args


if __name__ == "__main__":
    main()
