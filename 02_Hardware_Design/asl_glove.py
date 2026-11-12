#!/usr/bin/env python3
"""ECET400 hardware bring-up: Winstar pixels, five flex channels, optional touch."""

import argparse
import csv
import signal
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from flex_calibration import bend_percent, load_calibration, save_calibration
from mcp3208 import FINGERS, MCP3208
from winstar_lcd import WinstarLCD


DEFAULT_CALIBRATION = Path(__file__).resolve().with_name("flex_calibration.json")


def font():
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 9)
    except OSError:
        return ImageFont.load_default()


def text_image(lines):
    image = Image.new("1", (128, 64), 0)
    draw = ImageDraw.Draw(image)
    for row, line in enumerate(lines[:6]):
        draw.text((1, row * 10), str(line), font=font(), fill=1, anchor="lt")
    return image


def display_pattern():
    image = Image.new("1", (128, 64), 0)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 127, 63), outline=1)
    draw.line((63, 15, 63, 63), fill=1)
    draw.line((0, 14, 127, 14), fill=1)
    draw.text((10, 3), "WINSTAR READY", font=font(), fill=1, anchor="lt")
    draw.text((5, 20), "LEFT / CS1", font=font(), fill=1, anchor="lt")
    draw.text((68, 20), "RIGHT/CS2", font=font(), fill=1, anchor="lt")
    draw.rectangle((7, 36, 29, 55), outline=1)
    draw.line((7, 36, 29, 55), fill=1)
    draw.ellipse((87, 36, 108, 56), outline=1)
    return image


def display_test(args):
    with WinstarLCD() as lcd:
        print("Showing border, both halves, text and shapes. Adjust RV1 for contrast.")
        print("Ctrl+C ends the test and disables pixels; the backlight stays powered.")
        pattern = display_pattern()
        start = time.monotonic()
        last_phase = None
        while not args.seconds or time.monotonic() - start < args.seconds:
            phase = int((time.monotonic() - start) / 4) % 2
            if phase != last_phase:
                lcd.show(pattern if phase == 0 else ImageOps.invert(pattern.convert("L")))
                last_phase = phase
            time.sleep(0.05)


def draw_image(args):
    with Image.open(args.image) as source:
        image = source.convert("1")
    if image.size != (128, 64):
        raise ValueError("Supply a 128 x 64 image; resize it deliberately before sending")
    with WinstarLCD() as lcd:
        lcd.show(image)
        print("Image displayed. Ctrl+C ends display; pixels turn off on exit.")
        while True:
            time.sleep(0.25)


def sensor_test(args):
    print("CH0 Thumb | CH1 Index | CH2 Middle | CH3 Ring | CH4 Little", flush=True)
    print("Ctrl+C ends acquisition. ADC counts range 0..4095.", flush=True)
    with ExitStack() as stack:
        adc = stack.enter_context(MCP3208())
        writer = None
        if args.csv:
            output = stack.enter_context(open(args.csv, "w", newline="", encoding="utf-8"))
            writer = csv.writer(output)
            writer.writerow(["monotonic_ns"] + list(FINGERS))
        while True:
            start = time.monotonic()
            frame = adc.read_frame(args.samples)
            print(" ".join("{}={:4d}".format(name, value)
                           for name, value in zip(FINGERS, frame.counts)), flush=True)
            if writer:
                writer.writerow([frame.timestamp_ns] + list(frame.counts))
                output.flush()
            time.sleep(max(0, 1.0 / args.rate - (time.monotonic() - start)))


def capture_pose(adc, lcd, label):
    lines = ["FLEX CALIBRATION", label, "Hold all 5 fingers", "Press ENTER on", "your keyboard"]
    lcd.show(text_image(lines))
    input(label + " Hold steady, then press ENTER: ")
    lcd.show(text_image(["FLEX CALIBRATION", label, "Sampling...", "Hold steady"]))
    frames = []
    for _ in range(30):
        frames.append(adc.read_frame(4).counts)
        time.sleep(0.02)
    return [round(sum(frame[ch] for frame in frames) / len(frames)) for ch in range(5)]


def calibrate(args):
    with MCP3208() as adc, WinstarLCD() as lcd:
        flat = capture_pose(adc, lcd, "FINGERS STRAIGHT")
        bent = capture_pose(adc, lcd, "FINGERS BENT")
        save_calibration(args.calibration, flat, bent)
        lcd.show(text_image(["CALIBRATION SAVED", "Ready for live", "sensor readings"]))
        print("Saved {}".format(args.calibration))
        print("Straight counts:", flat, "Bent counts:", bent)
        time.sleep(2)


def live_image(frame, calibration=None, touch_enabled=False, touch=None):
    percentages = bend_percent(frame.counts, calibration) if calibration else None
    if touch_enabled:
        title = "Touch: --" if touch is None else "T X{:4d} Y{:4d}".format(*touch)
    else:
        title = "ASL FLEX / {}".format("CAL" if calibration else "RAW")
    lines = [title]
    for index, (name, value) in enumerate(zip(FINGERS, frame.counts)):
        suffix = " {:3.0f}%".format(percentages[index]) if percentages else ""
        lines.append("{:<6} {:4d}{}".format(name, value, suffix))
    return text_image(lines)


def live(args):
    calibration = (load_calibration(args.calibration)
                   if Path(args.calibration).is_file() else None)
    if calibration is None:
        print("No calibration file: displaying raw counts. Run calibrate for bend %.", flush=True)
    with ExitStack() as stack:
        adc = stack.enter_context(MCP3208())
        lcd = stack.enter_context(WinstarLCD())
        touch_panel = None
        if args.touch:
            from touch_panel import TouchPanel
            touch_panel = stack.enter_context(TouchPanel(adc))
        lcd.show(text_image(["ASL LEARNING GLOVE", "Hardware starting", "Reading sensors..."]))
        next_display = next_console = 0
        while True:
            start = time.monotonic()
            frame = adc.read_frame(args.samples)
            now = time.monotonic()
            if now >= next_display:
                touch = touch_panel.read_raw() if touch_panel else None
                lcd.show(live_image(frame, calibration, args.touch, touch))
                next_display = time.monotonic() + 1.0 / args.display_rate
            if now >= next_console:
                print("{} {}".format(frame.timestamp_ns, frame.counts), flush=True)
                next_console = now + 1.0
            time.sleep(max(0, 1.0 / args.rate - (time.monotonic() - start)))


def touch_test(_args):
    from touch_panel import TouchPanel
    print("Touch raw test. MCP23017 address 0x20. Ctrl+C ends test.")
    print("CH5/CH6 must have their old ground connections REMOVED.")
    with MCP3208() as adc, TouchPanel(adc) as panel:
        while True:
            result = panel.read_raw()
            print("No press" if result is None else "Raw X={} Y={}".format(*result), flush=True)
            time.sleep(0.1)


def positive_float(value):
    number = float(value)
    if not 0 < number <= 100:
        raise argparse.ArgumentTypeError("Use a rate greater than 0 and at most 100 Hz")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    display = commands.add_parser("display-test", help="Text, shapes and both LCD halves")
    display.add_argument("--seconds", type=int, default=0, help="0 runs until Ctrl+C")
    display.set_defaults(run=display_test)
    image = commands.add_parser("draw-image", help="Display a 128 x 64 PNG/BMP image")
    image.add_argument("image")
    image.set_defaults(run=draw_image)
    sensors = commands.add_parser("sensor-test", help="Five flex readings in terminal")
    sensors.add_argument("--csv", help="Optional CSV log (overwrites this file)")
    sensors.set_defaults(run=sensor_test)
    app = commands.add_parser("live", help="Show all five readings on the Winstar LCD")
    app.add_argument("--display-rate", type=positive_float, default=2,
                     help="Display refresh target; 2 Hz default, actual rate may be slower")
    app.add_argument("--touch", action="store_true", help="Also scan the MCP23017 touch branch")
    app.set_defaults(run=live)
    calibration = commands.add_parser("calibrate", help="Save straight and bent counts")
    calibration.set_defaults(run=calibrate)
    for command in (app, calibration):
        command.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    for command in (sensors, app):
        command.add_argument("--rate", type=positive_float, default=10,
                             help="Sensor-frame target Hz; actual achieved rate may be slower")
        command.add_argument("--samples", type=int, choices=range(1, 65), default=4)
    touch = commands.add_parser("touch-test", help="Optional raw touch test, no position calibration")
    touch.set_defaults(run=touch_test)
    args = parser.parse_args(argv)
    if getattr(args, "seconds", 0) < 0:
        parser.error("--seconds must be 0 or positive")
    try:
        args.run(args)
        return 0
    except KeyboardInterrupt:
        print("\nStopped. LCD pixels disabled; backlight remains powered if connected.")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print("Error: {}".format(error), file=sys.stderr)
        print("See README.md: dependencies, bus enablement, permissions and wiring.", file=sys.stderr)
        return 1


def stop_on_signal(_signum, _frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, stop_on_signal)
    sys.exit(main())
