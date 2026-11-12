#!/usr/bin/env python3
"""Winstar touchscreen menu and personalized A/B/C glove recognition."""
import argparse
import collections
import contextlib
import signal
import statistics
import sys
import textwrap
import threading
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from mcp3208 import FINGERS, MCP3208
from recognition import (LETTERS, HoldGate, classify, load_profile, normalize,
                         save_profile, sensor_fault, validate_profile)
from touch_mapping import TARGETS, fit, hit, load_mapping, project, save_mapping
from touch_panel import TouchPanel
from winstar_lcd import WinstarLCD

LEARNING = (3, 16, 124, 36)
FREE = (3, 40, 124, 61)
MENU = (89, 0, 127, 13)
SETUP = (83, 49, 127, 63)
RECORD = (3, 49, 124, 63)
RETRY = (66, 49, 124, 63)
BACK = (3, 49, 61, 63)


def font(size=9):
    try:
        return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', size)
    except OSError:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()


def canvas():
    image = Image.new('1', (128, 64), 0)
    return image, ImageDraw.Draw(image)


def label(draw, xy, value, size=9):
    draw.text(xy, str(value), font=font(size), fill=1, anchor='lt')


def button(draw, rectangle, title):
    draw.rectangle(rectangle, outline=1)
    box = draw.textbbox((0, 0), title, font=font())
    width, height = box[2] - box[0], box[3] - box[1]
    left, top, right, bottom = rectangle
    label(draw, (left + (right-left-width)//2, top + (bottom-top-height)//2), title)


def home_image():
    image, draw = canvas()
    label(draw, (35, 2), 'ASL TOOL', 11)
    button(draw, LEARNING, 'Learning')
    button(draw, FREE, 'Free Mode')
    return image


def message_image(title, lines=(), buttons=()):
    image, draw = canvas()
    label(draw, (2, 2), title)
    for row, line in enumerate(lines[:3]):
        label(draw, (2, 15 + row*10), line)
    for rectangle, text in buttons:
        button(draw, rectangle, text)
    return image


def free_image(letter=None, status='SIGN A/B/C'):
    image, draw = canvas()
    label(draw, (2, 2), 'FREE ABC')
    button(draw, MENU, 'MENU')
    label(draw, (54, 15), letter or '-', 25)
    label(draw, (2, 40), status)
    label(draw, (2, 53), 'ABC only')
    button(draw, SETUP, 'SETUP')
    return image


class TouchUI:
    def __init__(self, panel, mapping=None):
        self.panel, self.mapping, self.lock = panel, mapping, None

    def raw(self):
        with self.lock if self.lock is not None else contextlib.nullcontext():
            return self.panel.read_raw()

    def position(self):
        raw = self.raw()
        return None if raw is None else project(raw, self.mapping)

    def wait_release(self, timeout=120):
        end, released = time.monotonic() + timeout, 0
        while released < 2:
            if time.monotonic() >= end:
                raise RuntimeError('Touch stayed pressed; check panel wiring or release the screen')
            released = released + 1 if self.raw() is None else 0
            time.sleep(0.02)

    def wait_tap(self, rectangles):
        self.wait_release()
        while True:
            point = self.position()
            if point is not None:
                for index, rectangle in enumerate(rectangles):
                    if hit(point, rectangle):
                        self.wait_release()
                        return index
            time.sleep(0.02)


def calibrate_touch(lcd, ui, path):
    raw_points = []
    for index, (x, y) in enumerate(TARGETS):
        ui.wait_release()
        image, draw = canvas()
        label(draw, (2, 2), 'TOUCH {}/5: HOLD +'.format(index+1))
        draw.line((x-4, y, x+4, y), fill=1)
        draw.line((x, y-4, x, y+4), fill=1)
        lcd.show(image)
        end = time.monotonic() + 120
        samples = []
        while len(samples) < 12:
            if time.monotonic() > end:
                raise RuntimeError('No stable touch capture; check raw touch with asl_glove.py touch-test')
            point = ui.raw()
            if point is None:
                samples = []
            else:
                samples.append(point)
            time.sleep(0.02)
        raw_points.append(tuple(statistics.median(point[axis] for point in samples) for axis in (0, 1)))
        print('Touch target', index+1, raw_points[-1], flush=True)
    ui.wait_release()
    mapping = fit(raw_points)
    save_mapping(path, mapping)
    ui.mapping = mapping
    print('Touch mapping saved:', path, flush=True)


def capture_pose(adc, lcd, ui, title, instructions):
    lcd.show(message_image(title, instructions, [(RECORD, 'RECORD')]))
    ui.wait_tap([RECORD])
    for second in (2, 1):
        lcd.show(message_image(title, ['Hold your sign still', 'Recording in ' + str(second)]))
        time.sleep(1)
    lcd.show(message_image(title, ['RECORDING...', 'Keep still']))
    samples, end = [], time.monotonic() + 1.25
    while time.monotonic() < end:
        values = adc.read_frame(samples=2).counts
        fault = sensor_fault(values)
        if fault:
            raise ValueError('SENSOR FAULT ' + fault)
        samples.append(values)
        time.sleep(0.02)
    if len(samples) < 5:
        raise ValueError('Too few sensor frames; repeat setup')
    if any(max(row[ch] for row in samples) - min(row[ch] for row in samples) > 120 for ch in range(5)):
        raise ValueError('Hand moved during capture; keep still and repeat')
    return [round(statistics.median(row[ch] for row in samples)) for ch in range(5)]


def setup_glove(adc, lcd, ui, path):
    while True:
        try:
            flat = capture_pose(adc, lcd, ui, 'GLOVE: FLAT HAND', ['Straighten all fingers', 'Then tap RECORD'])
            bent = capture_pose(adc, lcd, ui, 'GLOVE: CLOSED FIST', ['Bend all fingers', 'Then tap RECORD'])
            profile = {'version': 1, 'fingers': list(FINGERS), 'flat': flat, 'bent': bent, 'references': {}}
            if any(abs(b-a) < 200 for a, b in zip(flat, bent)):
                raise ValueError('A finger changed less than 200 counts; check sensors')
            for letter in LETTERS:
                samples = []
                for number in range(1, 4):
                    raw = capture_pose(adc, lcd, ui, 'SIGN {}: SAMPLE {}/3'.format(letter, number),
                                       ['Form ASL letter ' + letter, 'Then tap RECORD'])
                    samples.append(list(normalize(raw, profile)))
                    print('Recorded', letter, number, raw, flush=True)
                profile['references'][letter] = samples
            validate_profile(profile)
            save_profile(path, profile)
            print('Glove profile saved:', path, flush=True)
            return profile
        except ValueError as error:
            print('Setup:', error, flush=True)
            lcd.show(message_image('SETUP INCOMPLETE', textwrap.wrap(str(error), width=23),
                                   [(BACK, 'MENU'), (RETRY, 'RETRY')]))
            if ui.wait_tap([BACK, RETRY]) == 0:
                return None


class GloveReader(threading.Thread):
    """Acquire while the main thread draws the LCD. SPI access is serialized with touch."""
    def __init__(self, adc, profile, bus_lock):
        super().__init__(daemon=True)
        self.adc, self.profile, self.bus_lock = adc, profile, bus_lock
        self.stop_event, self.state_lock = threading.Event(), threading.Lock()
        self.state = (None, 'WARMING UP', time.monotonic())

    def publish(self, letter, status):
        with self.state_lock:
            self.state = letter, status, time.monotonic()

    def snapshot(self):
        with self.state_lock:
            return self.state

    def run(self):
        recent, gate = collections.deque(maxlen=5), HoldGate()
        recovering, valid_since = False, None
        while not self.stop_event.is_set():
            start = time.monotonic()
            try:
                with self.bus_lock:
                    frame = self.adc.read_frame(samples=2)
                fault = sensor_fault(frame.counts)
                if fault:
                    raise ValueError('FAULT ' + fault)
                if recovering:
                    if valid_since is None:
                        valid_since = time.monotonic()
                    if time.monotonic() - valid_since < 1:
                        self.publish(None, 'RECOVERING...')
                        self.stop_event.wait(0.05)
                        continue
                    recovering = False
                recent.append(frame.counts)
                if len(recent) < 5:
                    self.publish(None, 'WARMING UP')
                else:
                    values = [statistics.median(row[ch] for row in recent) for ch in range(5)]
                    candidate = classify(normalize(values, self.profile), self.profile)
                    accepted = gate.update(candidate, frame.timestamp_ns / 1e9)
                    status = 'MATCH' if accepted else ('HOLD ' + candidate if candidate else 'NO MATCH')
                    self.publish(accepted, status)
            except (OSError, ValueError, RuntimeError) as error:
                recent.clear()
                gate.reset()
                recovering, valid_since = True, None
                self.publish(None, str(error) if isinstance(error, ValueError) else 'ADC ERROR')
            self.stop_event.wait(max(0, 0.05 - (time.monotonic() - start)))

    def stop(self):
        self.stop_event.set()
        self.join(timeout=3)
        if self.is_alive():
            raise RuntimeError('ADC thread did not stop; stop the program before another hardware test')


def free_screen(adc, lcd, ui, profile):
    lock = threading.RLock()
    reader = GloveReader(adc, profile, lock)
    ui.lock = lock
    reader.start()
    down, released, next_draw, previous = False, 0, 0, None
    try:
        while True:
            point = ui.position()
            if point is None:
                released += 1
                if released >= 2:
                    down = False
            else:
                released = 0
                if not down:
                    down = True
                    if hit(point, MENU):
                        return 'menu'
                    if hit(point, SETUP):
                        return 'setup'
            letter, status, stamp = reader.snapshot()
            if time.monotonic() - stamp > 1:
                letter, status = None, 'ADC DATA STALE'
            now = time.monotonic()
            if now >= next_draw and previous != (letter, status):
                lcd.show(free_image(letter, status))
                previous = letter, status
                next_draw = time.monotonic() + 0.25
            time.sleep(0.02)
    finally:
        reader.stop()
        ui.lock = None


def run_session(adc, lcd, panel, args):
    directory = Path(args.profile_dir).expanduser()
    touch_path, glove_path = directory/'touch.json', directory/'glove_profile.json'
    ui = TouchUI(panel)
    if args.touch_calibrate or not touch_path.is_file():
        calibrate_touch(lcd, ui, touch_path)
    else:
        ui.mapping = load_mapping(touch_path)
    profile = None
    if glove_path.is_file():
        try:
            profile = load_profile(glove_path)
        except ValueError as error:
            print('Glove setup required:', error, flush=True)
    while True:
        lcd.show(home_image())
        selection = ui.wait_tap([LEARNING, FREE])
        if selection == 0:
            lcd.show(message_image('LEARNING', ['Lessons reserved for', 'the software team.'], [(RECORD, 'MENU')]))
            ui.wait_tap([RECORD])
            continue
        if profile is None or args.glove_setup:
            new_profile = setup_glove(adc, lcd, ui, glove_path)
            args.glove_setup = False
            if new_profile is None:
                continue
            profile = new_profile
        while True:
            action = free_screen(adc, lcd, ui, profile)
            if action == 'menu':
                break
            new_profile = setup_glove(adc, lcd, ui, glove_path)
            if new_profile is None:
                break
            profile = new_profile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile-dir', default='~/.local/share/asl-freemode')
    parser.add_argument('--touch-calibrate', action='store_true', help='Repeat five-point touch setup')
    parser.add_argument('--glove-setup', action='store_true', help='Repeat flat/fist and A/B/C recording')
    args = parser.parse_args(argv)
    try:
        with MCP3208() as adc, WinstarLCD() as lcd, TouchPanel(adc) as panel:
            run_session(adc, lcd, panel, args)
        return 0
    except KeyboardInterrupt:
        print('\nStopped. LCD pixels disabled; backlight remains powered.')
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print('Error:', error, file=sys.stderr)
        print('See README.md for display/touch checks and setup commands.', file=sys.stderr)
        return 1


def stop_on_signal(_number, _frame):
    raise KeyboardInterrupt


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, stop_on_signal)
    sys.exit(main())
