"""
ASL Learning Glove: the main program, and the only file you run.

Follows the main program flowchart in Section 6 of the Design Specification:
  Sheet 1  start-up: check the sensors, load or run calibration, show READY
  Sheet 2  the practice loop, 20 times per second: buttons, normalize, recognize
  Sheet 3  feedback: CORRECT, HOLD STEADY, ADJUST: [finger] or TRY AGAIN

Run from this folder:
  py main.py                                  the real glove (on the Pi)
  py main.py --mock                           a pretend glove (any computer)
  py main.py --mock --stuck-finger Ring       a loose sensor: type c + Enter to see calibration fail
  py main.py --mock --unplugged-finger Ring   an unplugged sensor: SENSOR FAULT at start-up
"""

import argparse
import sys
import threading
import time

import settings
from calibration import load_calibration, run_calibration
from hardware import PI_LIBRARIES_FOUND, Sampler, open_hardware
from lesson import LessonManager
from sign_feedback import HoldTimer, practice_step
from startup_check import find_faulty_finger


def main():
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)      # print each line right away, even in Git Bash or a log

    if args.mock:
        from mock_glove import open_mock_glove
        adc, display = open_mock_glove(args.stuck_finger, args.unplugged_finger)
        cal_file = settings.MOCK_CAL_FILE
    elif PI_LIBRARIES_FOUND:
        adc, display = open_hardware()
        cal_file = settings.CAL_FILE
    else:
        raise SystemExit("Raspberry Pi libraries not found. Install them (see README.md), "
                         "or use the pretend glove:  python3 main.py --mock")

    sampler = Sampler(adc)
    sampler.start()                                  # read all five sensors 200 times per second
    try:
        calibration = start_up(sampler, display, cal_file)
        if calibration is None:
            raise SystemExit("Stopped at start-up. The wrist display shows why.")
        practice_loop(sampler, display, calibration)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        sampler.stop()
        adc.close()


def start_up(sampler, display, cal_file):
    """Sheet 1. Returns the (flat, fist) calibration, or None if the glove has to stop."""
    faulty = find_faulty_finger(sampler.newest())    # "All five sensors in range?"
    if faulty:
        display.message("SENSOR FAULT", f"Check {faulty}")
        return None

    calibration = load_calibration(cal_file)         # "Calibration saved on microSD?"
    if calibration is None:
        calibration = run_calibration(sampler, display, cal_file)
        if calibration is None:                      # three failed tries: FAULT is on screen
            return None

    display.message("READY")
    time.sleep(settings.MESSAGE_SECONDS)             # leave READY up long enough to read
    return calibration


def practice_loop(sampler, display, calibration):
    """Sheets 2 and 3: one pass 20 times per second, until Ctrl+C."""
    lesson = LessonManager()
    lesson.select_next()          # no buttons: the lesson starts right after READY
    hold_timer = HoldTimer()
    while True:
        title, detail, success = practice_step(
            sampler.newest(), calibration, lesson.target, hold_timer, time.monotonic())
        display.message(title, detail)

        if success:               # held correctly long enough
            lesson.report_success()   # moves to the next letter
            hold_timer.reset()
            time.sleep(settings.SUCCESS_SECONDS)

        time.sleep(1 / settings.LOOP_HZ)


def parse_args():
    parser = argparse.ArgumentParser(description="ASL Learning Glove")
    parser.add_argument("--mock", action="store_true",
                        help="use a pretend glove, so no Raspberry Pi hardware is needed")
    parser.add_argument("--stuck-finger", choices=settings.FINGERS,
                        help="with --mock: this finger's sensor is loose, so calibration fails")
    parser.add_argument("--unplugged-finger", choices=settings.FINGERS,
                        help="with --mock: this finger's sensor is unplugged (SENSOR FAULT)")
    args = parser.parse_args()
    if (args.stuck_finger or args.unplugged_finger) and not args.mock:
        parser.error("--stuck-finger and --unplugged-finger only work with --mock")
    return args


if __name__ == "__main__":
    main()
