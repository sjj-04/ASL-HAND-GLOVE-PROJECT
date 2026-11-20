"""
ASL Learning Glove: the main program, and the only file you run.

What it does:
  1. Start-up: check the sensors, then calibrate this user's hand every time:
     "Hold hand FLAT" (the 0% reference), then "Make a FIST" (the 100% reference).
  2. READY, then straight into free practice.
  3. Free practice, 20 times per second: work out which sign the hand is making
     (A, B or C). When the same sign is held for FREE_HOLD_SECONDS, show it on
     the wrist display, then keep watching for the next sign.

Run from this folder:
  py main.py                                  the real glove (on the Pi)
  py main.py --mock                           a pretend glove (any computer)
  py main.py --mock --stuck-finger Ring       a loose sensor: calibration fails
  py main.py --mock --unplugged-finger Ring   an unplugged sensor: SENSOR FAULT at start-up
"""

import argparse
import sys
import time

import settings
from calibration import run_calibration
from hardware import PI_LIBRARIES_FOUND, Sampler, open_hardware
from sign_feedback import FaultGuard, normalize, recognize
from startup_check import find_faulty_finger

FREE_HOLD_SECONDS = 1.0        # hold a sign this long before its letter is shown


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
        free_practice(sampler, display, calibration)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        sampler.stop()
        adc.close()


def start_up(sampler, display, cal_file):
    """Check the sensors, then record this user's flat hand and fist.

    Returns (flat, fist), or None if the glove has to stop.
    """
    faulty = find_faulty_finger(sampler.newest())    # "All five sensors in range?"
    if faulty:
        display.message("SENSOR FAULT", f"Check {faulty}")
        return None

    # Always calibrate at start-up, so the references fit whoever is wearing the glove.
    # run_calibration() shows "Hold hand FLAT", then "Make a FIST", and saves both.
    calibration = run_calibration(sampler, display, cal_file)
    if calibration is None:                          # three failed tries: FAULT is on screen
        return None

    display.message("READY")
    time.sleep(settings.MESSAGE_SECONDS)             # leave READY up long enough to read
    return calibration


def free_practice(sampler, display, calibration):
    """Watch for A, B or C, 20 times per second, until Ctrl+C.

    A letter is shown once the same sign has been held for FREE_HOLD_SECONDS.
    It stays on screen while the glove keeps watching for the next sign.
    To count the same letter again, the user changes pose and then makes it again.
    """
    flat, fist = calibration
    fault_guard = FaultGuard()       # an unplugged or shorted sensor shows SENSOR FAULT
    holding = None                   # the sign the hand is making right now (None = no clear sign)
    held_since = None                # when that sign started; None once it has been counted
    shown = "--"                     # the last letter confirmed

    while True:
        now = time.monotonic()
        raw = sampler.newest()

        faulty = fault_guard.update(raw, now)
        if faulty:                                       # never read signs from a disconnected sensor
            holding = held_since = None
            display.message("SENSOR FAULT", f"Check {faulty}")
            time.sleep(1 / settings.LOOP_HZ)
            continue

        bends = normalize(raw, flat, fist)               # each finger as 0.0 (flat) to 1.0 (fist)
        sign, confidence = recognize(bends)              # closest of A, B, C
        if confidence < settings.ACCEPT_CONFIDENCE:
            sign = None                                  # not close enough to any letter

        if sign != holding:                              # the hand changed: start timing again
            holding = sign
            held_since = now if sign is not None else None
        elif held_since is not None and now - held_since >= FREE_HOLD_SECONDS:
            shown = sign                                 # held long enough: show the letter
            print(f"[SIGN] {sign}")
            held_since = None                            # counted once; change pose to count again

        display.message("Free practice", f"SIGN: {shown}")
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
