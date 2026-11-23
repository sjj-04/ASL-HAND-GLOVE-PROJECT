"""
ASL Learning Glove: the main program, and the only file you run.

What it does:
  1. Start-up: check the sensors, then use the saved calibration (calibration.json), so
     the same person doesn't have to calibrate every time. It is kept until someone taps
     Recalibrate. With no saved calibration yet, the touchscreen still opens on the home
     screen and asks for one when Free is tapped: "Hold hand FLAT", "Make a FIST", and with
     the gyroscope "Point fingers UP" and "Thumb UP" (see calibration.py).
  2. READY, then the touchscreen's home screen: [Free] [Lesson] [Recalibrate]
     (or straight into free practice if USE_TOUCHSCREEN is False).
  3. Free practice, 20 times per second: work out which letter the hand is making,
     from the finger bends and, with the gyroscope, which way the hand points and
     how it moves. When the same letter is held for FREE_HOLD_SECONDS (J and Z:
     traced for MOTION_HOLD_SECONDS), show it, then keep watching for the next one.
     Only one letter is ever shown at a time.

Run from this folder:
  py main.py                                  the real glove (on the Pi)
  py main.py --mock                           a pretend glove (any computer): a window shows the
                                              touchscreen; click it to touch it
  py main.py --mock --text                    the same, drawn in the terminal with typed commands
  py main.py --mock --no-touchscreen          the old pretend glove: free practice in the terminal
  py main.py --mock --stuck-finger Ring       a loose sensor: calibration fails
  py main.py --mock --unplugged-finger Ring   an unplugged sensor: SENSOR FAULT at start-up
  py main.py --touch-test                     on the Pi: line the touch panel up with the screen
"""

import argparse
import sys
import time

import settings
from alphabet import MOTION_LETTERS
from calibration import apply_saved_imu_calibration, load_calibration, run_full_calibration
from hardware import PI_LIBRARIES_FOUND, Sampler, open_hardware
from imu import hand_features
from sign_feedback import FaultGuard, normalize, recognize_with_margin
from startup_check import find_faulty_finger

FREE_HOLD_SECONDS = 1.0        # hold a sign this long before its letter is shown


def main():
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)      # print each line right away, even in Git Bash or a log
    touchscreen = settings.USE_TOUCHSCREEN and not args.no_touchscreen

    if args.touch_test:
        from touchscreen.app import touch_test
        return touch_test()
    if args.mock and touchscreen:
        from touchscreen.sim import run_simulator    # the touchscreen in a window (or the terminal)
        return run_simulator(args)

    if args.mock:
        from mock_glove import open_mock_glove, open_mock_imu
        adc, display = open_mock_glove(args.stuck_finger, args.unplugged_finger)
        imu = open_mock_imu(adc.hand) if settings.USE_IMU else None
        cal_file = settings.MOCK_CAL_FILE
    elif PI_LIBRARIES_FOUND:
        from imu import open_imu
        adc, display = open_hardware()
        imu = open_imu()                             # None if there's no gyroscope: flex sensors only
        cal_file = settings.CAL_FILE
    else:
        raise SystemExit("Raspberry Pi libraries not found. Install them (see README.md), "
                         "or use the pretend glove:  python3 main.py --mock")

    sampler = Sampler(adc)
    sampler.start()                                  # read all five sensors 200 times per second
    try:
        if touchscreen:
            from touchscreen.app import run_home
            run_home(sampler, display, adc, imu, cal_file)
        else:
            calibration = start_up(sampler, display, cal_file, imu)
            if calibration is None:
                raise SystemExit("Stopped at start-up. The wrist display shows why.")
            free_practice(sampler, display, calibration, imu)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if imu is not None:
            imu.stop()
        sampler.stop()
        adc.close()


def start_up(sampler, display, cal_file, imu=None, recalibrate=False):
    """Check the sensors, then use the saved calibration, or record one if there is none yet.

    recalibrate: True always records a new one (collect_data.py: every person calibrates
    their own hand before recording).
    Returns (flat, fist), or None if the glove has to stop.
    """
    faulty = find_faulty_finger(sampler.newest())    # "All five sensors in range?"
    if faulty:
        display.message("SENSOR FAULT", f"Check {faulty}")
        return None

    # "Calibration saved on microSD?" The latest one is used until someone recalibrates.
    calibration = None if recalibrate else load_calibration(cal_file)
    if calibration is None:
        if not recalibrate:
            display.message("CALIBRATION", "REQUIRED")
            time.sleep(settings.MESSAGE_SECONDS)
        # Flat hand, fist, and with the gyroscope fingers up and thumb up. Saved in calibration.json.
        if hasattr(display, "calibration_steps"):
            display.calibration_steps = 4 if imu is not None else 2       # the touchscreen's "Step 1 of 4"
        calibration = run_full_calibration(sampler, display, cal_file, imu)
        if calibration is None:                      # three failed tries: FAULT is on screen
            return None
    else:
        apply_saved_imu_calibration(imu, cal_file)   # the gyroscope's saved directions and still reading
    if imu is not None:
        imu.zero_gyro()                              # zero the gyroscope if the hand is resting still

    display.message("READY")
    time.sleep(settings.MESSAGE_SECONDS)             # leave READY up long enough to read
    return calibration


def free_practice(sampler, display, calibration, imu=None):
    """Watch for letters, 20 times per second, until Ctrl+C.

    A letter is shown once the same sign has been held for FREE_HOLD_SECONDS
    (J and Z, which are traced: MOTION_HOLD_SECONDS of tracing).
    It stays on screen while the glove keeps watching for the next sign.
    To count the same letter again, the user changes pose and then makes it again.
    imu: the gyroscope's ImuSampler, or None to read the flex sensors only.
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
        features = hand_features(bends, imu.state() if imu is not None else None)   # + tilt and motion
        sign, confidence, margin = recognize_with_margin(features)   # closest letter, and its lead
        if confidence < settings.ACCEPT_CONFIDENCE or margin < settings.MARGIN:
            sign = None                                  # not close enough, or too close to call

        hold = settings.MOTION_HOLD_SECONDS if sign in MOTION_LETTERS else FREE_HOLD_SECONDS
        if sign != holding:                              # the hand changed: start timing again
            holding = sign
            held_since = now if sign is not None else None
        elif held_since is not None and now - held_since >= hold:
            shown = sign                                 # held long enough: show the letter
            print(f"[SIGN] {sign}")
            held_since = None                            # counted once; change pose to count again

        display.message("Free practice", f"SIGN: {shown}")
        time.sleep(1 / settings.LOOP_HZ)


def parse_args():
    parser = argparse.ArgumentParser(description="ASL Learning Glove")
    parser.add_argument("--mock", action="store_true",
                        help="use a pretend glove, so no Raspberry Pi hardware is needed")
    parser.add_argument("--text", action="store_true",
                        help="with --mock: draw the touchscreen in the terminal instead of a window")
    parser.add_argument("--no-touchscreen", action="store_true",
                        help="skip the home screen: straight into free practice (as USE_TOUCHSCREEN = False)")
    parser.add_argument("--touch-test", action="store_true",
                        help="on the Pi: show where the panel says you touch, to set TOUCH_X/Y_MIN/MAX")
    parser.add_argument("--stuck-finger", choices=settings.FINGERS,
                        help="with --mock: this finger's sensor is loose, so calibration fails")
    parser.add_argument("--unplugged-finger", choices=settings.FINGERS,
                        help="with --mock: this finger's sensor is unplugged (SENSOR FAULT)")
    args = parser.parse_args()
    if (args.stuck_finger or args.unplugged_finger or args.text) and not args.mock:
        parser.error("--stuck-finger, --unplugged-finger and --text only work with --mock")
    return args


if __name__ == "__main__":
    main()
