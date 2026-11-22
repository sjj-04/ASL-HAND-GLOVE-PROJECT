"""
The touchscreen's home screen and what its three buttons do.

  Start-up      sensor check, load the saved calibration, READY. No calibrating: the
                latest calibration is kept until Recalibrate is tapped.
  Home          [Free] [Lesson] [Recalibrate], with how many letters the glove can read
                (with no calibration saved yet, Free asks for one first)
  Free          main.free_practice(), drawn as the free-mode screen: one big letter,
                with [Menu] and [Recal] buttons
  Lesson        "Coming soon" and [Back], until lessons are built
  Recalibrate   run_calibration() with a progress bar

main.py's own functions do the work; this file draws them on the touchscreen
and decides what each tap does.
"""

import time

import settings
import sign_feedback
from alphabet import MOTION_LETTERS
from calibration import load_calibration, run_calibration
from startup_check import find_faulty_finger

from . import screens
from .lcd import TouchLcd
from .panel import MockTouch, ResistiveTouch, TouchInput, to_pixel


def letter_count(imu=None):
    """How many letters the glove can read right now.

    The trained model's letters if USE_MODEL is on, otherwise the letters in
    REFERENCES. J and Z only count with a working gyroscope (and, for the model,
    if it was trained with one). If ACTIVE_LETTERS limits what is shown, only those count.
    """
    model = sign_feedback.load_model()
    if model is not None and (imu is not None or model.n_inputs <= len(settings.FINGERS)):
        known = set(model.letters())
    else:
        known = set(settings.REFERENCES)
    if imu is None:
        known -= set(MOTION_LETTERS)
    active = settings.ACTIVE_LETTERS
    return len(known & set(active)) if active else len(known)


class LeaveFreeMode(Exception):
    """Raised from inside free_practice() when Menu or Recal is tapped."""

    def __init__(self, button):
        super().__init__(button)
        self.button = button


class FreeModeDisplay:
    """What free_practice() draws on. It calls message("Free practice", "SIGN: B")
    20 times a second, so this is also where the Menu and Recal taps are checked."""

    def __init__(self, lcd, touch):
        self.lcd, self.touch = lcd, touch

    def message(self, title, detail=""):
        if title == "Free practice":
            letter = detail.split("SIGN:", 1)[-1].strip()
            self.lcd.show(screens.free(None if letter == "--" else letter))
        else:
            self.lcd.message(title, detail)                    # SENSOR FAULT and so on
        tap = self.touch.get_tap()
        button = self.lcd.current.button_at(*tap) if tap and self.lcd.current else None
        if tap:
            print(f"[TOUCH] tap at {tap[0]}, {tap[1]}" + (f" -> {button}" if button else ""))
        if button in ("menu", "recal"):
            raise LeaveFreeMode(button)


class HomeApp:

    def __init__(self, sampler, lcd, touch, cal_file, imu=None, sleep=time.sleep):
        self.sampler, self.lcd, self.touch, self.cal_file, self.imu = sampler, lcd, touch, cal_file, imu
        self.sleep = sleep
        self.calibration = None
        self.screen = "home"                                    # home or lesson

    def start(self):
        """Straight to the home screen: check the sensors and load the saved calibration.

        Returns True when ready, False if a sensor is faulty (the screen says which).
        """
        faulty = find_faulty_finger(self.sampler.newest())     # "All five sensors in range?"
        if faulty:
            self.lcd.message("SENSOR FAULT", f"Check {faulty}")
            return False
        self.calibration = load_calibration(self.cal_file)     # None until someone calibrates
        print("[CAL] Using the saved calibration." if self.calibration is not None
              else "[CAL] No calibration saved yet: tapping Free will ask for one.")
        if self.imu is not None:
            self.imu.zero_gyro()
        self.lcd.message("READY")
        self.sleep(settings.MESSAGE_SECONDS)
        self.touch.clear()
        return True

    def run(self):
        """Until Ctrl+C."""
        while True:
            self.step()
            self.sleep(1 / settings.LOOP_HZ)

    def step(self):
        """One pass: draw the screen, then act on a tap if there was one."""
        if self.screen == "home":
            self.lcd.show(screens.home(letter_count(self.imu)))
        else:
            self.lcd.show(screens.coming_soon())
        tap = self.touch.get_tap()
        if not tap:
            return
        button = self.lcd.current.button_at(*tap)
        print(f"[TOUCH] tap at {tap[0]}, {tap[1]}" + (f" -> {button}" if button else ""))
        if button == "free":
            self.free_mode()
        elif button == "lesson":
            self.screen = "lesson"
        elif button == "back":
            self.screen = "home"
        elif button == "recal":
            self.recalibrate()

    def free_mode(self):
        """main.free_practice() until Menu is tapped. Recal recalibrates and carries on."""
        import main as glove_main
        if self.calibration is None:                            # nothing saved yet: calibrate first
            self.recalibrate()
            if self.calibration is None:
                return
        while True:
            try:
                glove_main.free_practice(self.sampler, FreeModeDisplay(self.lcd, self.touch),
                                         self.calibration, self.imu)
            except LeaveFreeMode as leave:
                if leave.button == "menu":
                    return
                self.recalibrate()

    def recalibrate(self):
        """The glove's run_calibration(). If it fails, the old calibration is kept."""
        new = run_calibration(self.sampler, self.lcd, self.cal_file)
        if new is not None:
            self.calibration = new
            if self.imu is not None:
                self.imu.zero_gyro()
        else:
            self.sleep(settings.MESSAGE_SECONDS)                # leave FAULT up long enough to read
        self.touch.clear()


def open_touch(adc):
    """The real touch panel, or None if its pins can't be opened."""
    try:
        return ResistiveTouch(adc, settings.routine.lgpio)
    except Exception as error:
        print(f"[TOUCH] Touch panel not available ({error}).")
        return None


def run_home(sampler, display, adc, imu, cal_file):
    """main.py on the Pi: the home screen on the Winstar, with the real touch panel."""
    import main as glove_main
    lcd = TouchLcd(display, settings.routine)
    panel = open_touch(adc)
    if panel is None:                                           # no way to tap: go straight to free practice
        print("[TOUCH] Without touch the home screen can't be used, so free practice starts directly.")
        calibration = glove_main.start_up(sampler, lcd, cal_file, imu)
        if calibration is None:
            raise SystemExit("Stopped at start-up. The screen shows why.")
        return glove_main.free_practice(sampler, FreeModeDisplay(lcd, MockTouch()), calibration, imu)
    try:
        app = HomeApp(sampler, lcd, TouchInput(panel), cal_file, imu)
        if not app.start():
            raise SystemExit("Stopped at start-up. The screen shows why.")
        app.run()
    finally:
        lcd.stop_animation()
        panel.close()


def touch_test():
    """main.py --touch-test: a cross where the panel says you touched, and the raw numbers.
    Touch all four corners, then Ctrl+C for the numbers to put in settings.py."""
    from hardware import PI_LIBRARIES_FOUND, open_hardware
    if not PI_LIBRARIES_FOUND:
        raise SystemExit("--touch-test runs on the Pi, with the touch panel wired.")
    adc, display = open_hardware()
    lcd = TouchLcd(display, settings.routine)
    panel = open_touch(adc)
    if panel is None:
        raise SystemExit("The touch panel's pins couldn't be opened. Is main.py still running?")
    print("Touch the screen, including all four corners. Ctrl+C when done.")
    xs, ys = [], []
    lcd.show(screens.touch_test(None, None))
    try:
        while True:
            raw = panel.read()
            if raw:
                xs.append(raw[0])
                ys.append(raw[1])
                lcd.show(screens.touch_test(to_pixel(*raw), raw))
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        panel.close()
        adc.close()
    if not xs:
        print("\nNo touches seen. Check the four panel wires, and that X+ and Y+ reach CH5 and CH6.")
        return
    print("\nPut these in settings.py:")
    print(f"TOUCH_X_MIN, TOUCH_X_MAX = {min(xs)}, {max(xs)}")
    print(f"TOUCH_Y_MIN, TOUCH_Y_MAX = {min(ys)}, {max(ys)}")
    print("Then run --touch-test again. If the cross moves the wrong way, swap that line's two numbers.\n"
          "If it moves up and down when you move sideways, set TOUCH_SWAP_XY = True.")


__all__ = ["HomeApp", "FreeModeDisplay", "LeaveFreeMode", "MockTouch", "letter_count", "run_home", "touch_test"]
