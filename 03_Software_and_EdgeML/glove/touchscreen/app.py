"""
The touchscreen's home screen and what its three buttons do.

  Start-up      sensor check, load the saved calibration, READY. No calibrating: the
                latest calibration is kept until Recalibrate is tapped.
  Home          [Free] [Lesson] [Recalibrate], with how many letters the glove can read
                (with no calibration saved yet, Free asks for one first)
  Free          main.free_practice(), drawn as the free-mode screen: one big letter,
                with [Menu] and [Recal] buttons
  Lesson        the lesson list (lesson_list()): [Back] and one button per lesson in
                settings.LESSONS, with arrows if there are more than 3. Tapping one spells it
                one letter at a time (lesson_mode()), with [Menu] and [Skip], then Complete!
                with [Menu] and [Again]. Menu goes back to the list.
  Recalibrate   flat hand, fist, and with the gyroscope fingers up and thumb up
                (calibration.run_full_calibration()), with a progress bar

main.py's own functions do the work; this file draws them on the touchscreen
and decides what each tap does.
"""

import time

import settings
import sign_feedback
from alphabet import MOTION_LETTERS
from calibration import apply_saved_imu_calibration, load_calibration, run_full_calibration
from lesson import LessonManager
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

    def __init__(self, sampler, lcd, touch, cal_file, imu=None, sleep=time.sleep, clock=time.monotonic):
        self.sampler, self.lcd, self.touch, self.cal_file, self.imu = sampler, lcd, touch, cal_file, imu
        self.sleep, self.clock = sleep, clock                   # the tests swap in a fake clock
        self.calibration = None

    def start(self):
        """Straight to the home screen: check the sensors and load the saved calibration.

        Returns True when ready, False if a sensor is faulty (the screen says which).
        """
        faulty = find_faulty_finger(self.sampler.newest())     # "All five sensors in range?"
        if faulty:
            self.lcd.message("SENSOR FAULT", f"Check {faulty}")
            return False
        self.calibration = load_calibration(self.cal_file)     # None until someone calibrates
        if self.calibration is not None:
            apply_saved_imu_calibration(self.imu, self.cal_file)
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
        """One pass: draw the home screen, then act on a tap if there was one."""
        self.lcd.show(screens.home(letter_count(self.imu)))
        button = self.tapped_button()
        if button == "free":
            self.free_mode()
        elif button == "lesson":
            self.lesson_list()
        elif button == "recal":
            self.recalibrate()

    def tapped_button(self):
        """The name of the button just tapped on the screen shown now, or None."""
        tap = self.touch.get_tap()
        if not tap:
            return None
        button = self.lcd.current.button_at(*tap) if self.lcd.current else None
        print(f"[TOUCH] tap at {tap[0]}, {tap[1]}" + (f" -> {button}" if button else ""))
        return button

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

    def lesson_list(self):
        """The list of lessons until Back is tapped. Tapping a lesson runs it, then comes back here."""
        names = tuple(settings.LESSONS)
        top = 0                                                 # the first lesson shown
        while True:
            self.lcd.show(screens.lessons(names, top))
            button = self.tapped_button()
            if button == "back":
                return
            if button == "up":
                top -= screens.ROWS
            elif button == "down":
                top += screens.ROWS
            elif button is not None:
                chosen = next((name for name in names if name.lower() == button), None)
                if chosen is not None:
                    self.lesson_mode(chosen)
            self.sleep(1 / settings.LOOP_HZ)

    def lesson_mode(self, name=None):
        """Spell a lesson's word (settings.LESSONS) until Menu is tapped. Again starts it over."""
        if self.calibration is None:                            # nothing saved yet: calibrate first
            self.recalibrate()
            if self.calibration is None:
                return
        word = settings.LESSONS[name or settings.DEFAULT_LESSON]
        while self.spell(word) and self.lesson_complete(word):
            pass

    def spell(self, word):
        """Sign each letter of the word in turn. Returns True when it's spelled, False on Menu.

        A letter counts once it is CORRECT for HOLD_SECONDS (sign_feedback.HoldTimer).
        The hold timer is never reset between letters, so for a double letter (the L L in
        HELLO) the hand has to relax and make the letter again: still holding the first L
        doesn't count as the second.
        """
        lesson = LessonManager(word)
        lesson.select_next()                                    # the first letter
        hold_timer, fault_guard = sign_feedback.HoldTimer(), sign_feedback.FaultGuard()
        flat, fist = self.calibration
        while lesson.target is not None:
            now = self.clock()
            raw = self.sampler.newest()
            faulty = fault_guard.update(raw, now)
            if faulty:                                          # never grade a disconnected sensor
                hold_timer.reset()
                self.lcd.message("SENSOR FAULT", f"Check {faulty}")
            else:
                bends = sign_feedback.normalize(raw, flat, fist)
                result = sign_feedback.evaluate(bends, lesson.target)   # CORRECT, ADJUST: finger, TRY AGAIN
                if hold_timer.update(result == "CORRECT", now):         # held long enough: it counts
                    self.lcd.show(screens.lesson(word, lesson.index, "CORRECT!"))
                    self.sleep(settings.SUCCESS_SECONDS)                # leave CORRECT! up long enough to read
                    lesson.report_success()                             # on to the next letter
                    continue
                if result == "CORRECT":
                    result = f"Relax, then {lesson.target} again" if hold_timer.fired else "HOLD STEADY"
                self.lcd.show(screens.lesson(word, lesson.index, result))
            button = self.tapped_button()
            if button == "menu":
                return False
            if button == "skip":
                lesson.select_next()
            self.sleep(1 / settings.LOOP_HZ)
        return True

    def lesson_complete(self, word):
        """The Complete! screen. Returns True for Again, False for Menu."""
        while True:
            self.lcd.show(screens.lesson_done(word))
            button = self.tapped_button()
            if button in ("again", "menu"):
                return button == "again"
            self.sleep(1 / settings.LOOP_HZ)

    def recalibrate(self):
        """All the calibration steps (calibration.py). If the fingers fail, the old calibration is kept."""
        if hasattr(self.lcd, "calibration_steps"):
            self.lcd.calibration_steps = 4 if self.imu is not None else 2   # "Step 1 of 4" with the gyroscope
        new = run_full_calibration(self.sampler, self.lcd, self.cal_file, self.imu)
        if new is not None:
            self.calibration = new
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
