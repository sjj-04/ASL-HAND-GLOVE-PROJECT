"""Tests for the touchscreen: the screens, the LCD driver, the touch panel and the home screen's buttons."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parents[1]))   # lets the tests import the files in glove/

import settings
import hardware
import main as glove_main
import calibration as calibration_module
import mock_glove
from mock_glove import MockMCP3208
from touchscreen import app as home
from touchscreen import panel as touch_panel
from touchscreen import screens
from touchscreen.lcd import TouchLcd
from touchscreen.sim import SimHand

ts = settings                            # the TOUCH_ numbers live in settings.py
routine = settings.routine
TEST_REFERENCES = {"A": [0.1, 1.0, 1.0, 1.0, 1.0], "B": [0.9, 0.0, 0.0, 0.0, 0.0],
                   "C": [0.4, 0.5, 0.5, 0.5, 0.5]}


def all_screens():
    return [screens.home(26), screens.home(1), screens.free("B"), screens.free("U/V/R/H"), screens.free(None),
            screens.lesson("HELLO", 2, "ADJUST: Thumb"), screens.lesson("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 20, "TRY AGAIN"),
            screens.lesson_done("HELLO"), screens.lessons(("Hello", "Alphabet", "Zion")),
            screens.lessons(("Hello", "Alphabet", "Zion", "Thank you", "Name"), 3),
            screens.calibrating("Hold hand FLAT", 1, 0.5),
            screens.message("Calibration", "saved"), screens.touch_test((10, 10), (500, 600))]


class ScreensTest(unittest.TestCase):

    def test_every_screen_is_128x64_black_and_white(self):
        for screen in all_screens():
            self.assertEqual(screen.image.size, (128, 64), screen.summary)
            self.assertEqual(screen.image.mode, "1", screen.summary)
            self.assertTrue(np.array(screen.image).any(), screen.summary)

    def test_buttons_are_on_the_screen_and_big_enough_to_hit(self):
        for screen in all_screens():
            for name, (x0, y0, x1, y1) in screen.buttons.items():
                self.assertTrue(0 <= x0 < x1 <= 127 and 0 <= y0 < y1 <= 63, name)
                self.assertGreaterEqual(min(x1 - x0, y1 - y0), 14, f"{name} is too small for a finger")

    def test_home_layout(self):
        home_screen = screens.home(26)
        self.assertEqual(home_screen.button_at(30, 28), "free")        # top left
        self.assertEqual(home_screen.button_at(95, 28), "lesson")      # top right
        self.assertEqual(home_screen.button_at(64, 55), "recal")       # across the bottom
        self.assertIsNone(home_screen.button_at(64, 5))                # the title bar isn't a button
        self.assertIn("26 letters", home_screen.summary)
        self.assertIn("1 letter)", screens.home(1).summary)

    def test_long_text_is_shrunk_to_fit(self):
        picture = np.array(screens.message("Calibration", "a very long message for this tiny screen").image)
        self.assertFalse(picture[14:, 0].any() or picture[14:, 127].any())


class LetterCountTest(unittest.TestCase):

    def test_counts_the_references_when_the_model_is_off(self):
        with mock.patch.object(settings, "USE_MODEL", False), \
             mock.patch.object(settings, "REFERENCES", TEST_REFERENCES), \
             mock.patch.object(settings, "ACTIVE_LETTERS", None):
            self.assertEqual(home.letter_count(), 3)

    def test_counts_the_model_letters_when_it_is_on(self):
        model = mock.Mock()
        model.letters.return_value = [chr(c) for c in range(ord("A"), ord("Z") + 1)]   # J and Z too, later
        model.n_inputs = 11                                 # trained with the gyroscope
        with mock.patch.object(home.sign_feedback, "load_model", return_value=model), \
             mock.patch.object(settings, "REFERENCES", TEST_REFERENCES), \
             mock.patch.object(settings, "ACTIVE_LETTERS", None):
            self.assertEqual(home.letter_count(imu=object()), 26)
            self.assertEqual(home.letter_count(imu=None), 3)  # its inputs are missing: REFERENCES, no J or Z

    def test_j_and_z_need_the_gyroscope(self):
        refs = dict(TEST_REFERENCES, J=[0.8, 1.0, 1.0, 1.0, 0.0], Z=[0.7, 0.0, 1.0, 1.0, 1.0])
        with mock.patch.object(settings, "USE_MODEL", False), mock.patch.object(settings, "REFERENCES", refs), \
             mock.patch.object(settings, "ACTIVE_LETTERS", None):
            self.assertEqual(home.letter_count(imu=None), 3)
            self.assertEqual(home.letter_count(imu=object()), 5)

    def test_active_letters_limit_the_count(self):
        with mock.patch.object(settings, "USE_MODEL", False), \
             mock.patch.object(settings, "REFERENCES", TEST_REFERENCES), \
             mock.patch.object(settings, "ACTIVE_LETTERS", ["A", "B", "Z"]):
            self.assertEqual(home.letter_count(), 2)       # Z isn't recognized, so it doesn't count


class LcdTest(unittest.TestCase):
    """The glove's real WristDisplay, writing to mock_glove's pretend KS0108."""

    def setUp(self):
        self.bus = mock_glove.FakeGlcdBus()
        for name, value in [("lgpio", self.bus), ("Image", Image), ("ImageDraw", ImageDraw),
                            ("ImageFont", ImageFont)]:
            patcher = mock.patch.object(routine, name, value, create=True)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.lcd = TouchLcd(hardware.WristDisplay(), routine)

    def test_the_lcd_shows_exactly_the_screen(self):
        for screen in all_screens():
            self.lcd.show(screen)
            np.testing.assert_array_equal(self.bus.pixels(), np.array(screen.image, dtype=bool))

    def test_only_changed_bytes_are_sent(self):
        self.lcd.show(screens.home(26))
        self.bus.log.clear()
        self.lcd.show(screens.home(26))
        self.assertEqual(self.bus.bytes_written(), 0)
        self.lcd.show(screens.calibrating("Hold hand FLAT", 1, 0.2))
        self.bus.log.clear()
        self.lcd.show(screens.calibrating("Hold hand FLAT", 1, 0.3))
        self.assertLess(self.bus.bytes_written(), 40)


class FakePanel:
    """Pretend lgpio + MCP3208 wired to a 4-wire resistive panel, pressed at (fx, fy) (0.0-1.0) or not."""

    SET_PULL_UP = 32

    def __init__(self):
        self.mode = {}
        self.press = None

    def gpiochip_open(self, chip):
        return 7

    def gpiochip_close(self, handle):
        pass

    def gpio_claim_input(self, handle, gpio, lFlags=0):
        assert gpio not in self.mode, f"GPIO {gpio} claimed twice"
        self.mode[gpio] = "pull-up" if lFlags & self.SET_PULL_UP else "in"

    def gpio_claim_output(self, handle, gpio, level=0, lFlags=0):
        assert gpio not in self.mode, f"GPIO {gpio} claimed twice"
        self.mode[gpio] = level

    def gpio_free(self, handle, gpio):
        del self.mode[gpio]

    def gpio_read(self, handle, gpio):
        assert self.mode[gpio] == "pull-up"
        return 0 if self.press is not None and self.mode[ts.TOUCH_YM_PIN] == 0 else 1

    def read(self, channel):
        m = self.mode
        if self.press is None:
            return 1234
        fx, fy = self.press
        volts = None
        if channel == ts.TOUCH_YP_CHANNEL and m[ts.TOUCH_XP_PIN] == 1 and m[ts.TOUCH_XM_PIN] == 0:
            assert m[ts.TOUCH_YP_PIN] == "in" and m[ts.TOUCH_YM_PIN] == "in"
            volts = 3.3 * fx
        if channel == ts.TOUCH_XP_CHANNEL and m[ts.TOUCH_YP_PIN] == 1 and m[ts.TOUCH_YM_PIN] == 0:
            assert m[ts.TOUCH_XP_PIN] == "in" and m[ts.TOUCH_XM_PIN] == "in"
            volts = 3.3 * fy
        assert volts is not None, f"read CH{channel} with the panel wired wrong: {m}"
        return round(volts / 5.0 * 4095)


class TouchPanelTest(unittest.TestCase):

    def setUp(self):
        self.fake = FakePanel()
        self.panel = touch_panel.ResistiveTouch(self.fake, self.fake)

    def test_not_touched(self):
        self.assertIsNone(self.panel.read())

    def test_reads_where_it_is_pressed(self):
        full = round(3.3 / 5.0 * 4095)
        with mock.patch.multiple(ts, TOUCH_X_MIN=0, TOUCH_X_MAX=full, TOUCH_Y_MIN=0, TOUCH_Y_MAX=full):
            for fx, fy in [(0.1, 0.2), (0.5, 0.5), (0.9, 0.8)]:
                self.fake.press = (fx, fy)
                x, y = touch_panel.to_pixel(*self.panel.read())
                self.assertAlmostEqual(x, fx * 127, delta=1)
                self.assertAlmostEqual(y, fy * 63, delta=1)

    def test_mirrored_and_swapped_axes(self):
        with mock.patch.multiple(ts, TOUCH_X_MIN=2000, TOUCH_X_MAX=0, TOUCH_Y_MIN=0, TOUCH_Y_MAX=1000):
            self.assertEqual(touch_panel.to_pixel(2000, 0), (0, 0))
            self.assertEqual(touch_panel.to_pixel(-50, 5000), (127, 63))
        with mock.patch.multiple(ts, TOUCH_SWAP_XY=True, TOUCH_X_MIN=0, TOUCH_X_MAX=1000,
                                 TOUCH_Y_MIN=0, TOUCH_Y_MAX=1000):
            self.assertEqual(touch_panel.to_pixel(0, 1000), (127, 0))

    def test_one_tap_per_press(self):
        d = touch_panel.TapDetector()
        taps = [d.update(p) for p in [None, (50, 20), (51, 21), (50, 20), None, None, (10, 10), (10, 10)]]
        self.assertEqual([t for t in taps if t], [(50, 20), (10, 10)])


class FakeTime:
    """Stands in for the time module inside main.py, so free practice runs instantly."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        if self.now > 1200:
            raise RuntimeError("the test ran too long: a tap never arrived?")


class InstantSampler:
    def __init__(self, adc):
        self.adc = adc

    def _read(self):
        return np.array([self.adc.read(c) for c in range(len(settings.FINGERS))], dtype=float)

    def average(self):
        return np.mean([self._read() for _ in range(20)], axis=0)

    def newest(self):
        return np.median([self._read() for _ in range(5)], axis=0)


class TimedTouch:
    """Taps a button (by name) when the fake clock reaches a time. Can also run a step, like changing the hand."""

    def __init__(self, clock, lcd):
        self.clock, self.lcd, self.events = clock, lcd, []

    def at(self, seconds, action):
        self.events.append((self.clock.now + seconds, action))
        self.events.sort(key=lambda event: event[0])

    def get_tap(self):
        while self.events and self.events[0][0] <= self.clock.now:
            _, action = self.events.pop(0)
            if callable(action):
                action()
            else:
                x0, y0, x1, y1 = self.lcd.current.buttons[action]
                return (x0 + x1) // 2, (y0 + y1) // 2
        return None

    def clear(self):
        pass


class HomeAppTest(unittest.TestCase):
    """Taps and hand poses in, screens out, with the glove's own start_up() and free_practice()."""

    def setUp(self):
        self.clock = FakeTime()
        for target, name, value in [(routine, "POSE_SETTLE_SECONDS", 0), (routine, "AVERAGE_SECONDS", 0),
                                    (routine, "MESSAGE_SECONDS", 0), (routine, "time", self.clock),
                                    (glove_main, "time", self.clock), (settings, "USE_MODEL", False),
                                    (settings, "REFERENCES", TEST_REFERENCES), (settings, "ACTIVE_LETTERS", None)]:
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.make_app()

    imu = None                                             # see HomeAppWithGyroscopeTest

    def make_app(self, **hand_args):
        self.hand = SimHand(**hand_args)
        self.lcd = TouchLcd(None, routine, on_message=self.hand.follow)
        self.touch = TimedTouch(self.clock, self.lcd)
        self.shown = []
        push = self.lcd._push                              # every screen that reaches the LCD, in order
        self.lcd._push = lambda screen: (self.shown.append(screen.summary), push(screen))
        self.app = home.HomeApp(InstantSampler(MockMCP3208(self.hand)), self.lcd, self.touch,
                                Path(self.folder.name) / "cal.json", imu=self.imu, sleep=self.clock.sleep,
                                clock=self.clock.monotonic)

    def screen(self):
        return self.lcd.current.summary

    def tap(self, name):
        self.touch.at(0, name)
        self.app.step()

    def test_start_up_goes_straight_home_without_calibrating(self):
        self.assertTrue(self.app.start())
        self.app.step()
        self.assertEqual(self.screen(), "Home (3 letters): [Free] [Lesson] [Recalibrate]")   # no gyroscope
        self.assertFalse(any(s.startswith("CALIBRATING") for s in self.shown))
        self.assertIsNone(self.app.calibration)                 # nothing saved yet

    def test_free_asks_for_a_calibration_only_the_first_time(self):
        self.app.start()
        self.touch.at(0.1, "menu")
        self.tap("free")
        self.assertTrue(any(x.startswith("CALIBRATING  Hold hand FLAT") for x in self.shown))
        self.assertIsNotNone(self.app.calibration)
        self.make_app()                                         # the glove is switched off and on again
        self.app.start()
        self.assertIsNotNone(self.app.calibration)              # the saved one, no calibrating
        self.touch.at(0.1, "menu")
        self.tap("free")
        self.assertFalse(any(s.startswith("CALIBRATING") for s in self.shown))

    def test_unplugged_sensor_stops_at_start_up(self):
        self.make_app(unplugged_finger="Ring")
        self.assertFalse(self.app.start())
        self.assertEqual(self.screen(), "SENSOR FAULT  Check Ring")

    def test_free_mode_shows_held_letters_then_menu_goes_home(self):
        self.app.start()
        self.tap("recal")                                   # calibrate first (calibration waits for the hand)
        self.touch.at(0.5, lambda: self.hand.choose("B"))
        self.touch.at(2.0, lambda: self.hand.choose("C"))
        self.touch.at(3.5, "menu")
        self.tap("free")                                   # runs free_practice() until Menu at 3.5 s
        self.assertIn("Free mode: B  [Menu] [Recal]", self.shown)
        self.assertIn("Free mode: C  [Menu] [Recal]", self.shown)
        self.app.step()
        self.assertTrue(self.screen().startswith("Home"))

    def test_recal_from_free_mode_goes_back_to_free_mode(self):
        self.app.start()
        self.tap("recal")
        old = self.app.calibration
        self.touch.at(0.3, "recal")
        self.touch.at(0.6, lambda: self.hand.choose("A"))      # runs once free mode is back
        self.touch.at(20.0, "menu")                            # well after the recalibration
        self.tap("free")
        recal_at = next(i for i, x in enumerate(self.shown) if x.startswith("CALIBRATING  Hold hand FLAT"))
        self.assertIn("Free mode: A  [Menu] [Recal]", self.shown[recal_at:])
        self.assertIsNot(self.app.calibration, old)

    def use_lesson(self, word):
        """Run this word as the lesson (the test REFERENCES only have A, B and C)."""
        for name, value in [("LESSONS", {"Test": word}), ("DEFAULT_LESSON", "Test")]:
            patcher = mock.patch.object(settings, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_lesson_spells_the_word_and_a_double_letter_needs_signing_twice(self):
        self.use_lesson("ABBC")
        self.app.start()
        self.tap("recal")                                      # calibrate first
        self.touch.at(0.5, lambda: self.hand.choose("A"))
        self.touch.at(3.0, lambda: self.hand.choose("B"))      # first B, then keep holding it
        self.touch.at(8.0, lambda: self.hand.choose("fist"))   # relax...
        self.touch.at(9.0, lambda: self.hand.choose("B"))      # ...and make B again: the second B
        self.touch.at(12.0, lambda: self.hand.choose("C"))
        self.touch.at(16.0, "menu")                            # on the Complete! screen
        self.touch.at(16.5, "back")                            # the lesson list
        self.touch.at(0.01, "test")                            # pick the lesson from the list
        self.tap("lesson")
        self.assertIn("Lesson ABBC, letter 1 (A): CORRECT!  [Menu] [Skip]", self.shown)
        self.assertIn("Lesson ABBC, letter 2 (B): CORRECT!  [Menu] [Skip]", self.shown)
        self.assertIn("Lesson ABBC, letter 3 (B): Relax, then B again  [Menu] [Skip]", self.shown)
        self.assertIn("Lesson ABBC, letter 4 (C): CORRECT!  [Menu] [Skip]", self.shown)
        second_b = self.shown.index("Lesson ABBC, letter 3 (B): CORRECT!  [Menu] [Skip]")
        relaxed = next(i for i, x in enumerate(self.shown) if x.startswith("Lesson ABBC, letter 3 (B): TRY AGAIN"))
        self.assertLess(relaxed, second_b)                     # the held B only counted after relaxing
        self.assertIn("Lesson ABBC: complete  [Menu] [Again]", self.shown)
        self.app.step()
        self.assertTrue(self.screen().startswith("Home"))

    def test_lesson_skip_menu_and_again(self):
        self.use_lesson("AB")
        self.app.start()
        self.tap("recal")
        self.touch.at(0.1, "skip")                             # skip A
        self.touch.at(0.2, lambda: self.hand.choose("B"))
        self.touch.at(4.0, "again")                            # Complete!, then start over
        self.touch.at(4.5, "menu")
        self.touch.at(5.0, "back")
        self.touch.at(0.01, "test")
        self.tap("lesson")
        self.assertIn("Lesson AB, letter 2 (B): CORRECT!  [Menu] [Skip]", self.shown)
        self.assertNotIn("Lesson AB, letter 1 (A): CORRECT!  [Menu] [Skip]", self.shown)
        screens_in_order = [x for i, x in enumerate(self.shown) if i == 0 or x != self.shown[i - 1]]
        self.assertEqual(screens_in_order.count("Lesson AB: complete  [Menu] [Again]"), 1)
        menu_at = max(i for i, x in enumerate(self.shown) if x.startswith("Lesson AB, letter"))
        self.assertTrue(self.shown[menu_at].startswith("Lesson AB, letter 1 (A)"))   # Again went back to A
        self.assertTrue(self.shown[menu_at + 1].startswith("Lessons: [Back]"))      # Menu: the list
        self.app.step()
        self.assertTrue(self.screen().startswith("Home"))

    def test_lesson_list_shows_every_lesson_and_scrolls(self):
        names = {"One": "A", "Two": "B", "Three": "C", "Four": "A", "Five": "B"}
        patcher = mock.patch.object(settings, "LESSONS", names)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.app.start()
        self.touch.at(0.1, "down")
        self.touch.at(0.2, "up")
        self.touch.at(0.3, "back")
        self.tap("lesson")
        first = "Lessons: [Back] [One] [Two] [Three] [up] [down]"
        self.assertIn(first, self.shown)
        self.assertIn("Lessons: [Back] [Four] [Five] [up] [down]", self.shown)
        self.assertEqual(self.shown.count(first) > 0 and self.shown[-1], first)   # up went back to the top
        self.app.step()
        self.assertTrue(self.screen().startswith("Home"))

    def test_lesson_list_arrows_only_scroll_where_there_is_more(self):
        page_one = screens.lessons(("A1", "A2", "A3", "A4"), 0)
        self.assertIn("down", page_one.buttons)
        self.assertNotIn("up", page_one.buttons)             # already at the top
        self.assertNotIn("up", screens.lessons(("A1", "A2", "A3"), 0).buttons)   # 3 fit: no arrows
        self.assertEqual(page_one.button_at(50, 24), "a1")

    def test_recalibrate_from_home(self):
        self.app.start()
        old = self.app.calibration
        self.tap("recal")
        self.assertIsNot(self.app.calibration, old)
        self.app.step()
        self.assertTrue(self.screen().startswith("Home"))

    def test_failed_recalibration_keeps_the_old_one(self):
        self.app.start()
        self.tap("recal")                                      # a good calibration first
        old = self.app.calibration
        self.hand.stuck = settings.FINGERS.index("Pinky")  # the sensor comes loose later
        self.tap("recal")
        self.assertEqual(self.screen(), "FAULT  Check wiring")
        self.assertIs(self.app.calibration, old)


if __name__ == "__main__":
    unittest.main()


class FakeImu:
    """The gyroscope's state(), set by the test: still unless the pretend hand is tracing."""

    def __init__(self, hand):
        self.hand = hand

    def state(self):
        return np.array(self.hand.up, dtype=float), np.array(self.hand.motion, dtype=float)

    def recent(self, seconds):
        """Calibration's raw readings: 20 still readings of the hand's current pose."""
        accel, _ = self.hand.imu_reading(0.0)
        return np.tile(accel, (20, 1)), np.zeros((20, 3))

    def set_orientation(self, rotation, gyro_offset=None):
        self.rotation = rotation

    def zero_gyro(self):
        pass


class HomeAppWithGyroscopeTest(HomeAppTest):
    """The same taps with a gyroscope: J is read while traced, and I while held still."""

    def make_app(self, **hand_args):
        self.imu = None
        super().make_app(**hand_args)
        self.imu = self.app.imu = FakeImu(self.hand)

    def setUp(self):
        super().setUp()
        refs = dict(TEST_REFERENCES, I=[0.8, 1.0, 1.0, 1.0, 0.0], J=[0.8, 1.0, 1.0, 1.0, 0.0])
        patcher = mock.patch.object(settings, "REFERENCES", refs)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_start_up_goes_straight_home_without_calibrating(self):
        self.assertTrue(self.app.start())
        self.app.step()
        self.assertEqual(self.screen(), "Home (5 letters): [Free] [Lesson] [Recalibrate]")   # A B C I J

    def test_recalibrate_has_4_steps_with_the_gyroscope(self):
        self.app.start()
        self.tap("recal")
        self.assertIn("CALIBRATING  Point fingers UP  (step 3 of 4)", self.shown)
        self.assertIn("CALIBRATING  Thumb UP  (step 4 of 4)", self.shown)
        self.assertIsNotNone(calibration_module.load_imu_calibration(self.app.cal_file))

    def test_traced_j_and_still_i(self):
        self.app.start()
        self.tap("recal")
        self.touch.at(0.2, lambda: self.hand.choose("I"))
        self.touch.at(2.0, lambda: self.hand.choose("J"))         # SimHand traces J: the gyroscope moves
        self.touch.at(3.0, "menu")
        self.tap("free")
        frees = [s for s in self.shown if s.startswith("Free mode")]
        self.assertIn("Free mode: I  [Menu] [Recal]", frees)
        self.assertIn("Free mode: J  [Menu] [Recal]", frees)
        self.assertLess(frees.index("Free mode: I  [Menu] [Recal]"), frees.index("Free mode: J  [Menu] [Recal]"))
