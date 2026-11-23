"""Tests for calibration steps 3 and 4: the gyroscope's hand directions and still reading."""

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))   # lets the tests import the files in glove/

import settings
import calibration
import imu
from main import start_up
from mock_glove import FIST_READINGS, FLAT_READINGS, InstantSampler, MockIMU, MockMCP3208, MockOLED, PretendHand

routine = settings.routine
# The board turned on the wrist: its x points out of the back of the hand, its y along the
# fingers, its z toward the thumb (rows: glove axes -> board axes).
TURNED = np.array([[0.0, 0.0, 1.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0]])


def rotation_about_x(degrees):
    a = np.radians(degrees)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


class OrientationMathTests(unittest.TestCase):

    def test_lined_up_board_gives_no_rotation(self):
        np.testing.assert_allclose(imu.orientation_from([1, 0, 0], [0, 1, 0]), np.eye(3), atol=1e-9)

    def test_turned_board_is_turned_back(self):
        fingers_up, thumb_up = TURNED @ [1, 0, 0], TURNED @ [0, 1, 0]      # what the board reads
        rotation = imu.orientation_from(fingers_up, thumb_up)
        np.testing.assert_allclose(rotation @ TURNED, np.eye(3), atol=1e-9)

    def test_slightly_sloppy_poses_still_give_a_proper_rotation(self):
        rotation = imu.orientation_from([0.98, 0.15, 0.1], [0.2, 0.95, -0.1])
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0)              # right-handed: back = along x thumb

    def test_poses_pointing_the_same_way_are_refused(self):
        self.assertIsNone(imu.orientation_from([1, 0, 0], [0.9, 0.3, 0]))  # about 18 degrees apart
        self.assertIsNone(imu.orientation_from([0, 0, 0], [0, 1, 0]))      # no gravity reading


@mock.patch.multiple(routine, MESSAGE_SECONDS=0)
@mock.patch.multiple(calibration, HOLD_STILL_SECONDS=0.4, STEP_TIMEOUT_SECONDS=1.5)
class OrientationStepTests(unittest.TestCase):
    """Steps 3 and 4 in real time with the pretend gyroscope, mounted turned and with a gyro offset."""

    def setUp(self):
        self.hand = PretendHand()
        self.hand.mount, self.hand.gyro_bias = TURNED, (12.0, -7.0, 4.0)
        self.display = MockOLED(self.hand)
        self.sampler = imu.ImuSampler(MockIMU(self.hand))
        self.sampler.start()
        self.addCleanup(self.sampler.stop)

    def test_steps_3_and_4_find_the_hand_directions_and_still_reading(self):
        with mock.patch("builtins.print"):
            rotation, offset = calibration.calibrate_orientation(self.sampler, self.display)
        self.assertEqual([d for _, d in self.display.history], ["Point fingers UP", "Thumb UP"])
        np.testing.assert_allclose(rotation @ TURNED, np.eye(3), atol=0.05)
        np.testing.assert_allclose(offset, self.hand.gyro_bias, atol=1.0)

        self.sampler.set_orientation(rotation, offset)
        self.hand.up = (0.0, 1.0, 0.0)                   # thumb up, after calibration
        time.sleep(0.6)
        tilt, motion = self.sampler.state()
        np.testing.assert_allclose(tilt, [0, 1, 0], atol=0.05)   # in glove axes, whatever the mounting
        self.assertFalse(imu.is_moving(motion))          # the gyro offset is gone

    def test_moving_hand_is_asked_to_hold_still_then_fault(self):
        self.hand.motion = (0.4, 0.4, 0.4)
        original = self.hand.imu_reading

        def always_moving(now):                          # keeps turning even during calibration
            accel, gyro = original(now)
            return accel, gyro + 80 * np.sin(2 * np.pi * 3 * now + np.arange(3))

        self.hand.imu_reading = always_moving
        with mock.patch("builtins.print"):
            self.assertIsNone(calibration.calibrate_orientation(self.sampler, self.display))
        titles = [t for t, _ in self.display.history]
        self.assertEqual(titles.count("Hold still"), routine.MAX_CAL_TRIES)
        self.assertEqual(self.display.history[-1], ("FAULT", "Check gyroscope"))

    def test_thumb_up_must_point_a_different_way(self):
        self.hand.CALIBRATION_UP = {"FINGERS UP": (1.0, 0.0, 0.0), "THUMB UP": (1.0, 0.1, 0.0)}  # not turned
        with mock.patch("builtins.print"):
            self.assertIsNone(calibration.calibrate_orientation(self.sampler, self.display))
        self.assertIn("Turn hand more", [t for t, _ in self.display.history])


class SaveLoadTests(unittest.TestCase):

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.cal_file = Path(folder.name) / "calibration.json"
        calibration.save_calibration(np.array(FLAT_READINGS, float), np.array(FIST_READINGS, float), self.cal_file)

    def test_saved_next_to_the_finger_readings(self):
        rotation = rotation_about_x(30)
        calibration.save_imu_calibration(self.cal_file, rotation, np.array([1.0, 2.0, 3.0]))
        saved_rotation, saved_offset = calibration.load_imu_calibration(self.cal_file)
        np.testing.assert_allclose(saved_rotation, rotation, atol=1e-4)
        np.testing.assert_allclose(saved_offset, [1, 2, 3])
        self.assertIsNotNone(calibration.load_calibration(self.cal_file))   # the fingers are still there
        self.assertIsNotNone(routine.load_calibration(self.cal_file))       # and Calibration_Routine can read it

    def test_old_or_damaged_files_have_no_gyroscope_calibration(self):
        self.assertIsNone(calibration.load_imu_calibration(self.cal_file))  # saved before the gyroscope
        data = json.loads(self.cal_file.read_text())
        data["imu"] = {"rotation": [[2, 0, 0], [0, 1, 0], [0, 0, 1]], "gyro_offset": [0, 0, 0]}
        self.cal_file.write_text(json.dumps(data))
        self.assertIsNone(calibration.load_imu_calibration(self.cal_file))  # not a rotation


@mock.patch.multiple(routine, POSE_SETTLE_SECONDS=0, AVERAGE_SECONDS=0, MESSAGE_SECONDS=0)
class FullCalibrationTests(unittest.TestCase):
    """All 4 steps through the glove's own start_up(), with a gyroscope that reads the pretend hand."""

    class InstantImu:
        """A gyroscope with no background thread: recent() reads the hand right now."""

        def __init__(self, hand):
            self.hand, self.rotation, self.gyro_offset = hand, np.eye(3), np.zeros(3)

        def recent(self, seconds):
            readings = [self.hand.imu_reading(i / 100) for i in range(30)]
            return np.array([a for a, _ in readings]), np.array([g for _, g in readings])

        def set_orientation(self, rotation, gyro_offset=None):
            self.rotation, self.gyro_offset = rotation, gyro_offset

        def zero_gyro(self):
            return False

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.cal_file = Path(folder.name) / "calibration.json"
        self.hand = PretendHand()
        self.hand.mount = TURNED
        self.imu = self.InstantImu(self.hand)

    def start(self, **kwargs):
        display = MockOLED(self.hand)
        with mock.patch("time.sleep"), mock.patch("builtins.print"):
            result = start_up(InstantSampler(MockMCP3208(self.hand)), display, self.cal_file, self.imu, **kwargs)
        return result, display.history

    def test_four_steps_then_saved(self):
        result, shown = self.start()
        self.assertIsNotNone(result)
        prompts = [d for t, d in shown if t == "CALIBRATING"]
        self.assertEqual(prompts, ["Hold hand FLAT", "Make a FIST", "Point fingers UP", "Thumb UP"])
        self.assertIn(("Fingers", "saved"), shown)
        self.assertIn(("Calibration", "saved"), shown)
        np.testing.assert_allclose(self.imu.rotation @ TURNED, np.eye(3), atol=0.05)
        self.assertIsNotNone(calibration.load_imu_calibration(self.cal_file))

    def test_next_start_up_uses_the_saved_gyroscope_calibration(self):
        self.start()
        self.imu.rotation = np.eye(3)                    # the glove is switched off and on again
        result, shown = self.start()
        self.assertEqual(shown, [("READY", "")])         # no calibrating
        np.testing.assert_allclose(self.imu.rotation @ TURNED, np.eye(3), atol=0.05)

    def test_recording_always_calibrates_the_new_person(self):
        self.start()
        result, shown = self.start(recalibrate=True)
        self.assertIn(("CALIBRATING", "Hold hand FLAT"), shown)
        self.assertNotIn(("CALIBRATION", "REQUIRED"), shown)

    def test_without_a_gyroscope_only_two_steps(self):
        self.imu = None
        result, shown = self.start()
        prompts = [d for t, d in shown if t == "CALIBRATING"]
        self.assertEqual(prompts, ["Hold hand FLAT", "Make a FIST"])
        self.assertIn(("Calibration", "saved"), shown)

    def test_failed_gyroscope_steps_keep_the_fingers_and_the_previous_directions(self):
        self.start()
        good = calibration.load_imu_calibration(self.cal_file)
        self.hand.CALIBRATION_UP = {"FINGERS UP": (1.0, 0.0, 0.0), "THUMB UP": (1.0, 0.05, 0.0)}
        result, shown = self.start(recalibrate=True)
        self.assertIsNotNone(result)                     # the fingers were fine
        self.assertEqual(shown[-2], ("FAULT", "Check gyroscope"))
        kept = calibration.load_imu_calibration(self.cal_file)
        np.testing.assert_allclose(kept[0], good[0], atol=1e-4)


class SlowHand(PretendHand):
    """A user who takes their time: after each new prompt, the hand stays where it was
    for `delay` readings before moving, and can wobble for a while."""

    def __init__(self, delay=60, wobble=0):
        super().__init__()
        self.delay, self.wobble = delay, wobble
        self.reads_since_prompt, self.target, self.current = 0, [0.0] * 5, [0.0] * 5

    def follow(self, title, detail):
        before = getattr(self, "calibration_prompt", None)
        super().follow(title, detail)
        if self.calibrating and self.calibration_prompt != before:
            self.reads_since_prompt = 0
            self.target = list(self.calibration_pose)

    def pose(self):
        return self.current

    def reading(self, channel):
        if channel == 0:                                 # once per full set of readings
            self.reads_since_prompt += 1
            if self.reads_since_prompt > self.delay:
                self.current = self.target
            if 0 < self.reads_since_prompt - self.delay < self.wobble:   # still settling into the pose
                self.current = [min(1.0, b + 0.3 * (self.reads_since_prompt % 2)) for b in self.target]
        return super().reading(channel)


@mock.patch.multiple(routine, MESSAGE_SECONDS=0)
class WaitingStepTests(unittest.TestCase):
    """Steps 1 and 2 wait for the user instead of a 2-second timer."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.cal_file = Path(folder.name) / "calibration.json"

    def calibrate(self, hand):
        display = MockOLED(hand)
        with mock.patch("time.sleep"), mock.patch("builtins.print"):
            result = calibration.run_finger_calibration(InstantSampler(MockMCP3208(hand)), display, self.cal_file)
        return result, display.history

    def test_a_slow_fist_is_waited_for(self):
        # Each reading is 5 sets (InstantSampler's median), and the hand waits 60 sets: about 0.6 s
        # at 20 readings a second. The old timer would also have coped; 400 sets is about 4 s.
        result, shown = self.calibrate(SlowHand(delay=400))
        flat, fist = result
        np.testing.assert_allclose(fist, FIST_READINGS, atol=15)   # the real fist, not the flat hand
        self.assertEqual(shown[-1], ("Calibration", "saved"))

    def test_settling_into_the_pose_restarts_the_hold(self):
        result, _ = self.calibrate(SlowHand(delay=20, wobble=200))
        np.testing.assert_allclose(result[1], FIST_READINGS, atol=15)  # recorded after the wobbling stopped

    def test_a_pose_that_never_comes_gives_up(self):
        hand = SlowHand(delay=10 ** 9)                   # never makes the fist
        with mock.patch.object(calibration, "STEP_TIMEOUT_SECONDS", 1.0):
            result, shown = self.calibrate(hand)
        self.assertIsNone(result)
        self.assertEqual([t for t, _ in shown].count("Too slow"), routine.MAX_CAL_TRIES)
        self.assertEqual(shown[-1], ("FAULT", "Check wiring"))

    def test_the_touchscreen_bar_shows_the_hold(self):
        hand = SlowHand(delay=40)
        display = MockOLED(hand)
        bars = []
        display.progress = bars.append
        with mock.patch("time.sleep"), mock.patch("builtins.print"):
            calibration.run_finger_calibration(InstantSampler(MockMCP3208(hand)), display, self.cal_file)
        self.assertEqual(max(bars), 1.0)
        self.assertIn(0.0, bars)                         # empty while waiting for the fist


if __name__ == "__main__":
    unittest.main()
