"""Tests for the gyroscope: its readings, J and Z, the recordings' new columns, and one letter at a time."""

import sys
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))   # lets the tests import the files in glove/

import settings
import collect_data
import imu
import sign_feedback
from alphabet import EXPECTED_MOTION, EXPECTED_POSES, EXPECTED_TILT, HOW_TO, LETTERS, MOTION_LETTERS
from asl_model import NONE, augment, train_network
from mock_glove import MockIMU, PretendHand

STILL = (np.array([1.0, 0.0, 0.0]), np.zeros(3))
TRACING = (np.array([1.0, 0.0, 0.0]), np.array([0.5, 0.1, 0.3]))
REFS = {"A": [0.1, 1.0, 1.0, 1.0, 1.0], "I": [0.8, 1.0, 1.0, 1.0, 0.0],
        "J": [0.8, 1.0, 1.0, 1.0, 0.0], "Z": [0.7, 0.0, 1.0, 1.0, 1.0]}


class AlphabetTests(unittest.TestCase):

    def test_every_letter_including_j_and_z_has_a_pose_tilt_motion_and_prompt(self):
        for letter in LETTERS:
            self.assertEqual(len(EXPECTED_POSES[letter]), 5, letter)
            self.assertEqual(len(EXPECTED_TILT[letter]), 3, letter)
            self.assertIn(letter, HOW_TO)
        self.assertEqual(len(LETTERS), 26)

    def test_only_j_and_z_move(self):
        for letter in LETTERS:
            self.assertEqual(any(EXPECTED_MOTION[letter]), letter in MOTION_LETTERS, letter)

    def test_j_and_z_are_in_the_references(self):
        self.assertIn("J", settings.REFERENCES)
        self.assertIn("Z", settings.REFERENCES)


class FeatureTests(unittest.TestCase):

    def test_bends_only_without_a_gyroscope(self):
        np.testing.assert_array_equal(imu.hand_features([0.1] * 5), [0.1] * 5)

    def test_eleven_numbers_with_one(self):
        features = imu.hand_features([0.1] * 5, TRACING)
        self.assertEqual(len(features), len(imu.FEATURE_NAMES))
        bends, tilt, motion = imu.split_features(features)
        np.testing.assert_array_equal(motion, TRACING[1])
        self.assertTrue(imu.is_moving(motion))
        self.assertFalse(imu.is_moving(STILL[1]))
        self.assertFalse(imu.is_moving(None))

    def test_axis_map(self):
        matrix = imu.axis_map(("-y", "+x", "+z"))          # the board is turned 90 degrees on the glove
        np.testing.assert_array_equal(matrix @ [0.0, -1.0, 0.0], [1.0, 0.0, 0.0])


class ImuSamplerTests(unittest.TestCase):

    def test_tilt_and_motion_from_the_pretend_gyroscope(self):
        hand = PretendHand()
        sampler = imu.ImuSampler(MockIMU(hand))
        sampler.start()
        try:
            time.sleep(0.3)
            tilt, motion = sampler.state()
            np.testing.assert_allclose(tilt, [1, 0, 0], atol=0.05)
            self.assertFalse(imu.is_moving(motion))
            hand.up, hand.motion = (0.0, 1.0, 0.0), EXPECTED_MOTION["J"]
            time.sleep(settings.MOTION_SECONDS + 0.2)
            tilt, motion = sampler.state()
            np.testing.assert_allclose(tilt, [0, 1, 0], atol=0.05)
            np.testing.assert_allclose(motion, EXPECTED_MOTION["J"], atol=0.12)
            self.assertTrue(imu.is_moving(motion))
        finally:
            sampler.stop()

    def test_no_gyroscope_means_none(self):
        with mock.patch.object(settings, "USE_IMU", False):
            self.assertIsNone(imu.open_imu())
        with mock.patch.object(imu, "MPU6050", side_effect=OSError("no device")):
            self.assertIsNone(imu.open_imu())                 # carries on without it

    def test_mpu6050_reads_g_and_degrees_per_second(self):
        bus = mock.Mock()
        bus.read_byte_data.return_value = 0x68
        # accel x = 1 g (16384), gyro z = 131 deg/s x 2 (262 counts)
        values = [16384, 0, 0, 0, 0, 0, 262]
        bus.read_i2c_block_data.return_value = [b for v in values for b in v.to_bytes(2, "big", signed=True)]
        smbus = mock.Mock(SMBus=mock.Mock(return_value=bus))
        with mock.patch.dict(sys.modules, {"smbus2": smbus}):
            board = imu.MPU6050()
        accel, gyro = board.read()
        self.assertEqual(board.name, "MPU-6050")
        np.testing.assert_allclose(accel, [1, 0, 0])
        np.testing.assert_allclose(gyro, [0, 0, 2])


@mock.patch.multiple(settings, REFERENCES=REFS, USE_MODEL=False, ACTIVE_LETTERS=None)
class MotionLetterTests(unittest.TestCase):

    def test_still_i_is_i_and_traced_i_is_j(self):
        bends = np.array(REFS["I"])
        self.assertEqual(sign_feedback.recognize_with_margin(imu.hand_features(bends, STILL))[0], "I")
        self.assertEqual(sign_feedback.recognize_with_margin(imu.hand_features(bends, TRACING))[0], "J")

    def test_traced_pointing_hand_is_z(self):
        bends = np.array(REFS["Z"])
        self.assertEqual(sign_feedback.recognize_with_margin(imu.hand_features(bends, TRACING))[0], "Z")

    def test_without_a_gyroscope_j_and_z_never_show(self):
        self.assertEqual(sign_feedback.recognize_with_margin(np.array(REFS["I"]))[0], "I")
        self.assertNotEqual(sign_feedback.recognize(np.array(REFS["Z"]))[0], "Z")


class ModelWithGyroscopeTests(unittest.TestCase):
    """A network trained on bends + tilt + motion splits letters that bend the same."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(3)
        letters = ["L", "G", "I", "J", "U", "V"]
        features, labels = [], []
        for letter in letters:
            for _ in range(60):
                bends = np.clip(np.array(EXPECTED_POSES[letter]) + rng.normal(0, 0.03, 5), 0, 1)
                up = np.array(EXPECTED_TILT[letter]) + rng.normal(0, 0.05, 3)
                motion = np.clip(np.array(EXPECTED_MOTION[letter]) + np.abs(rng.normal(0, 0.02, 3)), 0, 1)
                features.append(imu.hand_features(bends, (up / np.linalg.norm(up), motion)))
                labels.append(letter)
        cls.model = train_network(np.array(features), np.array(labels), letters + [NONE], epochs=60)
        cls.model.groups = [["U", "V"]]

    def feature(self, letter):
        return imu.hand_features(EXPECTED_POSES[letter], (np.array(EXPECTED_TILT[letter]),
                                                          np.array(EXPECTED_MOTION[letter])))

    def test_tilt_splits_g_from_l_and_motion_splits_j_from_i(self):
        for letter in ("L", "G", "I", "J"):
            self.assertEqual(self.model.recognize_with_margin(self.feature(letter), max_distance=0.15)[0], letter)

    def test_one_letter_at_a_time(self):
        sign, _, _ = self.model.recognize_with_margin(self.feature("V"), use_groups=False)
        self.assertIn(sign, ("U", "V"))                    # one letter, never "U/V"
        self.assertEqual(self.model.recognize_with_margin(self.feature("V"))[0], "U/V")   # groups on

    def test_the_glove_shows_one_letter_by_default(self):
        self.assertFalse(settings.SHOW_LOOK_ALIKE_GROUPS)
        with mock.patch.object(sign_feedback, "load_model", return_value=self.model):
            sign, _, _ = sign_feedback.recognize_with_margin(self.feature("V"))
        self.assertNotIn("/", sign or "")

    def test_a_gyroscope_model_without_a_gyroscope_falls_back_to_references(self):
        self.assertEqual(self.model.n_inputs, 11)
        with mock.patch.object(sign_feedback, "load_model", return_value=self.model), \
             mock.patch.object(settings, "REFERENCES", REFS):
            self.assertIsNone(sign_feedback.model_for(np.array(REFS["A"])))
            self.assertEqual(sign_feedback.recognize_with_margin(np.array(REFS["A"]))[0], "A")

    def test_augment_keeps_tilt_a_direction_and_motion_in_range(self):
        out = augment(np.array([self.feature("G")] * 50), np.random.default_rng(0))
        np.testing.assert_allclose(np.linalg.norm(out[:, 5:8], axis=1), 1.0, atol=1e-6)
        self.assertTrue(np.all((out[:, 8:11] >= 0) & (out[:, 8:11] <= 1)))


class RecordingTests(unittest.TestCase):

    def test_columns_have_the_gyroscope(self):
        for name in imu.IMU_FEATURE_NAMES + ["accel_along", "gyro_back"]:
            self.assertIn(name, collect_data.COLUMNS)

    def test_a_row_with_and_without_the_gyroscope(self):
        flat, fist = np.array([2250.0] * 5), np.array([1500.0] * 5)
        frame = {"raw": np.array([1875.0] * 5), "tilt": np.array([1.0, 0, 0]), "motion": np.array([0.5, 0, 0]),
                 "accel": np.array([1.0, 0, 0]), "gyro": np.array([125.0, 0, 0])}
        row = collect_data.frame_row(frame, flat, fist)
        self.assertEqual(len(row) + 5, len(collect_data.COLUMNS))     # + person, session, letter, rep, frame
        self.assertEqual(row[0], "0.5000")
        self.assertEqual(row[8], "0.5000")                            # motion_along
        empty = collect_data.frame_row(dict(frame, tilt=None, motion=None, accel=None, gyro=None), flat, fist)
        self.assertEqual(empty[5:11], [""] * 6)

    def test_j_and_z_are_recorded_only_with_a_gyroscope(self):
        self.assertEqual(len(collect_data.choose_letters(None, imu=object())), 26)
        self.assertNotIn("J", collect_data.choose_letters(None, imu=None))
        with self.assertRaises(SystemExit):
            collect_data.choose_letters("A,J", imu=None)


if __name__ == "__main__":
    unittest.main()
