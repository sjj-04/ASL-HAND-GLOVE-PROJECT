"""Tests for sign_feedback.py: normalize, recognize, evaluate, the hold timer and the messages."""

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))   # lets the tests import the files in glove/

import settings
from sign_feedback import HoldTimer, evaluate, normalize, practice_step, recognize

# Fixed values for these tests, so tuning settings.py later doesn't break them.
PINNED = dict(
    REFERENCES={"A": [0.1, 1.0, 1.0, 1.0, 1.0],
                "B": [0.9, 0.0, 0.0, 0.0, 0.0],
                "C": [0.4, 0.5, 0.5, 0.5, 0.5]},
    ACCEPT_CONFIDENCE=0.85, ADJUST_TOLERANCE=0.25, HOLD_SECONDS=1.0)
A, C = PINNED["REFERENCES"]["A"], PINNED["REFERENCES"]["C"]

FLAT, FIST = np.full(5, 2250.0), np.full(5, 1500.0)   # a calibration: straight reads 2250, bent 1500


def raw_for(bends):
    """The raw readings that the calibration above turns into these bends."""
    return FLAT + np.array(bends) * (FIST - FLAT)


def grade(bends, target):
    bends = np.array(bends)
    recognized, confidence = recognize(bends)
    return evaluate(bends, target, recognized, confidence)


class NormalizeTests(unittest.TestCase):

    def test_design_spec_example(self):
        self.assertAlmostEqual(normalize(1875.0, 2250.0, 1500.0), 0.5)

    def test_flat_is_zero_and_fist_is_one(self):
        np.testing.assert_allclose(normalize(FLAT, FLAT, FIST), 0.0)
        np.testing.assert_allclose(normalize(FIST, FLAT, FIST), 1.0)

    def test_readings_past_the_calibrated_range_are_clipped(self):
        self.assertEqual(normalize(2400.0, 2250.0, 1500.0), 0.0)     # straighter than flat
        self.assertEqual(normalize(1300.0, 2250.0, 1500.0), 1.0)     # tighter than the fist

    def test_works_if_bending_raises_the_reading(self):
        self.assertAlmostEqual(normalize(1875.0, 1500.0, 2250.0), 0.5)


@mock.patch.multiple(settings, **PINNED)
class RecognizeTests(unittest.TestCase):

    def test_each_reference_is_recognized_with_full_confidence(self):
        for sign, reference in PINNED["REFERENCES"].items():
            recognized, confidence = recognize(np.array(reference))
            self.assertEqual(recognized, sign)
            self.assertAlmostEqual(confidence, 1.0)

    def test_picks_the_closest_sign(self):
        recognized, confidence = recognize(np.array([0.2, 0.9, 0.9, 1.0, 1.0]))
        self.assertEqual(recognized, "A")
        self.assertAlmostEqual(confidence, 0.94)


@mock.patch.multiple(settings, **PINNED)
class EvaluateTests(unittest.TestCase):

    def test_exact_sign_is_correct(self):
        self.assertEqual(grade(A, "A"), "CORRECT")

    def test_one_finger_out_of_place_is_named(self):
        self.assertEqual(grade([0.1, 0.0, 1.0, 1.0, 1.0], "A"), "ADJUST: Index")

    def test_several_fingers_off_means_try_again(self):
        self.assertEqual(grade([0.0, 0.0, 0.0, 0.0, 0.0], "A"), "TRY AGAIN")

    def test_close_but_not_confident_means_try_again(self):
        self.assertEqual(grade([0.3, 0.8, 0.8, 0.8, 0.8], "A"), "TRY AGAIN")   # all a little off

    def test_confident_match_with_one_finger_off_is_not_correct(self):
        index_half_open = [0.1, 0.3, 1.0, 1.0, 1.0]
        _, confidence = recognize(np.array(index_half_open))
        self.assertGreaterEqual(confidence, 0.85)        # confident on average...
        self.assertEqual(grade(index_half_open, "A"), "ADJUST: Index")   # ...but one finger is far off


@mock.patch.multiple(settings, **PINNED)
class HoldTimerTests(unittest.TestCase):

    def test_needs_one_unbroken_second(self):
        timer = HoldTimer()
        self.assertFalse(timer.update(True, now=10.0))   # starts timing
        self.assertFalse(timer.update(True, now=10.5))
        self.assertTrue(timer.update(True, now=11.0))

    def test_a_wrong_reading_restarts_the_clock(self):
        timer = HoldTimer()
        timer.update(True, now=10.0)
        timer.update(False, now=10.6)                    # the hold was broken
        self.assertFalse(timer.update(True, now=11.0))   # timing starts again here
        self.assertTrue(timer.update(True, now=12.0))


@mock.patch.multiple(settings, **PINNED)
class PracticeStepTests(unittest.TestCase):
    """The messages from flowchart sheets 2 and 3, from raw readings to the display."""

    def step(self, bends, target, timer, now):
        return practice_step(raw_for(bends), (FLAT, FIST), target, timer, now)

    def test_correct_sign_shows_hold_steady_then_correct(self):
        timer = HoldTimer()
        self.assertEqual(self.step(A, "A", timer, 0.0), ("Target: A", "HOLD STEADY", False))
        self.assertEqual(self.step(A, "A", timer, 0.5), ("Target: A", "HOLD STEADY", False))
        self.assertEqual(self.step(A, "A", timer, 1.0), ("CORRECT: A", "", True))

    def test_wrong_sign_shows_feedback(self):
        timer = HoldTimer()
        self.assertEqual(self.step([0.1, 0.0, 1.0, 1.0, 1.0], "A", timer, 0.0),
                         ("Target: A", "ADJUST: Index", False))
        self.assertEqual(self.step([0.0, 0.0, 0.0, 0.0, 0.0], "A", timer, 0.0),
                         ("Target: A", "TRY AGAIN", False))

    def test_free_practice_names_the_sign_but_never_says_correct(self):
        timer = HoldTimer()
        self.assertEqual(self.step(C, None, timer, 0.0), ("Free practice", "SIGN: C", False))
        self.assertEqual(self.step([0.6, 0.2, 0.9, 0.3, 0.7], None, timer, 0.0),
                         ("Free practice", "SIGN: --", False))


if __name__ == "__main__":
    unittest.main()
