"""Tests for calibration.py: every path through the run_calibration() flowchart, and the saved file."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))   # lets the tests import the files in glove/

import settings
from calibration import load_calibration, run_calibration, save_calibration
from mock_glove import (FIST_READINGS, FLAT_READINGS, NOISE, InstantSampler, MockMCP3208,
                           MockOLED, PretendHand)


def failed_try(attempt, finger="Ring"):
    """The messages one failed attempt shows."""
    return [("CALIBRATING", "Hold hand FLAT"),
            ("CALIBRATING", "Make a FIST"),
            (f"Check {finger}", f"Try {attempt} of {settings.MAX_CAL_TRIES}")]


SUCCESS = [("CALIBRATING", "Hold hand FLAT"), ("CALIBRATING", "Make a FIST"), ("Calibration", "saved")]


class HandThatFixesItsSensor(PretendHand):
    """Starts with a loose ring sensor, then re-seats it when the glove says "Check ..."."""

    def follow(self, title, detail):
        super().follow(title, detail)
        if title.startswith("Check"):
            self.stuck = None


class CalibrationTestCase(unittest.TestCase):
    """Gives each test its own throwaway calibration file."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.cal_file = Path(folder.name) / "calibration.json"


class RunCalibrationTests(CalibrationTestCase):

    def calibrate(self, hand):
        """Run calibration, skipping the waits. Returns (result, messages shown)."""
        display = MockOLED(hand)
        with mock.patch("time.sleep"), mock.patch("builtins.print"):
            result = run_calibration(InstantSampler(MockMCP3208(hand)), display, self.cal_file)
        return result, display.history

    def test_good_hand_saves_and_returns_the_readings(self):
        result, shown = self.calibrate(PretendHand())
        self.assertEqual(shown, SUCCESS)
        flat, fist = result
        np.testing.assert_allclose(flat, FLAT_READINGS, atol=NOISE)
        np.testing.assert_allclose(fist, FIST_READINGS, atol=NOISE)
        saved = json.loads(self.cal_file.read_text())
        np.testing.assert_allclose(saved["flat"], flat, atol=0.05)

    def test_loose_sensor_gets_every_try_then_fault(self):
        result, shown = self.calibrate(PretendHand(stuck_finger="Ring"))
        self.assertIsNone(result)
        every_try = [message for attempt in range(1, settings.MAX_CAL_TRIES + 1)
                     for message in failed_try(attempt)]
        self.assertEqual(shown, every_try + [("FAULT", "Check wiring")])
        self.assertFalse(self.cal_file.exists())         # nothing is saved after a failure

    def test_succeeds_on_second_try_after_the_sensor_is_fixed(self):
        result, shown = self.calibrate(HandThatFixesItsSensor(stuck_finger="Ring"))
        self.assertIsNotNone(result)
        self.assertEqual(shown, failed_try(1) + SUCCESS)


class SavedCalibrationTests(CalibrationTestCase):
    """The start-up question "Calibration saved on microSD?"."""

    def test_saved_calibration_loads_back(self):
        save_calibration(np.array(FLAT_READINGS), np.array(FIST_READINGS), self.cal_file)
        flat, fist = load_calibration(self.cal_file)
        np.testing.assert_array_equal(flat, FLAT_READINGS)
        np.testing.assert_array_equal(fist, FIST_READINGS)

    def test_missing_file_means_not_calibrated(self):
        self.assertIsNone(load_calibration(self.cal_file))

    def test_damaged_file_means_not_calibrated(self):
        self.cal_file.write_text('{"flat": [2250, 23')    # cut off partway through
        self.assertIsNone(load_calibration(self.cal_file))

    def test_wrong_number_of_fingers_means_not_calibrated(self):
        self.cal_file.write_text(json.dumps({"flat": [2250, 2300], "fist": [1750, 1500]}))
        self.assertIsNone(load_calibration(self.cal_file))

    def test_finger_with_too_small_a_range_means_not_calibrated(self):
        fist = list(FIST_READINGS)
        fist[3] = FLAT_READINGS[3]                       # the ring finger "never bent"
        save_calibration(np.array(FLAT_READINGS), np.array(fist), self.cal_file)
        self.assertIsNone(load_calibration(self.cal_file))


if __name__ == "__main__":
    unittest.main()
