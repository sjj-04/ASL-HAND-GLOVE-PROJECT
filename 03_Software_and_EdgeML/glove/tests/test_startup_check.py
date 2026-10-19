"""Tests for startup_check.py, and for every path through start-up (flowchart sheet 1)."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parents[1]))   # lets the tests import the files in glove/

from calibration import save_calibration
from main import start_up
from mock_glove import FIST_READINGS, FLAT_READINGS, InstantSampler, MockMCP3208, MockOLED, PretendHand
from startup_check import find_faulty_finger


class FindFaultyFingerTests(unittest.TestCase):

    def test_all_sensors_in_range(self):
        self.assertIsNone(find_faulty_finger([2250, 2300, 2280, 2260, 2240]))

    def test_unplugged_sensor_reads_near_zero(self):
        self.assertEqual(find_faulty_finger([2250, 2300, 2280, 3, 2240]), "Ring")

    def test_shorted_sensor_reads_near_4095(self):
        self.assertEqual(find_faulty_finger([4090, 2300, 2280, 2260, 2240]), "Thumb")

    def test_names_the_first_faulty_finger(self):
        self.assertEqual(find_faulty_finger([2250, 0, 2280, 4095, 2240]), "Index")


class StartUpTests(unittest.TestCase):
    """Each test walks one path through sheet 1 with the pretend glove."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.cal_file = Path(folder.name) / "calibration.json"

    def start(self, hand):
        """Run start-up, skipping the waits. Returns (result, messages shown)."""
        display = MockOLED(hand)
        with mock.patch("time.sleep"), mock.patch("builtins.print"):
            result = start_up(InstantSampler(MockMCP3208(hand)), display, self.cal_file)
        return result, display.history

    def test_unplugged_sensor_stops_with_sensor_fault(self):
        result, shown = self.start(PretendHand(unplugged_finger="Ring"))
        self.assertIsNone(result)
        self.assertEqual(shown, [("SENSOR FAULT", "Check Ring")])

    def test_saved_calibration_goes_straight_to_ready(self):
        save_calibration(FLAT_READINGS, FIST_READINGS, self.cal_file)
        result, shown = self.start(PretendHand())
        self.assertIsNotNone(result)
        self.assertEqual(shown, [("READY", "")])

    def test_nothing_saved_calibrates_first(self):
        result, shown = self.start(PretendHand())
        self.assertIsNotNone(result)
        self.assertEqual(shown[0], ("CALIBRATING", "Hold hand FLAT"))
        self.assertEqual(shown[-1], ("READY", ""))
        self.assertTrue(self.cal_file.exists())

    def test_failed_calibration_stops_without_ready(self):
        result, shown = self.start(PretendHand(stuck_finger="Ring"))
        self.assertIsNone(result)
        self.assertEqual(shown[-1], ("FAULT", "Check wiring"))


if __name__ == "__main__":
    unittest.main()
