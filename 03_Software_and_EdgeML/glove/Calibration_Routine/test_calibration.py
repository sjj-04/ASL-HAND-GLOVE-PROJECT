"""
Automated checks that run_calibration() follows the flowchart.
 
Run them from this folder with:
    python3 -m unittest -v
 
No Raspberry Pi needed: the tests use the pretend glove in mock_hardware.py.
"""
 
import contextlib
import io
import json
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock
 
import numpy as np
 
import glove_calibration as glove
from mock_hardware import FIST_READINGS, FLAT_READINGS, NOISE, MockMCP3208, MockOLED, PretendHand
 
RING = glove.FINGERS.index("Ring")
 
# What the wrist display should show on each path through the flowchart.
SUCCESS_MESSAGES = [
    ("CALIBRATING", "Hold hand FLAT"),
    ("CALIBRATING", "Make a FIST"),
    ("Calibration", "saved"),
]
 
 
def failed_try_messages(attempt, finger="Ring"):
    return [
        ("CALIBRATING", "Hold hand FLAT"),
        ("CALIBRATING", "Make a FIST"),
        (f"Check {finger}", f"Try {attempt} of 3"),
    ]
 
 
class InstantSampler:
    """Like glove.Sampler, but averages the pretend hand right away,
    so these tests don't have to wait in real time."""
 
    def __init__(self, adc):
        self.adc = adc
 
    def average(self):
        readings = [[self.adc.read(channel) for channel in range(len(glove.FINGERS))]
                    for _ in range(glove.WINDOW)]
        return np.mean(readings, axis=0)
 
 
class HandThatFixesItsSensor(PretendHand):
    """Starts with a loose sensor, then re-seats it when the glove says "Check ..."."""
 
    def follow(self, title, detail):
        super().follow(title, detail)
        if title.startswith("Check"):
            self.stuck_channel = None
 
 
class RunCalibrationTests(unittest.TestCase):
    """Each test walks one path through the run_calibration() flowchart."""
 
    def setUp(self):
        # Save calibration files in a throwaway folder, and skip the real waiting.
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.cal_file = Path(folder.name) / "calibration.json"
        no_waiting = mock.patch("time.sleep")
        no_waiting.start()
        self.addCleanup(no_waiting.stop)
 
    def calibrate(self, hand):
        """Run calibration with a pretend hand. Returns (result, messages shown)."""
        display = MockOLED(hand)
        sampler = InstantSampler(MockMCP3208(hand))
        with contextlib.redirect_stdout(io.StringIO()):    # keep the test output tidy
            result = glove.run_calibration(sampler, display, self.cal_file)
        return result, display.history
 
    def test_good_hand_saves_and_returns_the_readings(self):
        result, shown = self.calibrate(PretendHand())
 
        self.assertEqual(shown, SUCCESS_MESSAGES)
        flat, fist = result
        np.testing.assert_allclose(flat, FLAT_READINGS, atol=NOISE)
        np.testing.assert_allclose(fist, FIST_READINGS, atol=NOISE)
 
        saved = json.loads(self.cal_file.read_text())
        np.testing.assert_allclose(saved["flat"], flat, atol=0.05)
        np.testing.assert_allclose(saved["fist"], fist, atol=0.05)
 
    def test_loose_sensor_gets_three_tries_then_fault(self):
        result, shown = self.calibrate(PretendHand(stuck_channel=RING))
 
        self.assertIsNone(result)
        self.assertEqual(shown, failed_try_messages(1)
                                + failed_try_messages(2)
                                + failed_try_messages(3)
                                + [("FAULT", "Check wiring")])
        self.assertFalse(self.cal_file.exists())          # nothing is saved after a failure
 
    def test_succeeds_on_second_try_after_user_fixes_the_sensor(self):
        result, shown = self.calibrate(HandThatFixesItsSensor(stuck_channel=RING))
 
        self.assertIsNotNone(result)
        self.assertEqual(shown, failed_try_messages(1) + SUCCESS_MESSAGES)
        self.assertTrue(self.cal_file.exists())
 
 
class SavedCalibrationTests(unittest.TestCase):
    """The start-up check: "Calibration saved on microSD?"."""
 
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.cal_file = Path(folder.name) / "calibration.json"
 
    def test_saved_calibration_loads_back(self):
        flat = np.array([2250.0, 2300.0, 2280.0, 2260.0, 2240.0])
        fist = np.array([1750.0, 1500.0, 1450.0, 1480.0, 1550.0])
        glove.save_calibration(flat, fist, self.cal_file)
 
        loaded_flat, loaded_fist = glove.load_calibration(self.cal_file)
        np.testing.assert_array_equal(loaded_flat, flat)
        np.testing.assert_array_equal(loaded_fist, fist)
 
    def test_missing_file_means_not_calibrated(self):
        self.assertIsNone(glove.load_calibration(self.cal_file))
 
    def test_damaged_file_means_not_calibrated(self):
        self.cal_file.write_text('{"flat": [2250, 23')     # cut off partway through
        self.assertIsNone(glove.load_calibration(self.cal_file))
 
    def test_wrong_number_of_fingers_means_not_calibrated(self):
        self.cal_file.write_text(json.dumps({"flat": [1, 2, 3], "fist": [4, 5, 6]}))
        self.assertIsNone(glove.load_calibration(self.cal_file))
 
 
class FakeSpiDev:
    """Pretends to be spidev.SpiDev: remembers what was sent and always
    answers 0xABC (2748), with junk in the bits the code should ignore."""
 
    def open(self, bus, device):
        self.sent = []
 
    def xfer2(self, data):
        self.sent.append(list(data))
        return [0xFF, 0xFA, 0xBC]
 
    def close(self):
        pass
 
 
class HardwareTests(unittest.TestCase):
 
    def test_adc_sends_the_datasheet_command_for_every_channel(self):
        # Single-ended request bytes from the MCP3208 datasheet (Figure 6-1).
        expected = {
            0: [0x06, 0x00, 0x00], 1: [0x06, 0x40, 0x00],
            2: [0x06, 0x80, 0x00], 3: [0x06, 0xC0, 0x00],
            4: [0x07, 0x00, 0x00], 5: [0x07, 0x40, 0x00],
            6: [0x07, 0x80, 0x00], 7: [0x07, 0xC0, 0x00],
        }
        fake_spidev = types.SimpleNamespace(SpiDev=FakeSpiDev)
        with mock.patch.object(glove, "spidev", fake_spidev, create=True):
            adc = glove.MCP3208()
 
        for channel, command in expected.items():
            self.assertEqual(adc.read(channel), 2748)
            self.assertEqual(adc.spi.sent[-1], command)
 
    def test_sampler_keeps_only_the_newest_half_second(self):
        # Runs in real time (about 2 seconds) to check the background thread.
        hand = PretendHand()
        sampler = glove.Sampler(MockMCP3208(hand))
        sampler.start()
        self.addCleanup(sampler.stop)
 
        time.sleep(1.0)        # the window fills with flat readings...
        hand.pose = "fist"
        time.sleep(1.0)        # ...then fist readings push all of them out
 
        self.assertEqual(len(sampler.window), glove.WINDOW)
        np.testing.assert_allclose(sampler.average(), FIST_READINGS, atol=NOISE)
 
 
if __name__ == "__main__":
    unittest.main()
 
