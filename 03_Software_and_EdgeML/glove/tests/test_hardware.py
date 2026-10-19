"""Tests for hardware.py: the ADC's request bytes and the Sampler with its spike filter. No Pi needed."""

import sys
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))   # lets the tests import the files in glove/

import hardware
import settings
from mock_glove import FIST_READINGS, NOISE, MockMCP3208, PretendHand


class FakeSpiDev:
    """Pretends to be spidev.SpiDev: remembers each request and always answers 0xABC (2748),
    with junk in the bits the code should ignore."""

    def open(self, bus, device):
        self.sent = []

    def xfer2(self, data):
        self.sent.append(list(data))
        return [0xFF, 0xFA, 0xBC]


class MCP3208Tests(unittest.TestCase):

    def test_sends_the_datasheet_request_for_every_channel(self):
        expected = {   # single-ended requests from the MCP3208 datasheet, section 6.1
            0: [0x06, 0x00, 0x00], 1: [0x06, 0x40, 0x00], 2: [0x06, 0x80, 0x00], 3: [0x06, 0xC0, 0x00],
            4: [0x07, 0x00, 0x00], 5: [0x07, 0x40, 0x00], 6: [0x07, 0x80, 0x00], 7: [0x07, 0xC0, 0x00],
        }
        # The ADC class lives in Calibration_Routine/glove_calibration.py ("routine"), so patch spidev there.
        with mock.patch.object(settings.routine, "spidev", types.SimpleNamespace(SpiDev=FakeSpiDev), create=True):
            adc = hardware.MCP3208()
        for channel, request in expected.items():
            self.assertEqual(adc.read(channel), 0xABC)
            self.assertEqual(adc.spi.sent[-1], request)


class SamplerTests(unittest.TestCase):

    def test_keeps_only_the_newest_half_second(self):
        # Runs in real time (about 1.3 seconds) to check the background thread.
        hand = PretendHand()
        sampler = hardware.Sampler(MockMCP3208(hand))
        sampler.start()
        self.addCleanup(sampler.stop)
        time.sleep(0.5)                                  # the window fills with flat-hand readings...
        hand.follow("CALIBRATING", "Make a FIST")
        time.sleep(0.7)                                  # ...then fist readings push all of them out
        self.assertEqual(len(sampler.window), settings.WINDOW)
        np.testing.assert_allclose(sampler.average(), FIST_READINGS, atol=NOISE)

    def test_newest_ignores_a_single_spike(self):
        sampler = hardware.Sampler(adc=None)             # not started: fill the window by hand
        sampler.window.extend([[2250] * 5, [2252] * 5, [4095] * 5, [2248] * 5, [2250] * 5])
        np.testing.assert_allclose(sampler.newest(), 2250)


if __name__ == "__main__":
    unittest.main()
