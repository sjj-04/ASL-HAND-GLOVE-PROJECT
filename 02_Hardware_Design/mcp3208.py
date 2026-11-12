"""MCP3208 CH0..CH4 flex acquisition over SPI0 / CE0 via TXU0304."""

import time
from dataclasses import dataclass


FINGERS = ("Thumb", "Index", "Middle", "Ring", "Little")


@dataclass(frozen=True)
class SensorFrame:
    timestamp_ns: int
    counts: tuple                  # Thumb, Index, Middle, Ring, Little


class MCP3208:
    def __init__(self, spi=None, speed_hz=20000):
        if not 10000 <= speed_hz <= 20000:
            raise ValueError("This assembly starts at 10-20 kHz; use 20 kHz")
        if spi is None:
            import spidev
            spi = spidev.SpiDev()
        self.spi = spi
        try:
            spi.open(0, 0)         # /dev/spidev0.0; hardware CE0 controls ADC /CS
            spi.max_speed_hz = speed_hz
            spi.mode = 0
            spi.bits_per_word = 8
            spi.lsbfirst = False
            spi.cshigh = False
        except BaseException:
            spi.close()
            raise

    def read_channel(self, channel):
        if not isinstance(channel, int) or not 0 <= channel <= 7:
            raise ValueError("MCP3208 channel must be an integer from 0 through 7")
        # Leading 00000, START=1, SGL=1, D2; next byte starts D1,D0.
        reply = self.spi.xfer2([0x06 | (channel >> 2), (channel & 3) << 6, 0x00])
        return ((reply[1] & 0x0F) << 8) | reply[2]

    def read_average(self, channel, samples=4):
        if not isinstance(samples, int) or not 1 <= samples <= 64:
            raise ValueError("samples must be 1 through 64")
        self.read_channel(channel)  # discard first conversion after mux change
        return round(sum(self.read_channel(channel) for _ in range(samples)) / samples)

    def read_frame(self, samples=4):
        counts = tuple(self.read_average(ch, samples) for ch in range(5))
        return SensorFrame(time.monotonic_ns(), counts)

    def close(self):
        self.spi.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
