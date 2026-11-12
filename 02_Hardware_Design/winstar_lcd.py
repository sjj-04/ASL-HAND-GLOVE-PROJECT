"""Write-only Winstar WDG0151-TMI-V#N00 driver for the ECET400 wiring.

BCM GPIO numbering. /CS1 and /CS2 are ACTIVE LOW on this exact module.
R/W (LCD pin 7) must be wired to GND; this driver never reads the LCD bus.
"""

import time

from PIL import Image


def wait_us(microseconds):
    """Minimum software delay, not a promise of real-time Linux timing."""
    deadline = time.perf_counter_ns() + int(microseconds * 1000)
    while time.perf_counter_ns() < deadline:
        pass


def pack_image(image):
    """1024 bytes: page 0..7, column 0..127; bit 0 is top of each page."""
    if image.size != (128, 64):
        raise ValueError("LCD image must be exactly 128 x 64 pixels")
    pixels = image.convert("1").load()
    return bytes(
        sum((1 << bit) for bit in range(8) if pixels[x, page * 8 + bit])
        for page in range(8)
        for x in range(128)
    )


class WinstarLCD:
    WIDTH, HEIGHT = 128, 64
    CS = (17, 27)                         # Pi physical 11, 13 -> LCD 1, 2
    DI, E, RESET = 22, 23, 24            # Pi physical 15, 16, 18 -> LCD 6, 8, 17
    DATA = (5, 6, 12, 13, 16, 19, 20, 21)  # DB0..DB7 -> LCD 9..16

    def __init__(self, gpio_module=None, delay_us=wait_us, sleep=time.sleep,
                 swap_halves=False):
        if gpio_module is None:
            import RPi.GPIO as gpio_module
        self.gpio = gpio_module
        self.delay_us, self.sleep = delay_us, sleep
        self.swap_halves = swap_halves
        self.pins = self.CS + (self.DI, self.E, self.RESET) + self.DATA
        self._previous = None
        self._closed = False
        gpio_module.setmode(gpio_module.BCM)
        try:
            for pin in self.CS:
                gpio_module.setup(pin, gpio_module.OUT, initial=gpio_module.HIGH)
            for pin in (self.E, self.RESET, self.DI) + self.DATA:
                gpio_module.setup(pin, gpio_module.OUT, initial=gpio_module.LOW)
            self.initialize()
        except BaseException:
            gpio_module.cleanup(self.pins)
            raise

    def _write(self, half, value, data=False):
        if self._closed:
            raise RuntimeError("LCD is closed")
        if half not in (0, 1) or not 0 <= value <= 255:
            raise ValueError("Invalid LCD half or byte")
        gpio = self.gpio
        gpio.output(self.E, 0)
        gpio.output(self.CS, (1, 1))
        gpio.output(self.DI, int(data))
        gpio.output(self.DATA, tuple((value >> bit) & 1 for bit in range(8)))
        physical_half = 1 - half if self.swap_halves else half
        gpio.output(self.CS[physical_half], 0)
        self.delay_us(2)                 # data and control setup
        gpio.output(self.E, 1)
        self.delay_us(2)                 # E high; data captured on falling E
        gpio.output(self.E, 0)
        self.delay_us(2)                 # data/control hold
        gpio.output(self.CS, (1, 1))
        self.delay_us(100)               # no busy-flag reads with R/W grounded

    def initialize(self):
        self.gpio.output(self.E, 0)
        self.gpio.output(self.CS, (1, 1))
        self.gpio.output(self.RESET, 0)
        self.sleep(0.050)                # allow power to settle
        self.sleep(0.010)                # external reset low
        self.gpio.output(self.RESET, 1)
        self.sleep(0.010)
        for half in (0, 1):
            self._write(half, 0x3E)      # display off while clearing RAM
            self._write(half, 0xC0)      # start line 0
        self._previous = None
        self.show(Image.new("1", (128, 64), 0), force=True)
        self.display_on(True)

    def display_on(self, enabled=True):
        """Enable/disable pixels. Backlight is controlled by the power wiring."""
        for half in (0, 1):
            self._write(half, 0x3F if enabled else 0x3E)

    def show(self, image, force=False):
        """Send a Pillow image; update changed column spans only."""
        current = pack_image(image)
        for page in range(8):
            for half in (0, 1):
                base = page * 128 + half * 64
                changed = [col for col in range(64)
                           if force or self._previous is None
                           or current[base + col] != self._previous[base + col]]
                if not changed:
                    continue
                first, last = changed[0], changed[-1]
                self._write(half, 0xB8 | page)
                self._write(half, 0x40 | first)
                for col in range(first, last + 1):
                    self._write(half, current[base + col], data=True)
        self._previous = current

    def close(self):
        if self._closed:
            return
        try:
            self.display_on(False)
            self.gpio.output(self.RESET, 0)
        finally:
            self.gpio.cleanup(self.pins)  # never reset the SPI or I2C pins here
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
