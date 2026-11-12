"""Optional raw touch scanner matching the MCP23017 touch addendum.

Not an I2C touch-controller board: Adafruit 3575 is a passive ribbon adapter.
Raw X/Y require physical ribbon verification and later positional calibration.
"""

import time


class TouchPanel:
    IODIRA, IODIRB = 0x00, 0x01
    IPOLA, IPOLB, GPINTENA, GPINTENB = 0x02, 0x03, 0x04, 0x05
    GPPUA, GPPUB, GPIOA, OLATA = 0x0C, 0x0D, 0x12, 0x14

    def __init__(self, adc, bus=None, address=0x20, sleep=time.sleep):
        if bus is None:
            from smbus import SMBus
            bus = SMBus(1)
        self.bus, self.adc, self.address, self.sleep = bus, adc, address, sleep
        # This driver owns the whole expander. BANK=0; no other MCP23017 driver
        # may run at the same time. Normalize BANK=1 safely: 0x05 is IOCON then;
        # with BANK=0, 0x05 is GPINTENB (also safe to clear).
        try:
            self._write(0x05, 0)
            self._write(0x0A, 0)         # IOCON, BANK=0, sequential mode default
            self._write(self.IODIRA, 0xFF)
            self._write(self.IODIRB, 0xFF)
            for reg in (self.IPOLA, self.IPOLB, self.GPINTENA, self.GPINTENB,
                        self.GPPUA, self.GPPUB):
                self._write(reg, 0)
        except BaseException:
            self.bus.close()
            raise

    def _write(self, register, value):
        self.bus.write_byte_data(self.address, register, value)

    def _state(self, direction, latch, pullups=0):
        self._write(self.IODIRA, 0xFF)   # release ALL electrodes before axis change
        self._write(self.GPPUA, 0)
        self._write(self.OLATA, latch)   # set future levels while pins are inputs
        self._write(self.GPPUA, pullups)
        self._write(self.IODIRA, direction)
        self.sleep(0.001)

    def pressed(self):
        # GPA0 / X+ input with weak pull-up; GPA3 / Y- low; other pins inputs.
        self._state(0xF7, 0x00, pullups=0x01)
        return not bool(self.bus.read_byte_data(self.address, self.GPIOA) & 0x01)

    def read_raw(self):
        try:
            if not self.pressed():
                return None
            self._state(0xFC, 0x01)      # X+ high, X- low; both Y pins inputs
            x = self.adc.read_average(6, 4)  # sense Y+ at ADC CH6 / pin 7
            self._state(0xF3, 0x04)      # Y+ high, Y- low; both X pins inputs
            y = self.adc.read_average(5, 4)  # sense X+ at ADC CH5 / pin 6
            return (x, y) if self.pressed() else None
        finally:
            self._write(self.IODIRA, 0xFF)
            self._write(self.GPPUA, 0)

    def close(self):
        try:
            self._write(self.IODIRA, 0xFF)
            self._write(self.GPPUA, 0)
        finally:
            self.bus.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
