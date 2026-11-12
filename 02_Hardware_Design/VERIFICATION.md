# Winstar / original Raspberry Pi Zero software review

Reviewed November 8, 2026. The original download is preserved. This copy is
intended for displaying text/images, reading five flex sensors, saving simple
flex calibration, and optionally checking raw touch input.

## Result

The display driver's thirteen BCM GPIO assignments match the supplied
`ASL_Final_Assembly_Schematic.pdf`. No display pin reassignment or command change
was needed. `WinstarLCD` uses GPIO directly; the MCP23017 is used only to drive
the optional touchscreen electrodes.

| Winstar pin | Function | Pi physical pin | BCM GPIO |
| --- | --- | --- | --- |
| 1 | /CS1 | 11 | 17 |
| 2 | /CS2 | 13 | 27 |
| 6 | D/I | 15 | 22 |
| 8 | E | 16 | 23 |
| 17 | /RESET | 18 | 24 |
| 9 | DB0 | 29 | 5 |
| 10 | DB1 | 31 | 6 |
| 11 | DB2 | 32 | 12 |
| 12 | DB3 | 33 | 13 |
| 13 | DB4 | 36 | 16 |
| 14 | DB5 | 35 | 19 |
| 15 | DB6 | 38 | 20 |
| 16 | DB7 | 40 | 21 |

The Winstar specification confirms active-low chip selects/reset, the
NT7108-compatible page/column commands, and a 128-by-64 framebuffer divided
between two 64-column controllers. Reset, display-off/clear/display-on,
page addressing, data bit order, and both halves were checked. The driver's
2 microsecond setup/high/hold delays exceed the documented digital timing
minimums. Actual edge rise/fall times require measurement.

The specified logic-high threshold is 2.0 V, compatible with the intended
3.3 V Pi output levels. LCD R/W must remain grounded: the GPIO data pins are
outputs only, and the driver uses fixed delays rather than reading busy status.
The 5 V LCD supply and backlight are supplied by the circuit, not switched by
the software.

The ADC uses SPI0/CE0, mode 0, MSB first, 24-clock single-ended transactions,
and a 12-bit result mask. CH0-CH4 correspond to Thumb, Index, Middle, Ring and
Little/Pinky. The 20 kHz starting clock is within the ADC's documented range;
source-impedance errors and actual conversion accuracy still require bench checks.

The optional MCP23017 scan uses I2C bus 1/address 0x20 and GPA0-GPA3. It releases
all electrodes before changing drive pairs and senses X on CH6 and Y on CH5,
matching the touch addendum. Raw touch is not calibrated screen positioning.

## Corrections in this copy

- Reject non-object JSON calibration files with a readable ValueError instead
  of an unhandled AttributeError. Reject boolean values as ADC counts.
- Correct the README's unsupported backlight reference to R21/33 ohm.
- Add regression checks tied to the actual schematic pin map, all eight data
  bits, erasure on both display halves, and malformed calibration files.
- State the initial-check defaults and calibration-format difference explicitly.

## Hardware details that software cannot settle

- The assembly drawing omits Pi power-rail and ground connections. The intended
  connections are Pi physical pin 1 to +3.3 V, pin 2 to +5 V, and pin 6 to the
  common ground, consistent with the later touch addendum.
- R8 is labeled 33 kohm and bypassed by a wire in the supplied drawing. The
  original README instead specified R21/33 ohm. R21 actually connects LCD reset
  to ground through 10 kohm. Backlight current limiting requires confirmation
  of the actual module and corrected circuit; this review does not approve the
  drawn backlight branch.
- With touch fitted, ADC pins 6/7 must connect to X+/Y+ taps instead of ground.
  ADC pin 8 stays grounded. Physical ribbon electrode order remains to be checked.
- The older application's GPIO17/GPIO27 pushbuttons conflict with LCD chip
  selects. Do not run that hardware configuration concurrently.

## Scope and validation

All ten hardware-independent unittest cases pass, and all packaged Python files
compile. A CLI check confirms malformed calibration is reported without opening
hardware. Simulated LCD RAM verifies cleared startup memory, page/half boundaries,
active-low selection, every data bit, pixel erasure, and unchanged-frame suppression.

This is software verification on a Windows computer. It establishes no measured
Pi GPIO voltage, oscilloscope timing, optical display result, ADC accuracy, touch
orientation, or startup reliability. On the original Pi Zero, first run:

```sh
python3 asl_glove.py display-test --seconds 20
python3 asl_glove.py sensor-test
python3 asl_glove.py live
```

Use the README for installation and optional touch/startup steps. The package
targets Python 3 with Pillow, RPi.GPIO and spidev on a 32-bit Raspberry Pi OS;
optional touch also needs smbus. Hardware imports remain deferred so offline
tests and CLI help work without Pi libraries.

This is not the complete ASL lesson/scoring system or the Deliverable 2 acceptance
runner. Defaults remain 10 Hz sensor frames, 2 Hz display, 20 kHz SPI, and a
50-count flex-calibration span. It does not establish the separate 200 Hz/
250 kHz/10 display writes per second/500-count acceptance requirements.

## References reviewed

- Supplied assembly: `C:/Users/Milly/Documents/ECET400/ASL_Final_Assembly_Schematic.pdf`.
- Supplied touch addendum: `ASL_Touch_Ribbon_MCP23017_Schematic.pdf`, November 5, 2026.
- Winstar WDG0151-TMI-V#N00, electrical characteristics, interface, timing and
  display instructions: https://download.mikroe.com/documents/datasheets/glcd-128x64-datasheet.pdf
- Microchip MCP3204/3208, SPI framing and minimum clock frequency:
  https://ww1.microchip.com/downloads/en/DeviceDoc/21298E.pdf
- Microchip MCP23017/MCP23S17, register addressing and GPIO configuration:
  https://ww1.microchip.com/downloads/en/DeviceDoc/20001952C.pdf
