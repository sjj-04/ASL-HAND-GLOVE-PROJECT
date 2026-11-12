# ECET400 — original Pi Zero / Winstar software

This package draws text, lines, shapes and 128 x 64 images on the Winstar
WDG0151-TMI-V#N00. It initializes the LCD, reads the five MCP3208 flex inputs,
saves two-position flex calibration and displays live counts or bend percentages.
An optional raw touch test matches the MCP23017 touch addendum.

The driver targets the exact final-assembly GPIO wiring plus the touch addendum.
It is a new Python integration package. The earlier Python source file was not
available for a direct merge. Existing application code can import these modules
using the example below. The acquisition API preserves a `read_frame()` with a
monotonic timestamp and Thumb / Index / Middle / Ring / Little ordering.

Software checks passed with simulated peripherals. The actual Pi, LCD, ADC,
translator and touch panel have not been bench tested with this code. Fixed
delays are conservative starting values. Check the assembled signals and readings.

## 1. Boot the original Pi Zero

Use Raspberry Pi Imager to write **Raspberry Pi OS Lite (32-bit)** to a microSD
card. Select the original Zero as the device. Lite opens a text login; it is
sufficient for this application. A desktop OS is also supported but consumes
more resources. Set a username and password in Imager.

For first setup, connect an HDMI monitor with a **mini-HDMI** adapter/cable and
a keyboard through a micro-USB OTG adapter in the port marked **USB**. Insert
the microSD card. Connect the regulated **5 V, 2.5 A** supply from the schematic
to **PWR IN**. The Pi starts as soon as power is applied.

The original Zero has no built-in Wi-Fi or Ethernet. Package installation below
needs an internet connection, for example through a supported USB Ethernet or
USB Wi-Fi adapter. A powered USB hub may be needed to share its USB port with
the keyboard. After installation, this software can run without internet access.

The Winstar is driven by this application. Use the HDMI monitor or a configured
remote terminal to install software and enter commands.

## 2. Put this package on the Pi

Download **ASL_PiZero_Winstar.zip** on your computer.

One transfer method without built-in Wi-Fi: after writing the OS and before
inserting the card in the Pi, copy the ZIP onto the Windows-visible `bootfs`
partition of the microSD card. Do not format or overwrite the other partition.
Eject the card and boot the Pi. Then enter:

```bash
sudo apt update
sudo apt install -y python3 python3-rpi.gpio python3-spidev python3-pil python3-smbus i2c-tools unzip fonts-dejavu-core
unzip /boot/firmware/ASL_PiZero_Winstar.zip -d "$HOME"
cd "$HOME/ASL_PiZero_Winstar"
```

If you already copied/extracted the folder another way, simply open that folder
in the Pi terminal. These packages are installed with apt; a pip virtual
environment is not required. `python3-smbus` and `i2c-tools` support the optional
MCP23017 touch branch.

## 3. Enable the buses

```bash
sudo raspi-config
```

Select **Interface Options → SPI → Yes** and **Interface Options → I2C → Yes**.
Finish and reboot:

```bash
sudo reboot
```

After logging in again:

```bash
cd "$HOME/ASL_PiZero_Winstar"
ls /dev/spidev0.0
ls /dev/i2c-1
```

SPI0 / CE0 is the ADC connection. The LCD uses normal GPIO output, so it needs
no separate kernel display overlay. Leave SPI1 and I2S overlays disabled because
the assigned LCD data pins include GPIO19, GPIO20 and GPIO21.

## 4. Check the powered assembly

With power removed, verify the complete assembly wiring before connecting the
supply. Keep the TXU0304 in the ADC path. Do not connect the ADC's 5 V DOUT to a
Pi GPIO directly. All circuit returns share common ground.

LCD connections used by the driver:

| LCD function | Winstar connector pin | Pi physical pin | BCM GPIO in code |
|---|---:|---:|---:|
| /CS1 | 1 | 11 | 17 |
| /CS2 | 2 | 13 | 27 |
| D/I | 6 | 15 | 22 |
| E | 8 | 16 | 23 |
| /RST | 17 | 18 | 24 |
| DB0 | 9 | 29 | 5 |
| DB1 | 10 | 31 | 6 |
| DB2 | 11 | 32 | 12 |
| DB3 | 12 | 33 | 13 |
| DB4 | 13 | 36 | 16 |
| DB5 | 14 | 35 | 19 |
| DB6 | 15 | 38 | 20 |
| DB7 | 16 | 40 | 21 |

Keep Winstar pin 3 (VSS), pin 7 (R/W) and pin 20 (K) at common GND. Pin 4
(VDD) goes to +5 V. Pin 19 (A) receives +5 V **through R21, 33 ohm / 0.5 W**.
RV1 is one **10 kohm linear potentiometer**: one outer lead to +5 V, the other
outer lead to Winstar pin 18 (VEE), and its wiper to Winstar pin 5 (V0).
Identify the physical wiper from the potentiometer's own pinout. VEE is the
negative-voltage output. Keep the schematic's startup resistors and decoupling.

When you connect power, the existing circuit supplies the LCD logic and
backlight immediately. Python does not switch the 5 V supply. Python then
holds /RST low, releases reset, clears both controller halves and sends the
display-on command. `display_on(False)` hides pixels; it does not switch the
backlight off. Software backlight switching would require a hardware change.

## 5. Test the Winstar first

Run this from the package folder:

```bash
python3 asl_glove.py display-test
```

You should see **WINSTAR READY**, a border, a left/right division, a rectangle
and a circle. The image alternates with its inverse every four seconds. Slowly
adjust RV1 until the pattern is readable. Press **Ctrl+C** to stop the test.
The program disables pixels on exit; the backlight stays powered.

To run for a fixed interval instead:

```bash
python3 asl_glove.py display-test --seconds 20
```

To show your own 128 x 64 image, or the provided example:

```bash
python3 asl_glove.py draw-image example_128x64.png
```

The driver uses low-active chip selects as specified by this Winstar module.
It uses fixed write delays because R/W is grounded. There is no LCD identity or
busy readback. A successful program exit by itself does not prove that the
display is connected. Verify the visible test pattern.

## 6. Test the five flex sensors

Stop the display test with Ctrl+C. Run:

```bash
python3 asl_glove.py sensor-test
```

The terminal prints five 12-bit readings, each in the range 0–4095:

| Finger | MCP3208 channel | MCP3208 physical pin |
|---|---:|---:|
| Thumb | CH0 | 1 |
| Index | CH1 | 2 |
| Middle | CH2 | 3 |
| Ring | CH3 | 4 |
| Little | CH4 | 5 |

Move one finger at a time. Its channel should change. The existing divider has
the flex sensor toward +5 V and its 22 kohm resistor toward GND. If the sensor's
resistance increases when bent, the ADC count decreases.

SPI mode is 0 and the requested clock is 20 kHz. The first conversion on every
channel is discarded, then four conversions are averaged by default. These
choices do not eliminate ADC leakage or guarantee the accuracy of a high
impedance divider. Compare stable input voltages with a meter. Approximate
voltage is `count * measured_VREF / 4096`. VREF and VDD remain connected to the
same +5 V rail in this assembly.

Optional log:

```bash
python3 asl_glove.py sensor-test --csv flex_readings.csv
```

The CSV includes an end-of-frame monotonic timestamp in nanoseconds. Channels
are sampled sequentially. The default 10 Hz acquisition target can be slower
on the original Zero. `--rate` changes the target and `--samples` changes the
averaging count. Display writes also consume time in the integrated program.

## 7. Put the sensor values on the screen

```bash
python3 asl_glove.py live
```

The Winstar shows all five raw counts. It refreshes at a target of 2 Hz to keep
the original Zero's GPIO display traffic modest. Terminal readings are printed
approximately once per second. Press Ctrl+C to stop.

To save usable straight/bent endpoints:

```bash
python3 asl_glove.py calibrate
```

Follow the screen/terminal prompts. Hold all fingers straight and press Enter
on the keyboard. Then hold the measured bent position and press Enter again.
Keep each pose steady while samples are collected. The command creates
`flex_calibration.json`. A movement span below 50 counts on any finger is
rejected as an initial wiring/pose check, not as a sensor manufacturer's limit.

Run `python3 asl_glove.py live` again. It now displays raw counts and normalized
bend percentages. Straight is 0% and the saved bent endpoint is 100%. This is
two-position normalization. ASL lesson logic and gesture classification can
consume these readings in the next application layer.

## 8. Optional: test the touch ribbon branch

The Adafruit 3575 / DigiKey 1528-2358-ND adapter is passive. The MCP23017 drives
the four electrodes. The MCP3208 senses the touch voltages.

**Remove the old ground wires from ADC pin 6 (CH5) and pin 7 (CH6)** before
connecting the touch taps. ADC pin 8 (CH7) remains at GND. Follow the touch
addendum's power-off procedure to establish the ribbon electrode pairs.

| MCP23017 pin | Function | Connection through 470 ohm | ADC tap on ribbon side |
|---|---|---|---|
| 21 | GPA0 | X+ | MCP3208 pin 6 / CH5 |
| 22 | GPA1 | X− | None |
| 23 | GPA2 | Y+ | MCP3208 pin 7 / CH6 |
| 24 | GPA3 | Y− | None |

MCP23017 pin 9 is 3.3 V, pin 10 is GND, pin 12 is Pi physical pin 5 / SCL,
pin 13 is Pi physical pin 3 / SDA. Pins 15–17 are GND. Pin 18 connects through
R22 (10 kohm) to 3.3 V. C8 (100 nF) connects between pins 9 and 10.

Check for its address, then run the test:

```bash
i2cdetect -y 1
python3 asl_glove.py touch-test
```

`i2cdetect` should show `20`. The test prints `No press` or raw X/Y counts.
Move a stylus/finger and check the raw coordinates. The software releases all
four electrodes before switching axes. Pull-ups are disabled during analog
measurements. X is sensed on CH6 while X+/X− are driven; Y is sensed on CH5
while Y+/Y− are driven.

To show five flex readings and raw touch information together:

```bash
python3 asl_glove.py live --touch
```

Touch positional calibration is separate from flex calibration. Actual ribbon
pair order, axis orientation and raw span must be established on the panel
before assigning 128 x 64 touchscreen button coordinates. The 3.3 V touch drive
against the 5 V ADC reference will not span the full 0–4095 count range.

## 9. Optional: start automatically when power is connected

Complete the manual tests first. Stop any manually running hardware program
with Ctrl+C. From the package folder, install the program and service:

```bash
sudo install -d /opt/asl_glove /var/lib/asl-glove
sudo install -m 0644 asl_glove.py winstar_lcd.py mcp3208.py flex_calibration.py touch_panel.py /opt/asl_glove/
sudo install -m 0644 asl-glove.service /etc/systemd/system/asl-glove.service
```

If you saved flex calibration, copy it for the service:

```bash
sudo install -m 0644 flex_calibration.json /var/lib/asl-glove/flex_calibration.json
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now asl-glove.service
sudo systemctl status asl-glove.service --no-pager
```

The service runs with system hardware access and initializes the LCD after
Linux boots. It needs neither a login nor an internet connection. The default
service reads five flex sensors. To include the verified touch branch, append
`--touch` to the service's `ExecStart` line, reload the service definitions and
restart it.

To troubleshoot:

```bash
sudo journalctl -u asl-glove.service -n 40 --no-pager
```

To return to manual tests, first stop the service so two processes do not
drive the same pins:

```bash
sudo systemctl stop asl-glove.service
```

To disable automatic startup:

```bash
sudo systemctl disable --now asl-glove.service
```

To shut down the Pi cleanly:

```bash
sudo poweroff
```

Wait for shutdown to complete before removing power.

## 10. Draw from an existing Python application

Keep the driver files in the same folder as your application, or add their
folder to its Python import path. Install the apt dependencies listed above.
This example initializes the screen, reads the five flex channels and draws
the thumb count. Replace the drawing statements with the lesson UI as needed.

```python
from PIL import Image, ImageDraw
from winstar_lcd import WinstarLCD
from mcp3208 import MCP3208

with WinstarLCD() as lcd, MCP3208() as adc:
    frame = adc.read_frame()  # .timestamp_ns and .counts (five values)
    canvas = Image.new("1", (128, 64), 0)
    draw = ImageDraw.Draw(canvas)
    draw.text((2, 2), "ASL Learning Glove", fill=1)
    draw.text((2, 18), "Thumb: " + str(frame.counts[0]), fill=1)
    draw.rectangle((2, 35, 100, 55), outline=1)
    lcd.show(canvas)
    input("Press Enter to finish")
```

Pixel coordinates are x=0–127 and y=0–63. A set pixel is the foreground.
`lcd.show(canvas)` transfers the image. Editing the Pillow image alone does not
change the physical LCD. This build uses GPIO, Pillow and spidev. Earlier
Arduino/U8g2 or SSD1306 I2C display calls need to be adapted to this interface.

## Troubleshooting

| Symptom | First checks |
|---|---|
| No backlight | Measure supply; verify LCD 19 through R21 to +5 V and 20 to GND. Software cannot correct missing backlight power. |
| Backlight but no pattern | Adjust RV1. Verify LCD pin 4 at +5 V; pin 3 and R/W pin 7 at GND; /RST pin 17 goes high during the test. Check connector orientation and all GPIO wires. |
| Only half the screen | Check /CS1, /CS2 and DB0–DB7. The driver's selects are active low. |
| Text halves reversed | Check that LCD pin 1 goes to GPIO17 and pin 2 to GPIO27. A verified different assembly can use the driver's `swap_halves=True`. |
| Counts stuck at 0/4095 | Measure the actual sensor node and ADC reference. Check SPI0 enablement, U2 direction/pinout and ADC supplies/grounds. The ADC has no identity register. |
| Module import error | Install the apt packages with the system Python. Run `/usr/bin/python3 asl_glove.py ...`. |
| Permission denied on GPIO/SPI/I2C | The Pi login user needs `gpio`, `spi` and `i2c` group access. Use `sudo usermod -aG gpio,spi,i2c "$USER"` and reboot. |
| Program exits when service is running | Stop the service before manual tests. One process owns these hardware interfaces. |
| No I2C 0x20 | Check MCP23017 3.3 V supply, ground, SDA/SCL, grounded A0–A2 and pin 18 pull-up. |

## Validation and design references

Run the hardware-independent checks on the Pi or another Python computer with
Pillow installed:

```bash
python3 -m unittest discover -s tests -v
```

Seven checks cover LCD pixel/page/half boundaries and startup RAM clearing,
active-low selection, unchanged-frame suppression, ADC command and result
packing for all eight channels, finger ordering/timestamps, touch state/sense
mapping, and calibration persistence and clipping. These checks use simulated
peripherals. Actual startup waveforms, GPIO levels, ADC settling/accuracy,
refresh rate, ribbon orientation and touch calibration remain bench checks.

The primary protocol references are:

- Winstar WDG0151-TMI-V#N00 specification, interface pins, timing and command table:
  https://www.farnell.com/datasheets/1878006.pdf
- Microchip MCP3204/3208, serial transfer framing:
  https://ww1.microchip.com/downloads/en/DeviceDoc/21298e.pdf
- Microchip MCP23017, register map and I/O direction:
  https://ww1.microchip.com/downloads/en/DeviceDoc/20001952C.pdf
- RPi.GPIO setup, initial output values and multi-channel output:
  https://sourceforge.net/p/raspberry-gpio-python/wiki/BasicUsage/
- Python spidev, `xfer2` and hardware chip-select handling:
  https://github.com/doceme/py-spidev
- Pillow drawing API:
  https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html
- Raspberry Pi OS and setup:
  https://www.raspberrypi.com/software/operating-systems/
  https://www.raspberrypi.com/documentation/computers/getting-started.html

Wiring authority: the final assembly engineering notes, Revision B (2026-10-24),
plus ASL_Touch_Ribbon_MCP23017_Schematic.pdf (2026-11-05). The touch addendum
supersedes the old CH5/CH6 ground ties in the base assembly.
