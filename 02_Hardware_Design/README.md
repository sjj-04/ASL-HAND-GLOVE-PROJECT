# Winstar touchscreen / Free Mode A B C

Prepared November 8, 2026 for the user's Raspberry Pi Zero W, Raspberry Pi OS
13 Trixie (32-bit), final Winstar parallel-display schematic, and MCP23017 touch
addendum. The original supplied files and earlier reviewed package are preserved.

## What this package does

- Draws a home screen with Learning and Free Mode touch buttons.
- Learning opens a reserved screen with a MENU button. Lessons are not included.
- Free Mode uses the five glove sensors to display a recognized A, B, or C.
- Saves five-point touch calibration, finger endpoints, and three samples per
  letter. Setup runs on the touchscreen, without keyboard input after launch.
- Rejects sensor faults, distant/ambiguous matches and inconsistent sign samples.
- Requires at least 0.5 s of consistent recognition before showing a letter.
- Keeps acquisition in a background thread while drawing. ADC access is shared
  with touch under a lock. It makes no 200 Hz acquisition claim.

Recognition is an initial personalized static A/B/C matcher, not a trained
full-alphabet model. Five bend sensors do not measure palm orientation, finger
crossing, or spatial movement. Unsupported handshapes can have the same bend
readings as recorded shapes; this device cannot reliably reject those aliases.
J/Z movement and full-alphabet discrimination need additional sensing and/or an
appropriate validated model. This package does not recognize words or translate
ASL. No production sign templates are invented: the wearer records actual poses.

## Existing electrical connections

The display and ADC drivers retain the pin assignments in the reviewed package.
All GPIO identifiers here are BCM numbers; physical Pi header numbers differ.

| Function | BCM GPIO | Pi physical pin | Display pin |
| --- | --- | --- | --- |
| /CS1 | 17 | 11 | 1 |
| /CS2 | 27 | 13 | 2 |
| D/I | 22 | 15 | 6 |
| E | 23 | 16 | 8 |
| /RESET | 24 | 18 | 17 |
| DB0 | 5 | 29 | 9 |
| DB1 | 6 | 31 | 10 |
| DB2 | 12 | 32 | 11 |
| DB3 | 13 | 33 | 12 |
| DB4 | 16 | 36 | 13 |
| DB5 | 19 | 35 | 14 |
| DB6 | 20 | 38 | 15 |
| DB7 | 21 | 40 | 16 |

Display pin 4 is +5 V; pins 3, 7 and 20 are common GND. Contrast pot ends are
+5 V and display pin 18; its wiper is display pin 5. The user confirmed the
backlight resistor is 33 ohm. It must be in series from +5 V to display pin 19,
with no jumper bypassing it. This software does not switch the backlight power.
Pi physical pin 6 joins common GND. Do not rewire while powered.

The ADC remains SPI0/CE0 through TXU0304, mode 0 at the existing 20 kHz starting
clock. CH0-CH4 are Thumb, Index, Middle, Ring, Little/Pinky. MCP23017 address is
0x20 on I2C bus 1. Its GPA0/1/2/3 connect through four individual 470 ohm
resistors to X+/X-/Y+/Y-. Ribbon-side X+ goes to ADC CH5/pin 6; Y+ goes to
CH6/pin 7. The old ground ties on ADC pins 6/7 must be removed; pin 8 stays GND.

## Transfer from Windows

The new ZIP is in `C:\Users\Milly\Downloads\ASL_PiZero_FreeMode.zip`.
In Windows PowerShell:

```powershell
scp "$env:USERPROFILE\Downloads\ASL_PiZero_FreeMode.zip" amads_rpi0@192.168.1.155:~/
```

Use the Pi login password. The address is the one reported in this session;
if it changes, use the Pi's current `hostname -I` IPv4 address.

## Install and launch on the Pi

The libraries and SPI/I2C enablement were already completed in this session.
For a fresh installation, the dependencies are python3-rpi.gpio, python3-spidev,
python3-pil, python3-smbus, i2c-tools, unzip and fonts-dejavu-core.

Stop any existing display/sensor test with Ctrl+C. If an earlier startup service
was enabled, stop it with `sudo systemctl stop asl-glove.service`. Use only one
hardware program at a time. Then run in the Pi Terminal:

```sh
unzip ~/ASL_PiZero_FreeMode.zip -d ~/asl-freemode
cd ~/asl-freemode/ASL_PiZero_FreeMode
python3 -m unittest discover -s tests -v
i2cdetect -y 1
python3 free_mode.py
```

Expect 25 tests ending OK and address 20 in the I2C table. If 20 is absent,
check the touch expander power/reset, SDA/SCL and address connections before
launching. Existing extraction files may cause unzip to prompt about replacement;
save any edits before replacing them. Profiles live separately in the home folder.

## First launch, entirely on the Winstar

1. Five touch targets appear one at a time. Press the cross steadily with a
   finger or blunt resistive-touch stylus until it advances, then release.
   Release between all five targets. This handles reversed/swapped axes.
2. The home screen shows Learning and Free Mode. Tap Free Mode.
3. Put on the glove. Straighten all fingers. Use the OTHER hand to tap RECORD.
   A two-second countdown gives time to settle; hold still through recording.
4. Repeat for a closed fist. Each finger needs at least 200 ADC counts of span.
5. Form ASL A and record three samples, then B three times, then C three times.
   Each prompt identifies the letter and sample number. Use correct handshapes
   verified by someone who knows ASL; setup does not teach handshapes.
6. Free Mode starts. Form a recorded letter and hold it. HOLD A/B/C precedes
   the large accepted letter. NO MATCH clears the large letter for readings
   outside the matching thresholds. MENU returns home; SETUP repeats glove setup.

Three samples for a letter must agree within 0.12 mean normalized difference;
different letters need at least 0.10 separation. Invalid setup can be retried or
canceled to the menu. A previous valid glove profile is preserved until an entire
new setup succeeds. Recognition requires mean error <=0.10, largest finger error
<=0.20, a 0.04 lead over the next candidate, and the steady hold. These values are
engineering starting thresholds, not measured recognition accuracy.

Readings below 50 or above 4045 counts report the affected finger and clear
recognition. After a fault, valid readings must persist for at least one second,
followed by filter warmup and the normal hold. These checks detect rail-like
faults; they cannot detect every wiring problem or a stuck midscale sensor.

Profiles persist in `~/.local/share/asl-freemode/touch.json` and
`~/.local/share/asl-freemode/glove_profile.json`. Reuse them only with the same
wearer, mounting, sensors, and reference poses. To repeat touch calibration:

```sh
python3 free_mode.py --touch-calibrate
```

To force complete glove setup:

```sh
python3 free_mode.py --glove-setup
```

Ctrl+C ends a manual session. The LCD pixels turn off; its wired backlight remains on.

## Optional automatic startup after manual validation

Stop the manual application with Ctrl+C first. From the extracted package folder:

```sh
mkdir -p ~/.config/systemd/user
cp asl-freemode.service ~/.config/systemd/user/
systemctl --user daemon-reload
sudo loginctl enable-linger "$USER"
systemctl --user enable --now asl-freemode.service
```

The saved calibration is reused. Startup can show the home screen without a
terminal or a laptop connection. To stop before running manual tests:

```sh
systemctl --user stop asl-freemode.service
```

To disable startup:

```sh
systemctl --user disable --now asl-freemode.service
```

View errors with `journalctl --user -u asl-freemode.service -n 40 --no-pager`.

## Troubleshooting

| Result | Action |
| --- | --- |
| Backlight only | Run `python3 asl_glove.py display-test`; slowly adjust contrast. Check reset/control/data connections if no pattern appears. Power alone supplies no pixel data. |
| Display test fails | Stop here and resolve the display before debugging recognition. |
| No touch response | Check address 0x20 and run `python3 asl_glove.py touch-test`. Require No press when released and changing coordinates when pressed. |
| Touch targets fail or buttons select incorrectly | Recheck electrode pairs, then rerun with `--touch-calibrate`. |
| Not enough finger movement | Run `python3 asl_glove.py sensor-test`; bend one finger at a time. Check sensor placement, channel order and divider connections. |
| A/B/C samples too similar | Re-form distinct correct signs. Five bend readings may not separate these poses with the current mounting; do not force a label. |
| Pi GPIO/I2C permission error | Check that the account has gpio, spi and i2c group membership. The supplied user service runs as that account. |
| Screen turns off when remote Terminal closes | Keep the manual session open, or use the optional startup service after testing. |

## Files and validation

`free_mode.py`: menu, capture UI, concurrent acquisition and startup entry point.
`recognition.py`: profile validation, normalization, A/B/C matching, hold gate.
`touch_mapping.py`: five-point affine mapping and touch hit testing.
`winstar_lcd.py`, `mcp3208.py`, `touch_panel.py`: reviewed hardware drivers.
`asl_glove.py`, `flex_calibration.py`: earlier display/sensor/touch diagnostics.
`tests/`: 25 hardware-independent tests; `asl-freemode.service`: optional startup.

All 25 tests passed on the preparation computer. All Python sources compile and
the CLI help runs without Pi hardware imports. Tests cover exact GPIO wiring,
display RAM/bit boundaries, SPI framing, touch drive/sense mapping, inverted touch
calibration, menu routing, profile persistence, ambiguity rejection, faults and
hold timing. Synthetic test vectors are not deployed sign templates. Hardware
operation, physical touch accuracy, actual sample rate, and recognition accuracy
on the wearer's glove still require the above Pi checks.
