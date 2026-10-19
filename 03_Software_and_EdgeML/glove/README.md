# ASL Learning Glove software

Runs on the Raspberry Pi Zero in the wrist enclosure and follows the main program flowchart in Section 6 of the Design Specification.

**This folder uses `Calibration_Routine`, which goes inside it** (`glove/Calibration_Routine/`). Calibration, the ADC, the wrist display and the sampler all come from `glove_calibration.py` there, unchanged, so calibrating with either program does the same thing and saves the same `calibration.json`.

| File | What it does |
|---|---|
| `main.py` | **The only file you run.** Start-up (flowchart sheet 1) and the practice loop (sheets 2 and 3) |
| `settings.py` | Every number you might tune. The ones calibration uses are set in `glove_calibration.py` and copied here |
| `hardware.py` | Calibration_Routine's ADC, OLED and `Sampler`, plus the Selection button, a spike filter and repeat-skipping |
| `mock_glove.py` | A pretend glove, so everything runs without a Pi |
| `startup_check.py` | `find_faulty_finger()`, the SENSOR FAULT check |
| `calibration.py` | Connects to Calibration_Routine's `run_calibration()`, plus one extra check on the saved file |
| `sign_feedback.py` | `normalize()`, `recognize()`, `evaluate()`, `HoldTimer` and `practice_step()` |
| `lesson.py` | `LessonManager`: the target sign and what the Selection button does |
| `tests/` | Automated checks for each file |

## Try it on any computer

You only need Python 3 and NumPy. Run these from this folder (on Windows, type `py` instead of `python3`):

```bash
pip install numpy
python3 -m unittest                                # runs every test in tests/
python3 main.py --mock                             # pretend glove: calibrates, then plays out a lesson
python3 main.py --mock --stuck-finger Ring         # loose sensor: type c + Enter to watch calibration fail
python3 main.py --mock --unplugged-finger Ring     # unplugged sensor: SENSOR FAULT at start-up
```

- **In mock mode, type `c` + Enter to press the Calibration button and `s` + Enter to press Selection.** Ctrl+C quits.
- **Mock mode saves its calibration to `Calibration_Routine/calibration_mock.json`,** so it never overwrites a real one. Delete that file to watch the start-up calibration again.
- **VS Code underlines the Pi-only imports on a laptop.** `pip install gpiozero luma.oled pillow` clears them (except `spidev`, which only installs on Linux).

## Wiring this code expects

The Calibration button pin is set in `Calibration_Routine/glove_calibration.py`, and the Selection pin in `settings.py`.

| Part | Raspberry Pi Zero |
|---|---|
| MCP3208 (through the TXU0304): SCLK, MOSI, MISO, CS | GPIO11 (pin 23), GPIO10 (pin 19), GPIO9 (pin 21), GPIO8 / CE0 (pin 24) |
| Wrist OLED: SDA, SCL | GPIO2 (pin 3), GPIO3 (pin 5) |
| Calibration button | GPIO17 (pin 11) to GND (pin 9) |
| Selection button | GPIO27 (pin 13) to GND (pin 14) |
| Flex sensors | MCP3208 CH0 Thumb, CH1 Index, CH2 Middle, CH3 Ring, CH4 Pinky |

The buttons use the Pi's internal pull-up, so they don't need resistors. If they end up wired to 3.3 V instead of GND, set `BUTTON_PULL_UP = False` in `glove_calibration.py`.

## Set up the Pi

Use Raspberry Pi OS Lite (32-bit). The Pi Zero can't run the 64-bit version.

```bash
sudo raspi-config nonint do_spi 0      # turn on SPI
sudo raspi-config nonint do_i2c 0      # turn on I2C
sudo apt update
sudo apt install -y python3-numpy python3-spidev python3-gpiozero python3-lgpio \
                    python3-pil i2c-tools fonts-dejavu-core
python3 -m venv --system-site-packages ~/glove-env
~/glove-env/bin/pip install luma.oled
sudo reboot
```

Then, from this folder: `~/glove-env/bin/python main.py`

The glove shows READY, then free practice. Press Selection to start the lesson.

## Bring-up checklist

1. **Find the OLED.** `i2cdetect -y 1` should show `3c`. If it shows `3d`, change `OLED_I2C_ADDRESS` in `glove_calibration.py`. The program still runs without the OLED and prints every screen message in the terminal.
2. **Check the finger order.** Each calibration prints a table of flat, fist and change values. Bend one finger at a time; if a different column moves, reorder `FINGERS` in `glove_calibration.py`.
3. **Tune `MIN_RANGE` in `glove_calibration.py`.** Calibrate a few people and look at the smallest change, usually the thumb. About half of that is a good setting.
4. **Record real reference signs.** The `REFERENCES` in `settings.py` are placeholders. Have someone who signs correctly hold each sign, and copy the bend values.

| Problem | Fix |
|---|---|
| `No such file or directory: '/dev/spidev0.0'` or `'/dev/i2c-1'` | SPI or I2C is off: rerun the `raspi-config` lines and reboot |
| `Permission denied` on SPI, I2C or GPIO | `sudo usermod -aG spi,i2c,gpio $USER`, then log out and back in |
| SENSOR FAULT at start-up | That finger reads near 0 V (unplugged) or 5 V (shorted): check its divider and connector |
| Screen shows garbage or is shifted a few pixels | The module may be an SH1106: change `ssd1306` to `sh1106` in `glove_calibration.py` |
| One Selection press skips two signs | The button bounces: raise `BUTTON_BOUNCE_SECONDS` to 0.1 |
| Readings jumpy | Lower `SPI_SPEED_HZ` in `glove_calibration.py` and keep the sensor wires short |

## Not built yet

- **The neural network.** `recognize()` is a stand-in. The network should take the same five bend values and return the same `(sign, confidence)` pair.
- **The teaching-screen app.** `lesson.py` prints what the HDMI monitor would show.
- **The RGB status LED.**
