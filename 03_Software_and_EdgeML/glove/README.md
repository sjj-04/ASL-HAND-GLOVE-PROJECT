# ASL Learning Glove software

Runs on the Raspberry Pi Zero in the wrist enclosure and follows the main program flowchart in Section 6 of the Design Specification.

**This folder uses `Calibration_Routine` (`glove/Calibration_Routine/`). Calibration, the ADC, the wrist display and the sampler all come from `glove_calibration.py` there, unchanged, so calibrating with either program does the same thing and saves the same `calibration.json`.

| File | What it does |
|---|---|
| `main.py` | **The only file you run.** Start-up (flowchart sheet 1), the touchscreen's home screen, and free practice (sheets 2 and 3) |
| `touchscreen/` | The Winstar touchscreen: home screen (Free / Lesson / Recalibrate), free mode, touch panel, and the laptop simulator |
| `imu.py` | The gyroscope (MPU-6050 on I2C): which way the hand points (tilt) and how it moves (motion) |
| `settings.py` | Every number you might tune. The ones calibration uses are set in `glove_calibration.py` and copied here |
| `hardware.py` | Calibration_Routine's ADC, OLED and `Sampler`, plus the Selection button, a spike filter and repeat-skipping |
| `mock_glove.py` | A pretend glove, so everything runs without a Pi |
| `startup_check.py` | `find_faulty_finger()`, the SENSOR FAULT check |
| `calibration.py` | Connects to Calibration_Routine's `run_calibration()`, plus one extra check on the saved file |
| `sign_feedback.py` | `normalize()`, `recognize()`, `evaluate()`, `HoldTimer` and `practice_step()` |
| `lesson.py` | `LessonManager`: the target sign and what the Selection button does |
| `tests/` | Automated checks for each file |
| `alphabet.py` | All 26 ASL letters (J and Z are traced): how to form each one, and which ones look alike to the sensors |
| `collect_data.py` | Records each person's letters for training (see `TRAINING.md`) |
| `train_model.py` | Trains and tests the letter network, saves `models/asl_model.npz` |
| `asl_model.py` | The letter network the Pi runs (numpy only): finger bends + gyroscope in, one letter out. Off until `USE_MODEL = True` |
| `TRAINING.md` | **Step-by-step guide to recording, training and turning on the letter model** |

## Try it on any computer

You only need Python 3 and NumPy. Run these from this folder (on Windows, type `py` instead of `python3`):

```bash
pip install numpy
python3 -m unittest                                # runs every test in tests/
python3 main.py --mock                             # pretend glove: the touchscreen in a window
python3 main.py --mock --text                      # the same, drawn in the terminal (type help)
python3 main.py --mock --no-touchscreen            # no home screen: straight into free practice
python3 main.py --mock --stuck-finger Ring         # loose sensor: calibration fails
python3 main.py --mock --unplugged-finger Ring     # unplugged sensor: SENSOR FAULT at start-up
```

- **In the window, click the screen to touch it.** The buttons under it (or the letter keys) pick what the pretend hand signs: hold a letter 1 second and free mode shows it. J and Z are traced, so the pretend gyroscope moves. The window needs Pillow (`pip install pillow`).
- **The pretend hand calibrates by itself** (all 4 steps) when you tap Recalibrate, or tap Free with nothing saved.
- **Mock mode saves its calibration to `Calibration_Routine/calibration_mock.json`,** so it never overwrites a real one. Delete that file to watch the start-up calibration again.
- **VS Code underlines the Pi-only imports on a laptop.** `pip install gpiozero luma.oled pillow` clears them (except `spidev`, which only installs on Linux).

## Wiring this code expects

The Calibration button pin is set in `Calibration_Routine/glove_calibration.py`, and the Selection pin in `settings.py`.

| Part | Raspberry Pi Zero |
|---|---|
| MCP3208 (through the TXU0304): SCLK, MOSI, MISO, CS | GPIO11 (pin 23), GPIO10 (pin 19), GPIO9 (pin 21), GPIO8 / CE0 (pin 24) |
| Calibration button | GPIO17 (pin 11) to GND (pin 9) |
| Selection button | GPIO27 (pin 13) to GND (pin 14) |
| Flex sensors | MCP3208 CH0 Thumb, CH1 Index, CH2 Middle, CH3 Ring, CH4 Pinky |
| Gyroscope (MPU-6050): SDA, SCL, VCC, GND | GPIO2 (pin 3), GPIO3 (pin 5), 3.3 V (pin 1), GND (pin 6) |
| Touch panel X+, X-, Y+, Y- | GPIO4 (pin 7) + MCP3208 CH5, GPIO14 (pin 8), GPIO15 (pin 10) + MCP3208 CH6, GPIO18 (pin 12) |

**The gyroscope** is optional: with `USE_IMU = True` the glove looks for it at start-up, and if it isn't there it says so and carries on with the flex sensors (J and Z can't be read without it). Mount it flat on the back of the hand or wrist. **Calibration learns how it sits on each person** (steps 3 and 4 below), so the mounting doesn't have to be exact. Check it with `python3 imu.py`: after a calibration, fingers pointing up reads about (1, 0, 0) and thumb up (0, 1, 0).

**Calibration has 4 steps with the gyroscope** (2 without), all in `calibration.py`. **Each step waits for the person:** nothing is recorded until they've reached the pose and held it still for 1 second, while the progress bar fills. Moving starts the hold again, so there's no rushing to read the prompt:

| Step | Prompt | What it measures |
|---|---|---|
| 1 | Hold hand FLAT | each finger's 0%. Records once the hand is still (it's often already flat) |
| 2 | Make a FIST | each finger's 100%. Waits for the fingers to bend, then still. Each finger keeps its own range, so the glove fits the person |
| 3 | Point fingers UP | the gyroscope's "along the fingers" direction and its still reading. Waits for the fingers to straighten (or the hand to turn), then still |
| 4 | Thumb UP | the gyroscope's "toward the thumb" direction. Waits for the hand to turn at least 60° from step 3, then still |

A step that doesn't happen within 20 seconds says "Too slow" (or "Hold still" / "Turn hand more" for the gyroscope steps) and tries again: 3 tries, then FAULT, keeping the previous gyroscope calibration. `HOLD_STILL_SECONDS` and `STEP_TIMEOUT_SECONDS` at the top of `calibration.py` set the timings. (`Calibration_Routine/glove_calibration.py` keeps the original 2-second timed routine for its own program and the Deliverable 3.1/3.2 tests.) Everything is saved in `calibration.json`. Until the first calibration with the gyroscope, `IMU_AXES` in `settings.py` is used as a rough guess.

**The touch panel** needs the serial port off, because GPIO14/15 are its pins: `sudo raspi-config` → Interface Options → Serial Port → login shell **No**, serial hardware **No**. Check the panel with a multimeter first: X+ to X- and Y+ to Y- should each read a few hundred ohms. Then run `python3 main.py --touch-test`, touch all four corners, Ctrl+C, and copy the numbers it prints into `settings.py`. If the touch panel can't be opened, main.py skips the home screen and goes straight to free practice.

The buttons use the Pi's internal pull-up, so they don't need resistors. If they end up wired to 3.3 V instead of GND, set `BUTTON_PULL_UP = False` in `glove_calibration.py`.

## Set up the Pi

Use Raspberry Pi OS Lite (32-bit). The Pi Zero can't run the 64-bit version.

```bash
sudo raspi-config nonint do_spi 0      # turn on SPI
sudo raspi-config nonint do_i2c 0      # turn on I2C
sudo apt update
sudo apt install -y python3-numpy python3-spidev python3-gpiozero python3-lgpio \
                    python3-pil python3-smbus i2c-tools fonts-dejavu-core
python3 -m venv --system-site-packages ~/glove-env
~/glove-env/bin/pip install luma.oled
sudo reboot
```

Then, from this folder: `~/glove-env/bin/python main.py`

The glove checks the sensors, loads the saved calibration (fingers and gyroscope), shows READY, then the home screen. It never forces a calibration at start-up: the latest one (`calibration.json`) is used until someone taps **Recalibrate**, so the same person can switch it off and on and carry on. The very first time (nothing saved yet), tapping **Free** asks for a flat hand and a fist first. On the home screen: tap **Free** for free practice (one letter at a time, **Menu** to go back), **Recalibrate** to calibrate again. **Lesson** says "Coming soon" for now. With `USE_TOUCHSCREEN = False` it goes straight into free practice.

## Bring-up checklist

1. **Find the gyroscope.** `i2cdetect -y 1` should show `68` (or `69` if its AD0 pin is tied to 3.3 V: set `IMU_ADDRESS = 0x69`). The program still runs without it.
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

- **Real training data.** The letter network is built (`asl_model.py`, see `TRAINING.md`) and reads the gyroscope too, but it needs each teammate's recordings (all 26 letters, with the gyroscope working) before it can be turned on.
- **Lessons on the touchscreen.** The Lesson button shows "Coming soon".
- **The teaching-screen app.** `lesson.py` prints what the HDMI monitor would show.
- **The RGB status LED.**
