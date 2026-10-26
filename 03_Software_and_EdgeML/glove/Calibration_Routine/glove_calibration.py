"""
ASL Learning Glove - calibration
================================
 
Runs on the Raspberry Pi Zero in the wrist enclosure and follows the
run_calibration() flowchart in Section 6 of the Design Specification.
 
  * Start-up: load calibration.json from the microSD card. If there isn't
    one, calibrate right away. If that calibration fails, stop.
  * Main loop: whenever the Calibration button is pressed, calibrate again.
 
run_calibration() asks for a flat hand, then a fist, and checks that every
finger's reading changed enough between the two poses. If a finger didn't,
the wrist display names it and the user tries again, up to three times,
before the display shows FAULT.
 
Prerequisites:
Install numpy (a python library) with following commands:
    use py --version to make sure py is installed
    py -m pip install numpy
     
How to run:
    py glove_calibration.py           # on the Pi, with the real glove
                                           OR
    py glove_calibration.py --mock    # on any computer, with a pretend glove (mock_hardware.py)

How to test if a finger is significantly out of place during calibration:
    py glove_calibration.py --mock --stuck-finger [Thumb, Index, Middle, Ring, or Pinky]
"""
 
import argparse
import json
import os
import threading
import time
from collections import deque
from pathlib import Path
 
import numpy as np
 
# The Raspberry Pi libraries are only needed for the real glove. When they
# aren't installed (on a laptop, say), --mock mode and the tests still work.
try:
    import spidev                                   # SPI: talks to the MCP3208 ADC
    from gpiozero import Button                     # GPIO: the pushbuttons
    from luma.core.interface.serial import i2c      # I2C: talks to the wrist OLED
    from luma.core.render import canvas
    from luma.oled.device import ssd1306
    from PIL import ImageFont
    PI_LIBRARIES_FOUND = True
except ImportError:
    PI_LIBRARIES_FOUND = False
 
 
# =============================================================================
# Settings: everything you might need to change during bring-up is here
# =============================================================================
 
# Which finger is on which ADC channel: CH0 = Thumb ... CH4 = Pinky.
# If the glove is wired in a different order, reorder this list to match.
FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
 
# Pins are BCM GPIO numbers, not physical pin numbers.
CALIBRATION_BUTTON_PIN = 25   # physical pin 22 (GPIO17 is now the LCD's /CS1)
BUTTON_PULL_UP = True         # True: button wired between the pin and GND (no resistor needed)
                              # False: button wired between the pin and 3.3 V

# Wrist LCD (Winstar, KS0108), wired straight to the Pi
LCD_CS1_PIN = 17              # physical pin 11, /CS1: left half of the screen
LCD_CS2_PIN = 27              # physical pin 13, /CS2: right half
LCD_CS_ACTIVE_LOW = True      # R9/R10 pull /CS1 and /CS2 up, so a low pin selects that half
LCD_DI_PIN = 22               # physical pin 15, D/I: low for a command, high for pixels
LCD_E_PIN = 23                # physical pin 16, E: the LCD reads the byte as this goes low
LCD_RST_PIN = 24              # physical pin 18, RST: low holds the LCD in reset
LCD_DATA_PINS = [5, 6, 12, 13, 16, 19, 20, 21]   # DB0-DB7 (physical pins 29, 31, 32, 33, 36, 35, 38, 40)
SPI_SPEED_HZ = 1_000_000      # the MCP3208 allows up to 2 MHz at 5 V; go lower if readings are noisy
 
# Sampling: all five sensors, 200 times per second
SAMPLE_HZ = 200
WINDOW = 100                          # readings kept: the newest 100, which is 0.5 second
AVERAGE_SECONDS = WINDOW / SAMPLE_HZ  # so each pose is averaged over 0.5 second
LOOP_HZ = 20                          # the main loop runs 20 times per second
 
# Calibration
POSE_SETTLE_SECONDS = 2.0   # time the user gets to form each pose
MIN_RANGE = 200             # every finger must change by at least this many ADC counts
                            # between FLAT and FIST (200 counts is about 0.24 V). Tune it
                            # once you see real readings; the thumb usually moves least.
MAX_CAL_TRIES = 3
MESSAGE_SECONDS = 2.0       # how long result messages stay on screen
 
# calibration.json is saved next to this file, on the Pi's microSD card.
CAL_FILE = Path(__file__).with_name("calibration.json")
MOCK_CAL_FILE = Path(__file__).with_name("calibration_mock.json")  # --mock never touches the real file
 
 
# =============================================================================
# Hardware: the ADC, the wrist display, and the background sampler
# =============================================================================
 
class MCP3208:
    """
    The 12-bit ADC that turns each flex sensor's voltage into a number (0-4095).
 
    The Pi can't read analog voltages itself, so each finger's voltage divider
    goes to one ADC channel, and the Pi asks the ADC for readings over SPI.
    """
 
    def __init__(self, bus=0, device=0):
        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)            # SPI0, chip select CE0 (GPIO8)
        self.spi.max_speed_hz = SPI_SPEED_HZ
        self.spi.mode = 0                     # SPI mode 0, which the MCP3208 supports
 
    def read(self, channel):
        """Return one reading (0-4095) from one channel (0-7)."""
        # Build the 3-byte request that tells the chip which input to read.
        command = [0x06 | (channel >> 2), (channel & 0x03) << 6, 0x00]
        # Send the request. The chip sends 3 bytes back at the same time.
        reply = self.spi.xfer2(command)
        # The reading is 12 bits: the last 4 bits of the second byte
        # plus all 8 bits of the third byte. Join them into one number.
        return ((reply[1] & 0x0F) << 8) | reply[2]
 
    def close(self):
        self.spi.close()
 
 
class WristDisplay:
    """The Winstar 128x64 graphic LCD on the wrist (KS0108, 8-bit parallel).

    The screen is two 64x64 halves: /CS1 selects the left half and /CS2 the right.
    Each message has a title line and a detail line, like "CALIBRATING" and
    "Hold hand FLAT". Messages are also printed in the terminal, which helps
    while debugging and keeps things working before the LCD is wired up.
    """

    def __init__(self):
        try:
            self.chip = lgpio.gpiochip_open(0)
            deselect = 1 if LCD_CS_ACTIVE_LOW else 0
            lgpio.gpio_claim_output(self.chip, LCD_CS1_PIN, deselect)
            lgpio.gpio_claim_output(self.chip, LCD_CS2_PIN, deselect)
            for pin in [LCD_DI_PIN, LCD_E_PIN, LCD_RST_PIN]:
                lgpio.gpio_claim_output(self.chip, pin, 0)
            lgpio.group_claim_output(self.chip, LCD_DATA_PINS)   # DB0-DB7, written as one byte
            time.sleep(0.001)                 # RST starts low, which resets the LCD...
            lgpio.gpio_write(self.chip, LCD_RST_PIN, 1)          # ...then let it run (R21 holds it low otherwise)
            time.sleep(0.01)
            for half in (0, 1):
                self.select(half)
                self.write(0x3F, data=False)  # display on
                self.write(0xC0, data=False)  # the top of the screen is RAM line 0
        except Exception as error:            # not wired yet, pin already in use, lgpio missing...
            print(f"Wrist LCD not found ({error}). Messages will only be printed here.")
            self.chip = None
        self.title_font = load_font(15)
        self.detail_font = load_font(14)

    def message(self, title, detail=""):
        print(f"[LCD] {title:<12} {detail}".rstrip())
        if self.chip is None:
            return
        image = Image.new("1", (128, 64))     # draw the new screen in memory first
        draw = ImageDraw.Draw(image)
        draw.text((0, 4), title, font=self.title_font, fill=1)
        draw.text((0, 34), detail, font=self.detail_font, fill=1)
        # The LCD stores the screen as 8 pages of 8 pixel rows. Each byte it
        # receives is one column of a page, with the top pixel in bit 0.
        pixels = np.array(image, dtype=np.uint8).reshape(8, 8, 128)
        pages = np.packbits(pixels, axis=1, bitorder="little")[:, 0, :]
        for half in (0, 1):
            self.select(half)
            for page in range(8):
                self.write(0xB8 | page, data=False)          # go to this page
                self.write(0x40, data=False)                 # and to column 0
                for byte in pages[page, half * 64:(half + 1) * 64]:
                    self.write(int(byte), data=True)         # the column moves right by itself

    def select(self, half):
        """Talk to the left half (0) or the right half (1) of the screen."""
        on, off = (0, 1) if LCD_CS_ACTIVE_LOW else (1, 0)
        lgpio.gpio_write(self.chip, LCD_CS1_PIN, on if half == 0 else off)
        lgpio.gpio_write(self.chip, LCD_CS2_PIN, on if half == 1 else off)

    def write(self, value, data):
        """Send one byte: a command (data=False) or 8 pixels (data=True)."""
        lgpio.gpio_write(self.chip, LCD_DI_PIN, 1 if data else 0)
        lgpio.group_write(self.chip, LCD_DATA_PINS[0], value)   # bit 0 to DB0 ... bit 7 to DB7
        lgpio.gpio_write(self.chip, LCD_E_PIN, 1)
        lgpio.gpio_write(self.chip, LCD_E_PIN, 0)               # the LCD reads the byte as E goes low

 
 
def load_font(size):
    """Use a clear TrueType font if the Pi has one, otherwise Pillow's built-in font."""
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()
 
 
class Sampler:
    """Reads all five sensors 200 times per second in a background thread.
 
    The newest WINDOW readings (half a second) are kept in self.window.
    Calibration averages them, and later the sign recognizer will read them too.
    """
 
    def __init__(self, adc):
        self.adc = adc
        self.window = deque(maxlen=WINDOW)    # the oldest reading drops off automatically
        self.lock = threading.Lock()          # stops two threads using the window at once
        self.running = False
        self.thread = threading.Thread(target=self.run, daemon=True)
 
    def start(self):
        self.running = True
        self.thread.start()
 
    def stop(self):
        self.running = False
        self.thread.join()
 
    def run(self):
        period = 1.0 / SAMPLE_HZ           # 5 ms
        next_time = time.monotonic()       # when the next reading is due
        while self.running:
            # Read every finger's sensor once.
            sample = [self.adc.read(channel) for channel in range(len(FINGERS))]
            # Save the readings. The lock stops other code from
            # using the window while we add to it.
            with self.lock:
                self.window.append(sample)
            # Wait until the next reading is due. Waiting for a set clock
            # time instead of a flat 5 ms keeps the loop from slowly falling behind.
            next_time += period
            time.sleep(max(0.0, next_time - time.monotonic()))
 
    def average(self):
        """Return each finger's average reading over the window (five numbers)."""
        with self.lock:
            readings = np.array(self.window, dtype=float)
        if len(readings) == 0:
            raise RuntimeError("No sensor readings yet. Was sampler.start() called?")
        return readings.mean(axis=0)
 
 
# =============================================================================
# Step 2: Calibration (the run_calibration() flowchart)
# =============================================================================
 
def record_pose(sampler, display, prompt):
    """Show a prompt, give the user time to hold the pose, and return each finger's average."""
    display.message("CALIBRATING", prompt)
    time.sleep(POSE_SETTLE_SECONDS)    # "Wait 2 seconds" while the user forms the pose,
    time.sleep(AVERAGE_SECONDS)        # then let 0.5 s of fresh readings fill the window
    return sampler.average()           # and "average each finger over 0.5 second"
 
 
def run_calibration(sampler, display, cal_file=CAL_FILE):
    """
    Record the flat-hand and closed-fist readings and save them.
 
    Returns (flat, fist) on success, or None after three failed attempts.
    Each is an array of five numbers, one per finger.
    """
    for attempt in range(1, MAX_CAL_TRIES + 1):
        flat = record_pose(sampler, display, "Hold hand FLAT")
        fist = record_pose(sampler, display, "Make a FIST")
 
        # How much did each finger's reading change between the two poses?
        # A finger that barely changed means its sensor slipped out of place,
        # a wire is loose, or the pose wasn't held.
        change = np.abs(fist - flat)
        print_readings(attempt, flat, fist, change)
 
        if np.all(change >= MIN_RANGE):
            # Every finger changed enough: save the readings and finish.
            save_calibration(flat, fist, cal_file)
            display.message("Calibration", "saved")
            time.sleep(MESSAGE_SECONDS)
            return flat, fist
 
        # Name the finger that changed the least, then go back to "Hold hand FLAT".
        weakest = FINGERS[int(np.argmin(change))]
        display.message(f"Check {weakest}", f"Try {attempt} of {MAX_CAL_TRIES}")
        time.sleep(MESSAGE_SECONDS)
 
    # That was the third failed attempt.
    display.message("FAULT", "Check wiring")
    return None
 
 
def save_calibration(flat, fist, cal_file=CAL_FILE):
    """Save both sets of readings to calibration.json.
 
    The data goes into a temporary file first, which then replaces the old
    file in one step. That way a power cut mid-save can't leave behind a
    half-written file that breaks the next start-up.
    """
    data = {
        "fingers": FINGERS,
        "flat": np.round(flat, 1).tolist(),
        "fist": np.round(fist, 1).tolist(),
    }
    temp_file = cal_file.with_suffix(".tmp")
    with open(temp_file, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())              # make sure the data is really on the card
    os.replace(temp_file, cal_file)       # swap in the new file, all or nothing
 
 
def load_calibration(cal_file=CAL_FILE):
    """
    Return the saved (flat, fist) readings, or None if there's no usable file.
 
    This is the start-up check "Calibration saved on microSD?".
    """
    try:
        with open(cal_file) as f:
            data = json.load(f)
        flat = np.array(data["flat"], dtype=float)
        fist = np.array(data["fist"], dtype=float)
    except (OSError, ValueError, KeyError, TypeError):
        return None                       # no file yet, or it's damaged: "not calibrated"
 
    if flat.shape != (len(FINGERS),) or fist.shape != (len(FINGERS),):
        return None                       # saved for a different number of sensors
    return flat, fist
 
 
def print_readings(attempt, flat, fist, change):
    """Print the raw numbers in the terminal. Handy for checking wiring and tuning MIN_RANGE."""
    print()
    print(f"Attempt {attempt}".ljust(10) + "".join(f"{name:>8}" for name in FINGERS))
    for label, values in [("flat", flat), ("fist", fist), ("change", change)]:
        print(f"  {label:<8}" + "".join(f"{value:8.0f}" for value in values))
    print(f"  (every finger needs a change of at least {MIN_RANGE})")
 
 
# =============================================================================
# Main program: start-up (flowchart sheet 1) and the Calibration button (sheet 2)
# =============================================================================
 
def main():
    args = parse_args()
    if not args.mock and not PI_LIBRARIES_FOUND:
        raise SystemExit("Raspberry Pi libraries not found. Install them (see README.md), "
                         "or try the pretend glove:  python3 glove_calibration.py --mock")
 
    adc, display, calibration_button = build_hardware(args)
    cal_file = MOCK_CAL_FILE if args.mock else CAL_FILE
    sampler = Sampler(adc)
    sampler.start()                       # start reading the five sensors 200 times per second
 
    try:
        # Sheet 1: use the saved calibration, or calibrate now if there isn't one.
        # calibration holds (flat, fist); normalize() will use it later.
        calibration = load_calibration(cal_file)
        if calibration is None:
            calibration = run_calibration(sampler, display, cal_file)
            if calibration is None:
                raise SystemExit("Calibration failed at start-up. Stopping.")
        display.message("READY")
 
        # Sheet 2: pressing the button only raises a flag, and the loop below
        # does the work. A 7-second calibration inside the button's callback
        # would hold up gpiozero's event thread, so callbacks stay tiny.
        calibration_requested = threading.Event()
        calibration_button.when_pressed = calibration_requested.set
        print("Press the Calibration button to re-calibrate (Enter in --mock mode). Ctrl+C quits.")
 
        while True:
            if calibration_requested.is_set():                 # "Calibration button pressed?"
                new_calibration = run_calibration(sampler, display, cal_file)
                if new_calibration is not None:
                    calibration = new_calibration
                    display.message("READY")
                else:
                    print("Re-calibration failed. Still using the previous calibration.")
                calibration_requested.clear()                  # ignore presses made while calibrating
 
            # Later: the Selection button, normalize(), recognize() and evaluate() go here.
            time.sleep(1 / LOOP_HZ)
 
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        sampler.stop()
        adc.close()
 
 
def parse_args():
    """
    Adds arguments to call a fake glove (--mock, when running on a laptop)
    """
    parser = argparse.ArgumentParser(description="ASL glove calibration")
    parser.add_argument("--mock", action="store_true",
                        help="use a pretend glove, so no Raspberry Pi hardware is needed")
    parser.add_argument("--stuck-finger", choices=FINGERS,
                        help="with --mock: pretend this finger's sensor is loose, "
                             "to see the retry and FAULT steps")
    args = parser.parse_args()
    if args.stuck_finger and not args.mock:
        parser.error("--stuck-finger only works together with --mock")
    return args
 
 
def build_hardware(args):
    """
    Create the ADC, the wrist display and the Calibration button (real or pretend).
    """
    if args.mock:
        from mock_hardware import MockButton, MockMCP3208, MockOLED, PretendHand
        stuck = FINGERS.index(args.stuck_finger) if args.stuck_finger else None
        hand = PretendHand(stuck_channel=stuck)
        return MockMCP3208(hand), MockOLED(hand), MockButton()
 
    return MCP3208(), WristDisplay(), Button(CALIBRATION_BUTTON_PIN, pull_up=BUTTON_PULL_UP)
 
 
if __name__ == "__main__":
    main()
 