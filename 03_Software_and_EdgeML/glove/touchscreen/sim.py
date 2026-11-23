"""
main.py --mock with the touchscreen: try the home screen on any computer.

  window  (main.py --mock)          the LCD in a window. Click it to touch it. Buttons (or
                                    letter keys) choose what the pretend hand signs.
  text    (main.py --mock --text)   the LCD drawn in the terminal, with typed commands.

The LCD is the glove's real WristDisplay driver writing to mock_glove's
FakeGlcdBus (a pretend KS0108), so the pixels are the bytes the Pi would send.
The pretend hand has a pretend gyroscope too: each letter points the way
alphabet.EXPECTED_TILT says, and J and Z are traced (EXPECTED_MOTION).
"""

import threading
import time
import traceback
from contextlib import ExitStack
from unittest import mock

import numpy as np

import settings
import sign_feedback
from alphabet import EXPECTED_MOTION, EXPECTED_POSES, EXPECTED_TILT, MOTION_LETTERS
from mock_glove import FakeGlcdBus, MockMCP3208, PretendHand, open_mock_imu

from .app import HomeApp, letter_count
from .lcd import TouchLcd
from .panel import MockTouch

LCD_LIGHT, LCD_DARK = (185, 204, 85), (31, 42, 14)      # yellow-green Winstar colours
SIM_CAL_FILE = settings.MOCK_CAL_FILE                   # the pretend glove never touches the real calibration


def pose_of(letter):
    """The bends the glove compares against (the model's or REFERENCES), else alphabet.py's guess."""
    try:
        return [float(b) for b in sign_feedback.reference_for(letter)]
    except KeyError:
        return list(EXPECTED_POSES[letter])


def hand_letters(imu):
    """The letters the pretend hand offers: the ones the glove can read right now."""
    model = sign_feedback.load_model()
    letters = model.letters() if model is not None else list(settings.REFERENCES)
    if imu is None:
        letters = [letter for letter in letters if letter not in MOTION_LETTERS]
    usable = []
    for letter in sorted(letters):
        try:
            pose_of(letter)
            usable.append(letter)
        except KeyError:
            pass
    return usable


class SimHand(PretendHand):
    """A pretend hand you control. During calibration it follows the prompts by itself;
    otherwise it holds the pose you pick: "flat", "fist" or a letter (J and Z are traced)."""

    def __init__(self, stuck_finger=None, unplugged_finger=None):
        super().__init__(stuck_finger, unplugged_finger)
        self.chosen = "flat"
        self.wrong_finger = False        # True: the index finger does the opposite

    def follow(self, title, detail):
        super().follow(title, detail)
        self._update_imu()

    def choose(self, pose):
        self.chosen = pose
        self._update_imu()
        print(f"[HAND] now making: {self.describe()}")

    def describe(self):
        tracing = " (tracing it)" if self.chosen in MOTION_LETTERS else ""
        return self.chosen + tracing + (" (index finger wrong)" if self.wrong_finger else "")

    def _update_imu(self):
        letter = None if self.calibrating else self.chosen
        self.up = EXPECTED_TILT.get(letter, (1.0, 0.0, 0.0))      # (calibration poses: PretendHand)
        self.motion = EXPECTED_MOTION.get(letter, (0.0, 0.0, 0.0))

    def pose(self):
        if self.calibrating:
            return self.calibration_pose
        if self.chosen == "flat":
            bends = [0.0] * len(settings.FINGERS)
        elif self.chosen == "fist":
            bends = [1.0] * len(settings.FINGERS)
        else:
            bends = pose_of(self.chosen)
        if self.wrong_finger:
            bends[1] = 0.0 if bends[1] >= 0.5 else 1.0
        return bends


def run_simulator(args, commands=None):
    """main.py --mock (window) or --mock --text. commands: typed lines for the tests, instead of the keyboard."""
    from PIL import Image, ImageDraw, ImageFont
    import hardware

    routine = settings.routine
    with ExitStack() as stack:
        bus = FakeGlcdBus()                              # a pretend KS0108 for the real driver to write to
        for name, value in [("lgpio", bus), ("Image", Image), ("ImageDraw", ImageDraw), ("ImageFont", ImageFont)]:
            stack.enter_context(mock.patch.object(routine, name, value, create=True))
        hand = SimHand(args.stuck_finger, args.unplugged_finger)
        lcd = TouchLcd(hardware.WristDisplay(), routine, on_message=hand.follow)
        touch = MockTouch()
        sampler = hardware.Sampler(MockMCP3208(hand))
        sampler.start()
        imu = open_mock_imu(hand) if settings.USE_IMU else None
        app = HomeApp(sampler, lcd, touch, SIM_CAL_FILE, imu)
        try:
            if args.text or commands is not None:
                run_text(app, lcd, bus, hand, touch, imu, commands)
            else:
                run_window(app, lcd, bus, hand, touch, imu)
        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            lcd.stop_animation()
            if imu is not None:
                imu.stop()
            sampler.stop()


def start_app_thread(app, on_error):
    def run():
        try:
            if app.start():
                app.run()
            else:
                print("[APP] stopped at start-up; the screen shows why")
        except Exception:
            traceback.print_exc()
            on_error()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def lcd_pixels(lcd, bus):
    """The screen as a 64x128 True/False array, read from the pretend KS0108."""
    with lcd.lock:
        pixels = bus.pixels()
        bus.log.clear()                                 # the bus logs every byte; don't let it grow forever
    return pixels


# Window mode -----------------------------------------------------------------------

def run_window(app, lcd, bus, hand, touch, imu, scale=5):
    try:
        import tkinter as tk
        from PIL import Image, ImageTk
    except ImportError as error:
        raise SystemExit(f"The window needs tkinter ({error}). Try the terminal instead:  main.py --mock --text")

    border = 14
    root = tk.Tk()
    root.title("ASL glove touchscreen - simulator")
    canvas = tk.Canvas(root, width=128 * scale + 2 * border, height=64 * scale + 2 * border,
                       bg="#2c2c2a", highlightthickness=0, cursor="hand2")
    canvas.pack(padx=10, pady=(10, 4))
    picture = canvas.create_image(border, border, anchor="nw")
    status = tk.StringVar(value="Starting... the pretend hand calibrates by itself.")
    tk.Label(root, textvariable=status, anchor="w", justify="left").pack(fill="x", padx=12)

    def on_click(event):
        x, y = (event.x - border) // scale, (event.y - border) // scale
        if 0 <= x < 128 and 0 <= y < 64:
            touch.tap(x, y)

    canvas.bind("<Button-1>", on_click)

    controls = tk.Frame(root)
    controls.pack(padx=10, pady=8)
    tk.Label(controls, text="Pretend hand:").grid(row=0, column=0, columnspan=4, sticky="w")
    poses = ["flat", "fist"] + hand_letters(imu)
    per_row = 10
    for i, pose in enumerate(poses):
        label = pose.capitalize() if len(pose) > 1 else pose
        tk.Button(controls, text=label, width=4, command=lambda p=pose: hand.choose(p)).grid(
            row=1 + i // per_row, column=i % per_row, padx=2, pady=2)
    wrong = tk.BooleanVar(value=False)

    def toggle_wrong():
        hand.wrong_finger = wrong.get()
        print(f"[HAND] now making: {hand.describe()}")

    tk.Checkbutton(controls, text="Index finger the wrong way", variable=wrong, command=toggle_wrong).grid(
        row=2 + len(poses) // per_row, column=0, columnspan=per_row, sticky="w")
    tk.Label(root, text="Click the screen to touch it.  Keys: a letter = sign it (hold 1 s; J and Z are "
                        "traced), space = flat hand, 1 = wrong finger on/off.", fg="#555").pack(pady=(0, 8))
    letters = set(poses)

    def on_key(event):
        key = event.char.upper()
        if key in letters:
            hand.choose(key)
        elif event.char == " ":
            hand.choose("flat")
        elif event.char == "1":
            wrong.set(not wrong.get())
            toggle_wrong()

    root.bind("<Key>", on_key)
    shown = {"bytes": None}
    failed = {"yes": False}

    def refresh():
        pixels = lcd_pixels(lcd, bus)
        if pixels.tobytes() != shown["bytes"]:
            shown["bytes"] = pixels.tobytes()
            rgb = np.where(pixels[..., None], LCD_DARK, LCD_LIGHT).astype(np.uint8)
            image = Image.fromarray(rgb).resize((128 * scale, 64 * scale), Image.NEAREST)
            canvas.photo = ImageTk.PhotoImage(image)    # keep a reference, or Tk drops the picture
            canvas.itemconfig(picture, image=canvas.photo)
        gyro = ""
        if imu is not None:
            tilt, motion = imu.state()
            gyro = (f"\nGyroscope: up = ({tilt[0]:+.1f}, {tilt[1]:+.1f}, {tilt[2]:+.1f})   "
                    f"motion = {max(motion):.2f}" + ("  (moving)" if max(motion) >= settings.MOTION_START else ""))
        status.set(("The program crashed: see the terminal.\n" if failed["yes"] else "")
                   + f"Screen: {lcd.last_summary or ''}\nHand: {hand.describe()}{gyro}")
        root.after(60, refresh)

    start_app_thread(app, lambda: failed.update(yes=True))
    refresh()
    root.mainloop()


# Text mode ---------------------------------------------------------------------------

TEXT_HELP = """Commands (then Enter):
  a button on the screen   free, lesson, recal, back, menu
  tap X Y                  touch pixel X (0-127), Y (0-63)
  a letter, flat or fist   what the pretend hand makes, e.g.  b  (j and z are traced)
  wrong                    index finger the wrong way, on/off
  show                     draw the LCD here
  quit                     stop"""


def run_text(app, lcd, bus, hand, touch, imu, commands=None):
    start_app_thread(app, lambda: None)
    print(TEXT_HELP)
    letters = set(hand_letters(imu))
    lines = iter(commands) if commands is not None else None
    while True:
        try:
            line = next(lines) if lines is not None else input()
        except (EOFError, StopIteration):
            return
        words = line.strip().split()
        if not words:
            continue
        word = words[0].lower()
        if word == "quit":
            return
        if word == "sleep" and len(words) == 2:           # for scripted runs
            time.sleep(float(words[1]))
        elif word == "show":
            print(render_text(lcd_pixels(lcd, bus)))
        elif word == "help":
            print(TEXT_HELP)
        elif word == "wrong":
            hand.wrong_finger = not hand.wrong_finger
            print(f"[HAND] now making: {hand.describe()}")
        elif word in ("flat", "fist"):
            hand.choose(word)
        elif word.upper() in letters:
            hand.choose(word.upper())
        elif word == "tap" and len(words) == 3:
            touch.tap(int(words[1]), int(words[2]))
        else:
            screen = lcd.current
            box = screen.buttons.get(word) if screen else None
            if box:
                touch.tap((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
            else:
                names = ", ".join(screen.buttons) if screen and screen.buttons else "none on this screen"
                print(f"Don't know '{line.strip()}'. Buttons here: {names}. Type help for the list.")


def render_text(pixels):
    """The LCD in the terminal: each character is 2 pixel rows (top, bottom)."""
    blocks = {(False, False): " ", (True, False): "▀", (False, True): "▄", (True, True): "█"}
    lines = ["".join(blocks[(bool(t), bool(b))] for t, b in zip(pixels[r], pixels[r + 1]))
             for r in range(0, 64, 2)]
    edge = "+" + "-" * 128 + "+"
    return "\n".join([edge] + [f"|{line}|" for line in lines] + [edge])


__all__ = ["SimHand", "run_simulator", "letter_count"]
