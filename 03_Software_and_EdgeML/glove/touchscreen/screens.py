"""
The screens the Winstar shows, drawn as 128x64 black-and-white pictures:
the home screen (Free / Lesson / Recalibrate), free mode, calibration and messages.

Each function returns a Screen: the picture, the touch buttons on it (name ->
box), and a one-line summary that gets printed in the terminal. Nothing here
touches the hardware, so the same pictures appear on the laptop and the Pi.
"""

from dataclasses import dataclass, field
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 128, 64
ON, OFF = 1, 0                                     # 1 = a dark pixel on the LCD

# Fonts: DejaVu on the Pi (README installs it), Arial on Windows, Liberation on Linux.
FONT_FILES = ["DejaVuSans-Bold.ttf", "arialbd.ttf", "LiberationSans-Bold.ttf", "Arial Bold.ttf"]
_fonts = {}


@dataclass
class Screen:
    image: Image.Image
    buttons: dict = field(default_factory=dict)    # name -> (x0, y0, x1, y1), edges included
    summary: str = ""

    def button_at(self, x, y):
        """The name of the button under (x, y), or None."""
        for name, (x0, y0, x1, y1) in self.buttons.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                return name
        return None


def font(size):
    """A bold font at this pixel size. Cached, because loading a font is slow on a Pi Zero."""
    if size not in _fonts:
        for name in FONT_FILES:
            try:
                _fonts[size] = ImageFont.truetype(name, size)
                break
            except OSError:
                continue
        else:
            try:
                _fonts[size] = ImageFont.load_default(size)   # Pillow 10.1 and newer
            except TypeError:
                _fonts[size] = ImageFont.load_default()
    return _fonts[size]


# Drawing helpers -----------------------------------------------------------------

def new_screen(inverted=False):
    image = Image.new("1", (WIDTH, HEIGHT), ON if inverted else OFF)
    return image, ImageDraw.Draw(image)


def text_in(draw, text, box, size, fill=ON):
    """Draw text centred in box. If it doesn't fit, use a smaller size until it does."""
    x0, y0, x1, y1 = box

    def fits(t, f):
        left, top, right, bottom = draw.textbbox((0, 0), t, font=f)
        return right - left <= x1 - x0 - 1 and bottom - top <= y1 - y0 + 1

    for s in range(size, 7, -1):
        f = font(s)
        if fits(text, f):
            break
    while len(text) > 1 and not fits(text, f):            # still too long at the smallest size: cut it
        text = text[:-2].rstrip() + "\u2026"
    left, top, right, bottom = draw.textbbox((0, 0), text, font=f)
    x = x0 + (x1 - x0 + 1 - (right - left)) // 2 - left
    y = y0 + (y1 - y0 + 1 - (bottom - top)) // 2 - top
    draw.text((x, y), text, font=f, fill=fill)


def button(draw, box, label, size=10):
    draw.rounded_rectangle(box, radius=2, outline=ON, width=1)
    x0, y0, x1, y1 = box
    text_in(draw, label, (x0 + 2, y0 + 1, x1 - 2, y1 - 1), size)


def title_bar(draw, text):
    draw.rectangle((0, 0, WIDTH - 1, 11), fill=ON)
    text_in(draw, text, (0, 1, WIDTH - 1, 10), 10, fill=OFF)


# The screens ----------------------------------------------------------------------

@lru_cache(maxsize=64)                 # the same screen 20 times a second: draw it once
def home(letter_count):
    """The first screen: [Free] [Lesson] on top, [Recalibrate] across the bottom."""
    image, draw = new_screen()
    letters = "1 letter" if letter_count == 1 else f"{letter_count} letters"
    title_bar(draw, f"ASL GLOVE  -  {letters}")
    buttons = {"free": (5, 16, 60, 40), "lesson": (67, 16, 122, 40), "recal": (5, 45, 122, 61)}
    button(draw, buttons["free"], "Free", 12)
    button(draw, buttons["lesson"], "Lesson", 12)
    button(draw, buttons["recal"], "Recalibrate")
    return Screen(image, buttons, f"Home ({letters}): [Free] [Lesson] [Recalibrate]")


@lru_cache(maxsize=64)
def free(letter):
    """Free mode: the last letter signed (or a look-alike group like U/V), or -- before the first."""
    image, draw = new_screen()
    text_in(draw, "Free mode", (2, 0, 70, 11), 10)
    text_in(draw, letter or "--", (24, 13, 103, 45), 34)
    buttons = {"menu": (1, 47, 42, 63), "recal": (70, 47, 126, 63)}
    button(draw, buttons["menu"], "Menu")
    button(draw, buttons["recal"], "Recal")
    return Screen(image, buttons, f"Free mode: {letter or '--'}  [Menu] [Recal]")


@lru_cache(maxsize=64)
def coming_soon():
    """What the Lesson button shows until lessons are built."""
    image, draw = new_screen()
    title_bar(draw, "LESSONS")
    text_in(draw, "Coming soon", (2, 15, WIDTH - 3, 40), 14)
    buttons = {"back": (37, 45, 90, 61)}
    button(draw, buttons["back"], "Back")
    return Screen(image, buttons, "Lessons: coming soon  [Back]")


def calibrating(prompt, step, progress, total=2):
    """step: 1 flat, 2 fist, 3 fingers up, 4 thumb up (total is 4 with the gyroscope, else 2).
    progress runs from 0.0 to 1.0 while the pose is held."""
    image, draw = new_screen()
    title_bar(draw, "CALIBRATING")
    text_in(draw, prompt, (0, 14, WIDTH - 1, 28), 12)
    draw.rectangle((12, 32, 115, 42), outline=ON)
    filled = round(100 * max(0.0, min(1.0, progress)))
    if filled:
        draw.rectangle((14, 34, 13 + filled, 40), fill=ON)
    text_in(draw, f"Step {step} of {total}", (0, 48, WIDTH - 1, 61), 10)
    return Screen(image, {}, f"CALIBRATING  {prompt}  (step {step} of {total})")


@lru_cache(maxsize=64)
def message(title, detail=""):
    """Any other glove message: Calibration saved, Check Ring, FAULT, SENSOR FAULT, READY..."""
    image, draw = new_screen()
    title_bar(draw, title)
    if detail:
        text_in(draw, detail, (2, 16, WIDTH - 3, 60), 14)
    return Screen(image, {}, f"{title}  {detail}".strip())


def touch_test(point, raw):
    """Bring-up: a cross where the panel says you touched, and the raw readings."""
    image, draw = new_screen()
    for x, y in [(0, 0), (WIDTH - 4, 0), (0, HEIGHT - 4), (WIDTH - 4, HEIGHT - 4)]:
        draw.rectangle((x, y, x + 3, y + 3), fill=ON)        # touch these four corners
    if raw is None:
        text_in(draw, "Touch the corners", (6, 20, WIDTH - 7, 42), 11)
        return Screen(image, {}, "Touch test: not touched")
    x, y = point
    draw.line((x - 6, y, x + 6, y), fill=ON)
    draw.line((x, y - 6, x, y + 6), fill=ON)
    text_in(draw, f"raw {raw[0]},{raw[1]}", (6, 0, WIDTH - 7, 11), 10)
    return Screen(image, {}, f"Touch test: raw x {raw[0]}, raw y {raw[1]} -> pixel {x}, {y}")
