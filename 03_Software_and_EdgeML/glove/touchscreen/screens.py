"""
The screens the Winstar shows, drawn as 128x64 black-and-white pictures:
the home screen (Free / Lesson / Recalibrate), free mode, the lesson list, lessons,
calibration and messages.

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
def lesson(word, index, feedback):
    """A lesson: the word across the top, the letter being signed boxed, and the feedback.

    word: the letters to sign ("HELLO"). index: which one is being signed now (0 = the first).
    feedback: CORRECT!, HOLD STEADY, ADJUST: [finger], TRY AGAIN, or a reminder to sign a letter again.
    Letters already signed are filled in; the current one is boxed. Long words (the alphabet)
    show the 9 letters around the current one.
    """
    image, draw = new_screen()
    title_bar(draw, f"SPELL: {word}")
    start = max(0, min(index - 4, len(word) - 9))         # at most 9 letters fit across the screen
    shown = word[start:start + 9]
    cell = min(18, (WIDTH - 4) // len(shown))
    left = (WIDTH - cell * len(shown)) // 2
    for i, letter in enumerate(shown):
        box = (left + i * cell + 1, 14, left + (i + 1) * cell - 1, 29)
        if start + i < index:                             # done: filled in
            draw.rectangle(box, fill=ON)
            text_in(draw, letter, box, 12, fill=OFF)
        else:
            if start + i == index:                        # signing now: boxed
                draw.rectangle(box, outline=ON)
            text_in(draw, letter, box, 12)
    text_in(draw, feedback, (2, 31, WIDTH - 3, 45), 12)
    buttons = {"menu": (1, 47, 42, 63), "skip": (85, 47, 126, 63)}
    button(draw, buttons["menu"], "Menu")
    button(draw, buttons["skip"], "Skip")
    return Screen(image, buttons, f"Lesson {word}, letter {index + 1} ({word[index]}): {feedback}  [Menu] [Skip]")


@lru_cache(maxsize=64)
def lesson_done(word):
    """The end of a lesson: every letter signed."""
    image, draw = new_screen()
    title_bar(draw, f"SPELL: {word}")
    text_in(draw, "Complete!", (2, 15, WIDTH - 3, 40), 14)
    buttons = {"menu": (1, 45, 60, 61), "again": (67, 45, 126, 61)}
    button(draw, buttons["menu"], "Menu")
    button(draw, buttons["again"], "Again")
    return Screen(image, buttons, f"Lesson {word}: complete  [Menu] [Again]")


ROWS = 3                                           # lessons per page: 3 rows of finger-sized buttons fit


@lru_cache(maxsize=64)
def lessons(names, top=0):
    """The lesson list: [Back], then one button per lesson (settings.LESSONS), 3 at a time.

    names: the lesson names, as a tuple. top: the first one shown (0, 3, 6...).
    With more than 3 lessons, up and down arrows scroll; a hollow arrow can't go further that way.
    Each lesson's button is named after it in lowercase ("hello"), so the simulator can type it.
    """
    image, draw = new_screen()
    buttons = {"back": (0, 0, 40, 14)}
    button(draw, buttons["back"], "Back")
    text_in(draw, "LESSONS", (44, 0, WIDTH - 1, 14), 11)
    draw.line((0, 15, WIDTH - 1, 15), fill=ON)
    scrolls = len(names) > ROWS
    right = 102 if scrolls else WIDTH - 1              # leave room for the arrows
    shown = names[top:top + ROWS]
    for i, name in enumerate(shown):
        y0 = 17 + i * 16
        buttons[name.lower()] = (0, y0, right, y0 + 14)
        button(draw, buttons[name.lower()], name, 11)
    if scrolls:
        for name, box, can_move in [("up", (105, 17, 127, 39), top > 0),
                                    ("down", (105, 41, 127, 63), top + ROWS < len(names))]:
            x0, y0, x1, y1 = box
            draw.rounded_rectangle(box, radius=2, outline=ON)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            tip = -5 if name == "up" else 5
            draw.polygon([(cx, cy + tip), (cx - 6, cy - tip * 4 // 5), (cx + 6, cy - tip * 4 // 5)],
                         fill=ON if can_move else None, outline=ON)
            if can_move:
                buttons[name] = box
    arrows = " [up] [down]" if scrolls else ""
    return Screen(image, buttons, "Lessons: [Back] " + " ".join(f"[{n}]" for n in shown) + arrows)


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
