"""
The ASL fingerspelling alphabet, as the glove's five flex sensors see it.

Each finger reads 0.0 (straight) to 1.0 (fully bent), Thumb to Pinky. A flex
sensor measures how far a finger bends, and nothing else. The gyroscope (IMU,
see imu.py) adds which way the hand points (tilt) and how much it is moving
(motion), which splits letters the flex sensors read the same (see
FLEX_LOOK_ALIKES) and makes J and Z possible.

  LETTERS          all 26 letters. All of them can be recorded and trained when the IMU is fitted.
  STATIC_LETTERS   the 24 letters held still: A to Y except J
  MOTION_LETTERS   J and Z are traced in the air. They need the IMU's motion reading: without it
                   they look like I and a pointing index finger.
  HOW_TO           a one-line reminder of each letter, shown while recording
  EXPECTED_POSES   rough guesses at each letter's bends (J and Z: the handshape while tracing)
  EXPECTED_TILT    rough guesses at which way is "up" for the glove during each letter
  EXPECTED_MOTION  rough guesses at how much the hand turns while tracing J and Z
  The EXPECTED_ guesses are only used by the pretend glove (--mock) and the tests;
  real recordings replace them.
"""

LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
MOTION_LETTERS = {
    "J": "the I handshape traced in a J (needs a motion sensor)",
    "Z": "the index finger traces a Z in the air (needs a motion sensor)",
}
STATIC_LETTERS = [letter for letter in LETTERS if letter not in MOTION_LETTERS]

HOW_TO = {
    "J": "Make an I (pinky up), then trace a J with the pinky: down, then hook toward you",
    "Z": "Point the index finger and trace a Z in the air: across, diagonal down, across",
    "A": "Fist, thumb straight up along the side of the index finger",
    "B": "Four fingers straight up together, thumb folded across the palm",
    "C": "Curve every finger and the thumb into a C",
    "D": "Index straight up, other fingertips curled to touch the thumb",
    "E": "Fingertips bent down to rest on the thumb, folded across the palm",
    "F": "Index and thumb tips touch in a circle, other three fingers up",
    "G": "Index and thumb straight and parallel, pointing sideways",
    "H": "Index and middle straight together, pointing sideways",
    "I": "Pinky straight up, thumb across the curled fingers",
    "K": "Index and middle up in a V, thumb up between them",
    "L": "Index straight up, thumb straight out: an L",
    "M": "Thumb tucked under the index, middle and ring fingers",
    "N": "Thumb tucked under the index and middle fingers",
    "O": "Every fingertip touches the thumb in an O",
    "P": "Like K, but pointing down",
    "Q": "Like G, but pointing down",
    "R": "Index and middle up and crossed",
    "S": "Fist, thumb across the front of the fingers",
    "T": "Fist, thumb tucked between index and middle",
    "U": "Index and middle straight up together",
    "V": "Index and middle straight up, spread apart",
    "W": "Index, middle and ring straight up, spread",
    "X": "Index bent into a hook, other fingers curled",
    "Y": "Thumb and pinky straight out, other fingers curled",
}

# Thumb, Index, Middle, Ring, Pinky. Guesses, not measurements.
EXPECTED_POSES = {
    "A": [0.10, 1.00, 1.00, 1.00, 1.00], "B": [0.90, 0.00, 0.00, 0.00, 0.00],
    "C": [0.40, 0.50, 0.50, 0.50, 0.50], "D": [0.60, 0.00, 0.85, 0.85, 0.85],
    "E": [0.90, 0.75, 0.80, 0.80, 0.80], "F": [0.60, 0.70, 0.00, 0.00, 0.00],
    "G": [0.10, 0.10, 1.00, 1.00, 1.00], "H": [0.75, 0.05, 0.05, 1.00, 1.00],
    "I": [0.80, 1.00, 1.00, 1.00, 0.00], "K": [0.35, 0.00, 0.15, 1.00, 1.00],
    "L": [0.00, 0.00, 1.00, 1.00, 1.00], "M": [0.85, 0.90, 0.90, 0.90, 1.00],
    "N": [0.75, 0.90, 0.90, 1.00, 1.00], "O": [0.55, 0.70, 0.70, 0.70, 0.70],
    "P": [0.35, 0.05, 0.25, 1.00, 1.00], "Q": [0.15, 0.20, 1.00, 1.00, 1.00],
    "R": [0.80, 0.05, 0.05, 1.00, 1.00], "S": [0.60, 1.00, 1.00, 1.00, 1.00],
    "T": [0.65, 0.85, 1.00, 1.00, 1.00], "U": [0.80, 0.00, 0.00, 1.00, 1.00],
    "V": [0.80, 0.00, 0.00, 1.00, 1.00], "W": [0.80, 0.00, 0.00, 0.00, 1.00],
    "X": [0.70, 0.55, 1.00, 1.00, 1.00], "Y": [0.00, 1.00, 1.00, 1.00, 0.00],
    # Motion letters: the handshape held while tracing. J's is I's, Z's is a pointing index.
    "J": [0.80, 1.00, 1.00, 1.00, 0.00], "Z": [0.70, 0.00, 1.00, 1.00, 1.00],
}

# Which way is "up" for the glove, as (along the fingers, toward the thumb, out of the
# back of the hand). Fingers pointing up: (1, 0, 0). See IMU_AXES in settings.py.
UP, SIDEWAYS, DOWN = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-0.7, 0.0, 0.7)
EXPECTED_TILT = {letter: UP for letter in LETTERS}
EXPECTED_TILT.update({"G": SIDEWAYS, "H": SIDEWAYS, "P": DOWN, "Q": DOWN, "Z": (0.6, 0.0, 0.8)})

# How hard the hand turns while tracing, per axis (0 = still, 1 = MOTION_FULL_SCALE_DPS).
# Every static letter is (0, 0, 0): it is held still.
EXPECTED_MOTION = {letter: (0.0, 0.0, 0.0) for letter in LETTERS}
EXPECTED_MOTION.update({"J": (0.55, 0.15, 0.30),   # the wrist twists as the pinky hooks
                        "Z": (0.10, 0.45, 0.50)})  # the hand swings side to side and down

# Letters that differ only in something a flex sensor doesn't measure. Expect the
# trainer to group these; it decides from the recordings, so a group may split if
# your sensors do tell them apart, or new ones may appear.
FLEX_LOOK_ALIKES = [
    ("U", "V", "R", "only spread or crossing differs (no sensor measures it)"),
    ("U", "H", "same shape; H points sideways (the IMU's tilt should split them)"),
    ("G", "L", "Q", "same shape; only pointing direction differs (the IMU's tilt should split them)"),
    ("K", "P", "same shape; P points down (the IMU's tilt should split them)"),
    ("I", "J", "same shape; J is traced (the IMU's motion should split them)"),
    ("A", "S", "T", "M", "N", "all fists; only where the thumb sits differs"),
]


def describe_look_alike(group):
    """Why these letters might be confused, if the alphabet already expects it."""
    for *letters, reason in FLEX_LOOK_ALIKES:
        if len(set(group) & set(letters)) >= 2:
            return reason
    return "these bend the fingers about the same amount on this glove"
