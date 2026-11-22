"""
Collect training data for the ASL letter network.

Each person who wears the glove runs this once. It calibrates their hand, then
asks for each letter several times, in a shuffled order. With the gyroscope
that is all 26 letters; without it, the 24 still letters (J and Z are traced, so
they need its motion reading). The terminal shows how to form each one:

    still letters:   SIGN A  1/5   "Make the sign"  ->  "Hold still"  ->  saved  ->  "Relax hand"
    J and Z:         SIGN J  1/5   "Trace it now"   ->  "Keep tracing" ->  saved  ->  "Relax hand"

Each saved sign is 1 second of readings (20 frames). For J and Z, make the
handshape, then trace the letter (again and again) until it says Saved.
Relaxing between tries matters: the network learns from how differently you
form the same letter each time, not from one long hold. At the end it records
NONE: 20 seconds of moving your hand around freely, so the network learns what
"not a letter" looks like.

Every frame has the five finger bends and, with the gyroscope, which way the
hand points (tilt) and how fast it is turning (motion), plus the raw readings
of both sensors (see COLUMNS).

Everything is written to data/<person>_<date>_<time>.csv as it is recorded,
so Ctrl+C keeps every sign saved so far. Run it again any time to add more.

A full run (26 letters x 5 tries) takes about 9 minutes. Recording in two
sittings is fine: every CSV in data/ is used.

Run from this folder (on the Pi, with the glove on), once for each of the team
(settings.TEAM: zion, amad, milly, sleytters, chris):
  python3 collect_data.py --person zion
  python3 collect_data.py --person amad
  python3 collect_data.py --person milly
  python3 collect_data.py --person sleytters
  python3 collect_data.py --person chris
  python3 collect_data.py --person test --mock          a pretend glove (any computer), saved in data/mock/
"""

import argparse
import csv
import random
import sys
import time
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np

import settings
from alphabet import EXPECTED_MOTION, EXPECTED_POSES, EXPECTED_TILT, HOW_TO, LETTERS, MOTION_LETTERS, STATIC_LETTERS
from hardware import PI_LIBRARIES_FOUND, Sampler, open_hardware
from imu import IMU_FEATURE_NAMES, is_moving
from main import start_up
from sign_feedback import normalize
from startup_check import find_faulty_finger

DATA_DIR = Path(__file__).with_name("data")

MOVE = 0.25              # a finger must move 25% of its range: the user has started the sign
STILL = 0.08             # every finger within 8% of its range for the whole second: held still
FRAMES = 20              # frames saved per sign: 1 second at LOOP_HZ (20 per second)
TRACE_SHARE = 0.6        # J and Z: at least 60% of the saved second must be moving
RELAX_SECONDS = 1.5      # "Relax hand" between tries
NONE_SECONDS = 20        # default length of the NONE recording

COLUMNS = (["person", "session", "letter", "rep", "frame"]
           + [f"bend_{f.lower()}" for f in settings.FINGERS]       # 0.0 flat .. 1.0 fist: what the network uses
           + IMU_FEATURE_NAMES                                     # tilt and motion: the network uses these too
           + [f"raw_{f.lower()}" for f in settings.FINGERS]        # ADC counts, kept so the data can be re-used
           + ["accel_along", "accel_thumb", "accel_back",          # raw gyroscope board readings, in g
              "gyro_along", "gyro_thumb", "gyro_back"]             # and degrees per second
           + [f"flat_{f.lower()}" for f in settings.FINGERS]       # this session's calibration
           + [f"fist_{f.lower()}" for f in settings.FINGERS])


def main():
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    person = args.person.strip().lower().replace(" ", "_")
    team = [name.lower() for name in getattr(settings, "TEAM", [])]
    if team and person not in team:
        print(f"WARNING: '{person}' isn't on the team list ({', '.join(team)}).\n"
              f"         If that's a typo, press Ctrl+C now: a misspelled name counts as a different person.\n"
              f"         Recording a guest on purpose is fine; add them to TEAM in settings.py.\n")

    if args.mock:
        adc, display, hand = open_mock_collector(person)
        imu = None
        if settings.USE_IMU:
            from mock_glove import open_mock_imu
            imu = open_mock_imu(hand)
        cal_file = settings.MOCK_CAL_FILE
    elif PI_LIBRARIES_FOUND:
        from imu import open_imu
        adc, display = open_hardware()
        imu = open_imu()
        cal_file = settings.CAL_FILE
    else:
        raise SystemExit("Raspberry Pi libraries not found. Run this on the Pi, or add --mock.")

    letters = choose_letters(args.letters, imu)
    if imu is None:
        print("[IMU] No gyroscope: recording the 5 finger bends only. Tilt and motion columns stay empty,\n"
              "      and J and Z are skipped. Fix the gyroscope first if you can: the network needs it\n"
              "      to tell G from L, K from P, H from U, and to read J and Z at all.")

    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = DATA_DIR / "mock" if args.mock else DATA_DIR      # pretend-glove data never mixes with real data
    path = folder / f"{person}_{session}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    sampler = Sampler(adc)
    sampler.start()
    saved = 0
    try:
        calibration = start_up(sampler, display, cal_file, imu)
        if calibration is None:
            raise SystemExit("Stopped at start-up. The wrist display shows why.")
        flat, fist = calibration

        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(COLUMNS)

            def write(letter, rep, frames):
                for number, frame in enumerate(frames):
                    writer.writerow([person, session, letter, rep, number] + frame_row(frame, flat, fist))
                file.flush()                                   # on disk now, even if Ctrl+C comes next

            # Shuffled rounds: every letter once per round, in a new order each round.
            for rep in range(1, args.reps + 1):
                order = letters[:]
                random.shuffle(order)
                for letter in order:
                    print(f"[SIGN] {letter} ({rep}/{args.reps}): {HOW_TO[letter]}")
                    progress = f"{rep}/{args.reps}"
                    if letter in MOTION_LETTERS:
                        frames = record_trace(sampler, imu, display, letter, progress)
                    else:
                        frames = record_sign(sampler, imu, display, flat, fist, letter, progress)
                    write(letter, rep, frames)
                    saved += 1
                    print(f"[DATA] {letter} try {rep} saved ({saved} signs so far)")
                    display.message(f"SIGN {letter}", "Relax hand")
                    time.sleep(RELAX_SECONDS)

            if args.none_seconds > 0:
                write("NONE", 0, record_none(sampler, imu, display, args.none_seconds))
                print(f"[DATA] NONE: {args.none_seconds} s saved")

        display.message("DATA SAVED", f"{saved} signs")
        print(f"\nSaved to {path}\nNext: python3 train_model.py"
              + (f" --data {folder}" if args.mock else ""))
    except KeyboardInterrupt:
        print(f"\nStopped by user. {saved} signs are saved in {path}")
    finally:
        if imu is not None:
            imu.stop()
        sampler.stop()
        adc.close()


def choose_letters(text, imu):
    """The letters to record: --letters, or every letter the glove can record (26 with the gyroscope)."""
    if not text:
        return list(LETTERS) if imu is not None else list(STATIC_LETTERS)
    letters = [letter.strip().upper() for letter in text.split(",") if letter.strip()]
    for letter in letters:
        if letter not in LETTERS:
            raise SystemExit(f"Unknown letter {letter!r}. Use letters from A to Z.")
        if letter in MOTION_LETTERS and imu is None:
            raise SystemExit(f"{letter} is {MOTION_LETTERS[letter]}: it needs the gyroscope, which isn't working.")
    return letters


def read_frame(sampler, imu):
    """One frame: the raw flex readings, and the gyroscope's tilt, motion and raw readings (or None)."""
    frame = {"raw": sampler.newest(), "tilt": None, "motion": None, "accel": None, "gyro": None}
    if imu is not None:
        frame["tilt"], frame["motion"] = imu.state()
        frame["accel"], frame["gyro"] = imu.latest()
    return frame


def frame_row(frame, flat, fist):
    """One CSV row's numbers, in COLUMNS order. Gyroscope columns are empty without one."""
    bends = normalize(frame["raw"], flat, fist)

    def numbers(values, digits):
        return [f"{v:.{digits}f}" for v in values] if values is not None else [""] * 3

    return ([f"{b:.4f}" for b in bends] + numbers(frame["tilt"], 4) + numbers(frame["motion"], 4)
            + [f"{r:.1f}" for r in frame["raw"]] + numbers(frame["accel"], 4) + numbers(frame["gyro"], 2)
            + [f"{x:.1f}" for x in flat] + [f"{x:.1f}" for x in fist])


def record_sign(sampler, imu, display, flat, fist, letter, progress):
    """Wait (no time limit) for the user to form the sign and hold it still. Returns FRAMES frames.

    1. "Make the sign": nothing counts until some finger moves MOVE away from where it was.
    2. "Hold still": once every finger has stayed within STILL for FRAMES frames in a row
       (and the gyroscope says the hand isn't moving), those frames are the recording.
       Moving again just restarts the count.
    """
    title = f"SIGN {letter} {progress}"
    start = normalize(sampler.newest(), flat, fist)
    started = False
    recent = []                                          # the last FRAMES (frame, bends) pairs

    while True:
        frame = read_frame(sampler, imu)
        faulty = find_faulty_finger(frame["raw"])
        if faulty:                                       # never record a disconnected sensor
            recent.clear()
            display.message("SENSOR FAULT", f"Check {faulty}")
            time.sleep(1 / settings.LOOP_HZ)
            continue

        bends = normalize(frame["raw"], flat, fist)
        if not started and np.max(np.abs(bends - start)) >= MOVE:
            started = True

        if started:
            if is_moving(frame["motion"]):
                recent = []                              # the hand is turning: not held still yet
            else:
                recent = (recent + [(frame, bends)])[-FRAMES:]
            window = np.array([b for _, b in recent]) if recent else np.zeros((0, 5))
            if len(recent) == FRAMES and np.all(window.max(axis=0) - window.min(axis=0) <= STILL):
                display.message(title, "Saved")
                return [f for f, _ in recent]
            display.message(title, "Hold still")
        else:
            display.message(title, "Make the sign")
        time.sleep(1 / settings.LOOP_HZ)


def record_trace(sampler, imu, display, letter, progress):
    """J or Z: wait for the hand to start moving, then save FRAMES frames of the tracing.

    Make the letter's handshape and trace it, again and again, until it says Saved.
    If the hand stopped moving for too much of that second, it asks again.
    """
    title = f"SIGN {letter} {progress}"
    while True:
        display.message(title, "Trace it now")
        while True:                                      # no time limit: wait for the tracing to start
            frame = read_frame(sampler, imu)
            faulty = find_faulty_finger(frame["raw"])
            if faulty:
                display.message("SENSOR FAULT", f"Check {faulty}")
            elif is_moving(frame["motion"]):
                break
            else:
                display.message(title, "Trace it now")
            time.sleep(1 / settings.LOOP_HZ)

        display.message(title, "Keep tracing")
        frames = []
        while len(frames) < FRAMES:
            frame = read_frame(sampler, imu)
            if find_faulty_finger(frame["raw"]) is None:
                frames.append(frame)
            time.sleep(1 / settings.LOOP_HZ)
        if sum(is_moving(f["motion"]) for f in frames) >= TRACE_SHARE * FRAMES:
            display.message(title, "Saved")
            return frames
        print(f"[DATA] {letter}: the hand stopped moving. Keep tracing the whole time.")
        display.message(title, "Trace it again")
        time.sleep(1.0)


def record_none(sampler, imu, display, seconds):
    """Record the user moving their hand freely: wiggling, half-signs, resting. Labelled NONE."""
    display.message("NOT A LETTER", "Get ready")
    time.sleep(2.0)
    display.message("NOT A LETTER", "Move freely")
    frames = []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        frame = read_frame(sampler, imu)
        if find_faulty_finger(frame["raw"]) is None:
            frames.append(frame)
        time.sleep(1 / settings.LOOP_HZ)
    return frames


# ============================================================================
# Pretend glove for --mock: a hand that makes the letters it is asked for
# ============================================================================

# Rough guesses at each letter (alphabet.py), only used by --mock. Real data replaces these.
MOCK_POSES = EXPECTED_POSES


def make_collecting_hand(person):
    """A pretend hand that follows this file's prompts. Each person gets their own hand shape and habits."""
    from mock_glove import PretendHand

    class CollectingHand(PretendHand):
        def __init__(self):
            super().__init__()
            self.rng = np.random.default_rng(zlib.crc32(person.encode()))
            self.style = self.rng.normal(0.0, 0.06, (len(MOCK_POSES), 5))   # this person's habits
            self.tilt_style = self.rng.normal(0.0, 0.12, (len(MOCK_POSES), 3))
            self.target = [0.0] * 5
            self.free_until = 0.0

        def follow(self, title, detail):
            super().follow(title, detail)
            if title.startswith("SIGN ") and detail in ("Make the sign", "Trace it now", "Relax hand"):
                if detail == "Relax hand":
                    self.target, self.up, self.motion = [0.0] * 5, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)
                else:
                    letter = title.split()[1]
                    index = list(MOCK_POSES).index(letter)
                    this_try = self.rng.normal(0.0, 0.03, 5)                  # never quite the same twice
                    self.target = list(np.clip(np.array(MOCK_POSES[letter]) + self.style[index] + this_try, 0, 1))
                    self.up = tuple(np.array(EXPECTED_TILT[letter]) + self.tilt_style[index]
                                    + self.rng.normal(0.0, 0.05, 3))
                    self.motion = tuple(np.array(EXPECTED_MOTION[letter]) * self.rng.uniform(0.8, 1.2))
            self.moving_freely = title == "NOT A LETTER" and detail == "Move freely"

        def pose(self):
            if self.calibrating:
                self.up, self.motion = (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)
                return self.calibration_pose
            if getattr(self, "moving_freely", False):
                if time.monotonic() > self.free_until:                       # a new random hand every 0.4 s
                    self.target = list(self.rng.uniform(0, 1, 5))
                    self.up = tuple(self.rng.normal(0, 1, 3))
                    self.motion = tuple(self.rng.uniform(0, 0.5, 3))
                    self.free_until = time.monotonic() + 0.4
            return self.target

    return CollectingHand()


def open_mock_collector(person):
    """A pretend glove whose hand follows this file's prompts."""
    from mock_glove import MockMCP3208, MockOLED
    hand = make_collecting_hand(person)
    return MockMCP3208(hand), MockOLED(hand), hand


def parse_args():
    parser = argparse.ArgumentParser(description="Collect ASL letter data for train_model.py")
    parser.add_argument("--person", required=True,
                        help="who is wearing the glove: zion, amad, milly, sleytters or chris (settings.TEAM)")
    parser.add_argument("--letters", help="default: all 26 with the gyroscope, the 24 still letters without")
    parser.add_argument("--reps", type=int, default=5, help="tries per letter (default 5)")
    parser.add_argument("--none-seconds", type=int, default=NONE_SECONDS,
                        help=f"seconds of NOT-A-LETTER movement at the end (default {NONE_SECONDS}, 0 to skip)")
    parser.add_argument("--mock", action="store_true", help="use a pretend glove")
    return parser.parse_args()


if __name__ == "__main__":
    main()

