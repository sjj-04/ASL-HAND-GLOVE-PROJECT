"""
Calibration for the full program: 4 steps with the gyroscope, 2 without.

  1. "Hold hand FLAT"     each finger's 0%
  2. "Make a FIST"        each finger's 100%
  3. "Point fingers UP"   gravity runs along the fingers: the "along the fingers" direction
  4. "Thumb UP"           gravity runs along the thumb: the "toward the thumb" direction

Steps 1 and 2 fit the flex sensors to this person's own range, finger by
finger: if someone can only bend a finger halfway, halfway is their 100%.

Steps 3 and 4 fit the gyroscope (IMU) to how it sits on this person's wrist,
so "which way the hand points" means the same on everyone, and they measure
the gyro's still reading (it reads a few deg/s even when it isn't turning).
Every step waits for the user instead of a timer: it records once the pose has
been reached and held still for HOLD_STILL_SECONDS, while a progress bar fills.
Moving starts the hold again. A step that never happens gives up after
STEP_TIMEOUT_SECONDS and tries again, up to MAX_CAL_TRIES times, then FAULT
(for the gyroscope steps, the previous gyroscope calibration is kept).

Everything is saved in calibration.json and used until the next calibration.
Without a gyroscope, steps 3 and 4 are skipped.

Calibration_Routine/glove_calibration.py still has the original timed routine
(2 seconds per pose) for its own program and tests; the glove uses the waiting
steps here. Both save the same calibration.json.
"""

import json
import os

import numpy as np

import settings
from settings import routine

run_calibration = routine.run_calibration       # the original timed flowchart (Calibration_Routine), unchanged
save_calibration = routine.save_calibration

HOLD_STILL_SECONDS = 1.0       # each pose is recorded once it has been held still this long
STEP_TIMEOUT_SECONDS = 20.0    # ...or the step gives up ("Too slow", try again) after this long
FINGER_STILL = 40              # "still" for the fingers: every finger within this many ADC counts
STRAIGHT_BEND = 0.35           # step 3: fingers this straight (0 flat, 1 fist) count as "pointing"
TURN_DEGREES = 30.0            # ...or the hand has turned this far since the prompt appeared
IMU_POLL_SECONDS = 0.1         # each reading looks at this much gyroscope data
STILL_GYRO_DPS = 5.0           # while holding a pose, the gyro may wobble this much (deg/s) per axis
STILL_ACCEL_G = 0.08           # ...and the accelerometer this much (g)
GRAVITY_RANGE = (0.8, 1.2)     # a still hand feels 1 g; outside this the reading isn't trustworthy


def load_calibration(cal_file=settings.CAL_FILE):
    """
    Return the saved (flat, fist) readings, or None if there's no usable file.

    Uses Calibration_Routine's loader, then also rejects a finger whose range is
    too small, because normalize() would divide by (almost) zero.
    """
    calibration = routine.load_calibration(cal_file)
    if calibration is None:
        return None
    flat, fist = calibration
    if np.any(np.abs(fist - flat) < settings.MIN_RANGE):
        return None                                  # wouldn't pass calibration, so don't trust it
    return calibration


def run_full_calibration(sampler, display, cal_file=settings.CAL_FILE, imu=None):
    """All the calibration steps: 1-2 for the fingers, then 3-4 for the gyroscope if there is one.

    Every step waits for the user: nothing is recorded until the pose has been reached
    and held still for HOLD_STILL_SECONDS (a progress bar fills while it's held).

    Returns (flat, fist), or None if the fingers failed (FAULT is on screen).
    If only the gyroscope steps fail, the fingers are still saved and the previous
    gyroscope calibration (if any) is kept.
    """
    previous = load_imu_calibration(cal_file)        # saving the fingers rewrites the file, so read it first
    finger_display = display if imu is None else _FingersSavedDisplay(display)
    calibration = run_finger_calibration(sampler, finger_display, cal_file)
    if calibration is None or imu is None:
        return calibration

    orientation = calibrate_orientation(imu, display, sampler, calibration)
    if orientation is None:
        orientation = previous                       # FAULT: keep the last good gyroscope calibration
    if orientation is not None:
        save_imu_calibration(cal_file, *orientation)
        imu.set_orientation(*orientation)
    if orientation is not None and orientation is not previous:
        display.message("Calibration", "saved")
        routine.time.sleep(routine.MESSAGE_SECONDS)
    return calibration


def run_finger_calibration(sampler, display, cal_file=settings.CAL_FILE):
    """Steps 1 and 2, waiting for the user. The same checks, messages and saved file as
    Calibration_Routine's run_calibration(), which records on a fixed 2-second timer instead.

      1. "Hold hand FLAT": recorded once the fingers have been still for HOLD_STILL_SECONDS
         (the hand is often already flat, so it doesn't wait for movement).
      2. "Make a FIST": waits until the fingers have bent well away from flat, then still.
    """
    for attempt in range(1, routine.MAX_CAL_TRIES + 1):
        flat = wait_for_pose(display, "Hold hand FLAT", sampler.newest,
                             ready=lambda raw: True, steady=fingers_steady)
        fist = None
        if flat is not None:
            flat = np.mean(flat, axis=0)
            fist = wait_for_pose(display, "Make a FIST", sampler.newest,
                                 ready=lambda raw: np.max(np.abs(raw - flat)) >= settings.MIN_RANGE,
                                 steady=fingers_steady)
        if fist is not None:
            fist = np.mean(fist, axis=0)
            change = np.abs(fist - flat)
            routine.print_readings(attempt, flat, fist, change)
            if np.all(change >= settings.MIN_RANGE):
                save_calibration(flat, fist, cal_file)
                display.message("Calibration", "saved")
                routine.time.sleep(routine.MESSAGE_SECONDS)
                return flat, fist
            weakest = settings.FINGERS[int(np.argmin(change))]
            display.message(f"Check {weakest}", f"Try {attempt} of {routine.MAX_CAL_TRIES}")
        else:                                        # the pose never came, or was never held still
            print(f"[CAL] No steady pose within {STEP_TIMEOUT_SECONDS:.0f} s (try {attempt})")
            display.message("Too slow", f"Try {attempt} of {routine.MAX_CAL_TRIES}")
        routine.time.sleep(routine.MESSAGE_SECONDS)

    display.message("FAULT", "Check wiring")
    return None


def calibrate_orientation(imu, display, sampler=None, calibration=None):
    """Steps 3 and 4, waiting for the user. Returns (rotation, gyro_offset), or None after MAX_CAL_TRIES.

      3. "Point fingers UP": waits until the fingers have straightened from the fist (or the
         hand has turned at least TURN_DEGREES), then held still.
      4. "Thumb UP": waits until the hand points at least imu.MIN_POSE_ANGLE away from
         step 3, then held still.
    sampler and calibration (flat, fist) let step 3 see the fingers straighten; without
    them it only waits for the hand to be still.
    """
    from imu import MIN_POSE_ANGLE, orientation_from

    def read():
        accel, gyro = imu.recent(IMU_POLL_SECONDS)
        raw = sampler.newest() if sampler is not None else None
        return accel, gyro, raw

    def imu_steady(window):
        accel = np.vstack([a for a, _, _ in window])
        gyro = np.vstack([g for _, g, _ in window])
        return len(accel) >= 5 and pose_problem(accel, gyro) is None

    def straightened(sample):
        if sampler is None or calibration is None:
            return True
        from sign_feedback import normalize
        return float(np.max(normalize(sample[2], *calibration))) <= STRAIGHT_BEND

    for attempt in range(1, routine.MAX_CAL_TRIES + 1):
        accel_now, _, _ = read()
        start = direction(accel_now)
        up_pose = wait_for_pose(display, "Point fingers UP", read, steady=imu_steady,
                                ready=lambda s: len(s[0]) > 0 and (straightened(s) or
                                                                  angle(direction(s[0]), start) >= TURN_DEGREES))
        problem = "Hold still"
        if up_pose is not None:
            fingers_up = direction(np.vstack([a for a, _, _ in up_pose]))
            thumb_pose = wait_for_pose(display, "Thumb UP", read, steady=imu_steady,
                                       ready=lambda s: len(s[0]) > 0 and
                                       angle(direction(s[0]), fingers_up) >= MIN_POSE_ANGLE)
            problem = "Turn hand more"               # never turned far enough from fingers-up
            if thumb_pose is not None:
                thumb_up = direction(np.vstack([a for a, _, _ in thumb_pose]))
                rotation = orientation_from(fingers_up, thumb_up)
                if rotation is not None:
                    gyro_offset = np.mean(np.vstack([g for _, g, _ in up_pose + thumb_pose]), axis=0)
                    print_orientation(rotation, gyro_offset)
                    return rotation, gyro_offset
        print(f"[CAL] Gyroscope step failed: {problem} (try {attempt} of {routine.MAX_CAL_TRIES})")
        display.message(problem, f"Try {attempt} of {routine.MAX_CAL_TRIES}")
        routine.time.sleep(routine.MESSAGE_SECONDS)

    display.message("FAULT", "Check gyroscope")
    routine.time.sleep(routine.MESSAGE_SECONDS)
    return None


def wait_for_pose(display, prompt, read, ready, steady):
    """Show a prompt, then wait (up to STEP_TIMEOUT_SECONDS) for the user to reach the pose and hold it.

    read():         one reading, LOOP_HZ times per second
    ready(reading): True once the hand has reached the pose
    steady(window): True if the readings in the window haven't moved
    Returns the HOLD_STILL_SECONDS of readings the pose was held for, or None on timeout.
    While it is held, the screen's progress bar fills; moving starts the hold again.
    """
    display.message("CALIBRATING", prompt)
    needed = max(2, round(HOLD_STILL_SECONDS * settings.LOOP_HZ))
    window = []
    for _ in range(round(STEP_TIMEOUT_SECONDS * settings.LOOP_HZ)):
        reading = read()
        if not ready(reading):
            window = []
        else:
            window = (window + [reading])[-needed:]
            if not steady(window):
                window = [reading]                   # it moved: start holding again from here
        show_progress(display, len(window) / needed)
        if len(window) == needed:
            return window
        routine.time.sleep(1 / settings.LOOP_HZ)
    return None


def fingers_steady(window):
    """Every finger within FINGER_STILL ADC counts for the whole window."""
    readings = np.array(window, dtype=float)
    return bool(np.all(readings.max(axis=0) - readings.min(axis=0) <= FINGER_STILL))


def show_progress(display, fraction):
    """Fill the touchscreen's progress bar, if this display has one."""
    if hasattr(display, "progress"):
        display.progress(fraction)


def direction(accel):
    """The average accelerometer reading as a unit vector (which way is up), or zeros if there's none."""
    accel = np.asarray(accel, dtype=float)
    if accel.size == 0:
        return np.zeros(3)
    mean = accel.reshape(-1, 3).mean(axis=0)
    size = np.linalg.norm(mean)
    return mean / size if size > 0 else np.zeros(3)


def angle(a, b):
    """Degrees between two directions (180 if either is missing)."""
    if not np.any(a) or not np.any(b):
        return 180.0
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def pose_problem(accel, gyro):
    """Why a pose can't be used ("Hold still"), or None if it's good."""
    if len(accel) < 5:
        return "No gyro data"
    if np.max(np.std(gyro, axis=0)) > STILL_GYRO_DPS or np.max(np.std(accel, axis=0)) > STILL_ACCEL_G:
        return "Hold still"
    size = float(np.linalg.norm(accel.mean(axis=0)))
    if not GRAVITY_RANGE[0] <= size <= GRAVITY_RANGE[1]:
        return "Hold still"
    return None


def load_imu_calibration(cal_file=settings.CAL_FILE):
    """The saved gyroscope calibration: (rotation 3x3, gyro_offset 3), or None if there isn't one."""
    try:
        with open(cal_file) as f:
            part = json.load(f)["imu"]
        rotation = np.array(part["rotation"], dtype=float)
        gyro_offset = np.array(part["gyro_offset"], dtype=float)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if rotation.shape != (3, 3) or gyro_offset.shape != (3,):
        return None
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-3):
        return None                                  # not a real rotation: damaged file
    return rotation, gyro_offset


def save_imu_calibration(cal_file, rotation, gyro_offset):
    """Add the gyroscope calibration to calibration.json, next to the finger readings.

    Written to a temporary file first and swapped in, like save_calibration(), so a
    power cut mid-save can't leave a half-written file.
    """
    with open(cal_file) as f:
        data = json.load(f)
    data["imu"] = {"rotation": np.round(rotation, 5).tolist(), "gyro_offset": np.round(gyro_offset, 3).tolist()}
    temp_file = cal_file.with_suffix(".tmp")
    with open(temp_file, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_file, cal_file)


def apply_saved_imu_calibration(imu, cal_file=settings.CAL_FILE):
    """Start-up: give the gyroscope its saved calibration. Returns True if there was one."""
    if imu is None:
        return False
    saved = load_imu_calibration(cal_file)
    if saved is None:
        print("[CAL] No gyroscope calibration saved yet: using IMU_AXES. Tap Recalibrate to set it up.")
        return False
    imu.set_orientation(*saved)
    return True


def print_orientation(rotation, gyro_offset):
    """Show what the gyroscope steps measured, in the terminal (handy when mounting the board)."""
    names = ["along fingers", "toward thumb", "back of hand"]
    print("[CAL] Gyroscope directions (in the board's x, y, z):")
    for name, row in zip(names, rotation):
        print(f"        {name:<14} {row[0]:+.2f} {row[1]:+.2f} {row[2]:+.2f}")
    print(f"      Gyro still reading: {gyro_offset[0]:+.1f} {gyro_offset[1]:+.1f} {gyro_offset[2]:+.1f} deg/s")


class _FingersSavedDisplay:
    """Passes messages on, but says "Fingers saved" instead of "Calibration saved" after step 2,
    because steps 3 and 4 still follow."""

    def __init__(self, display):
        self.display = display

    def message(self, title, detail=""):
        if (title, detail) == ("Calibration", "saved"):
            title = "Fingers"
        self.display.message(title, detail)

    def __getattr__(self, name):
        return getattr(self.display, name)
