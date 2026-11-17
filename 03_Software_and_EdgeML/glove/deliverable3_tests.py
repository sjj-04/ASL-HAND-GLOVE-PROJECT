"""
TC-DELIV-03: Calibration, Finger Level Scoring, and Recovery Verification.

Runs Tests 3.1-3.8 from Section 4.3 of the Deliverable and Test Procedures
document against the glove software, and prints each value the test report
form asks for, plus PASS or FAIL.

How it tests (Precondition 3, "replay and inject"):
  * Normalized poses are turned back into raw ADC counts with the calibration,
    then fed through the real normalize(), evaluate(), HoldTimer and
    practice_step(), one pass every 1/LOOP_HZ seconds of simulated time.
  * Match score = 100 x (1 - mean absolute finger error)   (Section 1.1 rule)
  * An "acceptance event" is practice_step() reporting success.

Run from the glove folder:
  py deliverable3_tests.py                  Tests 3.1-3.7 with the pretend glove (about 15 s)
  py deliverable3_tests.py --soak 30        also Test 3.8, a 30 minute soak run
  python3 deliverable3_tests.py --pi        Tests 3.1-3.2 read the real glove (on the Pi; wearer follows the LCD)
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import settings
from settings import routine
import calibration as cal
import sign_feedback as sf
import main as glove_main

FINGERS = settings.FINGERS
STRAIGHT = "ALL STRAIGHT"                  # the "all-straight target" Tests 3.5 and 3.6 select
settings.REFERENCES.setdefault(STRAIGHT, [0.0] * 5)
DT = 1 / settings.LOOP_HZ                  # one practice-loop pass
RESULTS = []


# ----------------------------------------------------------------------------- helpers

def verdict(test, title, passed, lines):
    RESULTS.append((test, title, passed))
    print(f"\n=== Test {test}: {title} ===")
    for line in lines:
        print("   " + line)
    print(f"   RESULT: {'PASS' if passed else 'FAIL'}")


def to_raw(bends, flat, fist):
    """A normalized pose (0.0 straight .. 1.0 bent) as the raw counts the ADC would give."""
    return np.asarray(flat, float) + np.asarray(bends, float) * (np.asarray(fist, float) - np.asarray(flat, float))


def score(bends, target):
    return 100 * (1 - float(np.mean(np.abs(np.asarray(bends) - np.array(settings.REFERENCES[target])))))


class RecordingDisplay:
    def __init__(self):
        self.log = []                                    # (time, title, detail)

    def message(self, title, detail=""):
        self.log.append((time.monotonic(), title, detail))


class FixedSampler:
    """Gives fixed readings that follow the calibration prompts (flat, then fist)."""

    def __init__(self, display, flat, fist):
        self.display, self.flat, self.fist = display, np.asarray(flat, float), np.asarray(fist, float)

    def average(self):
        detail = self.display.log[-1][2] if self.display.log else ""
        return self.fist.copy() if "FIST" in detail else self.flat.copy()

    newest = average


def run_poses(segments, calibration, target):
    """
    Replay [(pose, seconds), ...] through practice_step() at LOOP_HZ, like main.practice_loop().
    Returns the list of (time, title, detail, success) for every pass.
    """
    flat, fist = calibration
    timer, now, out = sf.HoldTimer(), 0.0, []
    guard = sf.FaultGuard() if hasattr(sf, "FaultGuard") else None   # the same fault check main.py uses
    for pose, seconds in segments:
        raw = to_raw(pose, flat, fist) if np.ndim(pose) and len(pose) == 5 and max(pose) <= 1.5 else np.asarray(pose, float)
        for _ in range(round(seconds / DT)):
            args = (raw, calibration, target, timer, now) + ((guard,) if guard else ())
            title, detail, success = sf.practice_step(*args)
            out.append((now, title, detail, success))
            now += DT
    return out


def events(log):
    return sum(1 for *_, success in log if success)


# ----------------------------------------------------------------------------- tests

def test_3_1(use_pi):
    """Flat and fist endpoints from a real calibration run; every span must be >= 500."""
    display, adc, sampler, cal_file = live_glove(use_pi)
    try:
        result = cal.run_calibration(sampler, display, cal_file)
    finally:
        sampler.stop()
        adc.close()
    if result is None:
        verdict("3.1", "Flat / fist calibration spans", False, ["Calibration failed (FAULT shown)."])
        return
    flat, fist = result
    LAST_CAL["value"] = result                       # Test 3.8 reuses the wearer's calibration
    span = flat - fist
    lines = [f"CH{i} {FINGERS[i]:<6}: {flat[i]:7.0f} / {fist[i]:7.0f}   F-B = {span[i]:6.0f}" for i in range(5)]
    lines.append(f"Minimum Difference: {span.min():.0f} counts   (need >= 500 on every channel)")
    lines.append(f"Software's own minimum (MIN_RANGE): {routine.MIN_RANGE}")
    verdict("3.1", "Flat / fist calibration spans", bool(np.all(span >= 500)), lines)


def test_3_2(use_pi):
    """Calibration state machine: FLAT phase lasts 2.50 +/- 0.1 s; two bad calibrations are rejected."""
    display, adc, sampler, cal_file = live_glove(use_pi)
    try:
        cal.run_calibration(sampler, display, cal_file)
    finally:
        sampler.stop()
        adc.close()
    times = {d: t for t, title, d in display.log if title == "CALIBRATING"}
    flat_seconds = times["Make a FIST"] - times["Hold hand FLAT"]

    # Two bad calibrations, replayed with the waits skipped:
    #   (a) Index never bends (a loose sensor), (b) Middle reads the same flat and fist (F = B, would be /0).
    good_flat, good_fist = [3000.0] * 5, [1000.0] * 5
    cases = [("Index", [1000, 2900, 1000, 1000, 1000]), ("Middle", [1000, 1000, 3000, 1000, 1000])]
    rejected = correct_channel = div_zero = 0
    real_sleep, routine.time.sleep = routine.time.sleep, lambda s: None
    try:
        for finger, fist in cases:
            d = RecordingDisplay()
            try:
                with np.errstate(all="raise"):
                    out = routine.run_calibration(FixedSampler(d, good_flat, fist), d, Path(tempfile.mkdtemp()) / "c.json")
            except (ZeroDivisionError, FloatingPointError):
                div_zero += 1
                continue
            rejected += out is None
            correct_channel += any(title == f"Check {finger}" for _, title, _ in d.log)
    finally:
        routine.time.sleep = real_sleep
    lines = [f"FLAT phase duration: {flat_seconds:.2f} s   (need 2.50 +/- 0.10)",
             f"Calibrations Rejected: {rejected} / 2",
             f"Correct Channel Reports: {correct_channel} / 2",
             f"/0 (divide by zero) Errors: {div_zero}"]
    ok = abs(flat_seconds - 2.5) <= 0.1 and rejected == 2 and correct_channel == 2 and div_zero == 0
    verdict("3.2", "Calibration state machine", ok, lines)


def test_3_3():
    """F = 3000, B = 1000: replay 3000, 2000, 1000, 3500, 500 on every channel."""
    flat, fist = np.full(5, 3000.0), np.full(5, 1000.0)
    expected = {3000: 0.0, 2000: 0.5, 1000: 1.0, 3500: 0.0, 500: 1.0}
    lines, worst = [], 0.0
    for raw, want in expected.items():
        got = sf.normalize(np.full(5, float(raw)), flat, fist)
        worst = max(worst, float(np.max(np.abs(got - want))))
        lines.append(f"{raw}: " + ", ".join(f"{v:.2f}" for v in got) + f"    (expect {want:.2f})")
    lines.append(f"Maximum Error: {worst:.3f}   (need <= 0.02, outputs within 0.00-1.00)")
    verdict("3.3", "Finger movement normalization", worst <= 0.02, lines)


def test_3_4():
    """Ten reloads of a saved profile, then three bad profiles must be rejected with CALIBRATION REQUIRED."""
    folder = Path(tempfile.mkdtemp())
    good = folder / "calibration.json"
    flat, fist = np.array([2816.0, 2810, 2820, 2805, 2815]), np.array([1252.0, 1260, 1240, 1255, 1250])
    cal.save_calibration(flat, fist, good)
    reloads = 0
    for _ in range(10):
        loaded = cal.load_calibration(good)
        reloads += loaded is not None and np.allclose(loaded[0], flat, atol=0.1) and np.allclose(loaded[1], fist, atol=0.1)

    missing = folder / "missing.json"
    malformed = folder / "malformed.json"
    malformed.write_text('{"flat": [2816, 2810,')
    same = folder / "f_equals_b.json"
    same.write_text(json.dumps({"fingers": FINGERS, "flat": [2000, 2810, 2820, 2805, 2815],
                                "fist": [2000, 1260, 1240, 1255, 1250]}))
    rejected = disabled = 0
    messages = []
    for bad in (missing, malformed, same):
        rejected += cal.load_calibration(bad) is None
        # Start-up with this file. If start-up tries to calibrate, the wearer holds [0,0,0,0,0] (flat)
        # for both poses, so calibration can't succeed and scoring must stay off.
        d = RecordingDisplay()
        real_sleep = routine.time.sleep, glove_main.time.sleep
        routine.time.sleep = glove_main.time.sleep = lambda s: None
        try:
            result = glove_main.start_up(FixedSampler(d, flat, flat), d, bad)
        finally:
            routine.time.sleep, glove_main.time.sleep = real_sleep
        disabled += result is None
        messages.append(" -> ".join(dict.fromkeys(f"{t} {x}".strip() for _, t, x in d.log)))
    lines = [f"Correct Reloads: {reloads} / 10",
             f"Invalid Files Rejected: {rejected} / 3",
             f"Scoring Disabled: {disabled} / 3"]
    lines += [f"GLCD ({name}): {m}" for name, m in zip(("missing", "malformed", "F=B CH0"), messages)]
    shows_required = all("CALIBRATION REQUIRED" in m.upper() for m in messages)
    lines.append(f"Shows CALIBRATION REQUIRED: {'YES' if shows_required else 'NO'}")
    verdict("3.4", "Profile reload and invalid-profile handling",
            reloads == 10 and rejected == 3 and disabled == 3 and shows_required, lines)


CAL = (np.full(5, 3000.0), np.full(5, 1000.0))      # replay calibration for 3.5-3.7


def test_3_5():
    """All-straight target: exact, index off by 0.40, all fingers off by 0.40."""
    cases = [("Exact Match", [0, 0, 0, 0, 0], 100), ("Index Error", [0, 0.4, 0, 0, 0], 92),
             ("All Finger", [0.4] * 5, 60)]
    lines, ok = [], True
    for name, pose, want in cases:
        s = score(pose, STRAIGHT)
        log = run_poses([(pose, 1.5)], CAL, STRAIGHT)
        msg = log[0][2] or log[0][1]
        accepted = events(log) > 0
        lines.append(f"{name}: score {s:.1f} (expect {want} +/-1), GLCD '{msg}', accepted {'YES' if accepted else 'NO'}")
        ok &= abs(s - want) <= 1
        if name == "Index Error":
            ok &= msg.upper().replace(":", "") == "ADJUST INDEX" and not accepted
        if name == "Exact Match":
            ok &= accepted
    verdict("3.5", "Gesture score and feedback", ok, lines)


def test_3_6():
    """Score and finger-error boundaries, and the hold timer."""
    def accepted(pose, seconds=0.60):
        return "ACCEPT" if events(run_poses([(pose, seconds)], CAL, STRAIGHT)) else "REJECT"
    exact, bad = [0.0] * 5, [1.0] * 5
    got = {
        "Input 0.100": (accepted([0.100] * 5), "ACCEPT"),
        "Input 0.105": (accepted([0.105] * 5), "REJECT"),
        "CH1 at 0.20": (accepted([0, 0.20, 0, 0, 0]), "ACCEPT"),
        "CH1 at 0.21": (accepted([0, 0.21, 0, 0, 0]), "REJECT"),
    }
    holds = {
        "0.40s Hold": (events(run_poses([(exact, 0.40), (bad, 0.5)], CAL, STRAIGHT)), 0),
        "0.60s Hold": (events(run_poses([(exact, 0.60), (bad, 0.5)], CAL, STRAIGHT)), 1),
        "Interrupted Hold": (events(run_poses([(exact, 0.30), (bad, 0.10), (exact, 0.30), (bad, 0.5)], CAL, STRAIGHT)), 0),
    }
    long_hold = events(run_poses([(exact, 2.0)], CAL, STRAIGHT))
    lines = [f"{k}: {v[0]}   (expect {v[1]})" for k, v in {**got, **holds}.items()]
    lines.append(f"Duplicate Events (2.0 s continuous hold): {max(0, long_hold - 1)}   (expect 0)")
    lines.append(f"Software hold time (HOLD_SECONDS): {settings.HOLD_SECONDS} s")
    ok = all(v[0] == v[1] for v in {**got, **holds}.values()) and long_hold == 1
    verdict("3.6", "Acceptance boundaries and hold timer", ok, lines)


def test_3_7():
    """Inject 49 counts on each channel during scoring; fault within 100 ms, no scores, recovery after >= 1 s."""
    exact_raw = to_raw([0.0] * 5, *CAL)
    delays, correct, scores_in_fault, recoveries = [], 0, 0, []
    for ch in range(5):
        faulted = exact_raw.copy()
        faulted[ch] = 49
        log = run_poses([(exact_raw, 1.0), (faulted, 1.0), (exact_raw, 3.0)], CAL, STRAIGHT)
        fault = [(t, ti, de, s) for t, ti, de, s in log if 1.0 <= t < 2.0]
        hit = next(((t, ti, de) for t, ti, de, s in fault if "FAULT" in ti.upper()), None)
        delays.append(None if hit is None else (hit[0] - 1.0) * 1000)
        correct += hit is not None and FINGERS[ch] in (hit[1] + " " + hit[2])
        scores_in_fault += sum(1 for *_, s in fault if s) + sum(1 for _, ti, *_ in fault if "FAULT" not in ti.upper())
        after = [(t, ti) for t, ti, de, s in log if t >= 2.0 and "FAULT" not in ti.upper()]
        recoveries.append(None if not after else after[0][0] - 2.0)
    fmt = lambda v, u: "none" if v is None else f"{v:.0f} {u}"
    lines = [f"Fault Delay CH{i}: {fmt(d, 'ms')}" for i, d in enumerate(delays)]
    lines += [f"Correct Channels: {correct} / 5",
              f"Recovery Delay: " + ", ".join("none" if r is None else f"{r:.2f} s" for r in recoveries),
              f"Scores During Fault: {scores_in_fault}   (passes that kept scoring instead of showing a fault)",
              "Recovery Without Calibration: YES (calibration is never re-run in this replay)"]
    ok = (all(d is not None and d <= 100 for d in delays) and correct == 5 and scores_in_fault == 0
          and all(r is not None and r >= 1.0 for r in recoveries))
    verdict("3.7", "Sensor fault during scoring and recovery", ok, lines)


def test_3_8(minutes, use_pi):
    """Soak: run the sampler and practice loop, cycling 3 reference poses and 5 single-finger deviations."""
    display, adc, sampler, cal_file = live_glove(use_pi, quiet=True)
    calibration = LAST_CAL.get("value") or (np.array(FLAT_COUNTS, float), np.array(FIST_COUNTS, float))
    cal.save_calibration(*calibration, cal_file)
    reads = {"n": 0}
    real_read = adc.read

    def counting_read(channel):
        if channel == 0:
            reads["n"] += 1
        return real_read(channel)

    adc.read = counting_read
    poses = [("A", settings.REFERENCES["A"], True), ("B", settings.REFERENCES["B"], True),
             ("C", settings.REFERENCES["C"], True)]
    for i in range(5):                                       # target B with one finger moved 0.40
        pose = list(settings.REFERENCES["B"])
        pose[i] = min(1.0, pose[i] + 0.4) if pose[i] < 0.5 else pose[i] - 0.4
        poses.append(("B", pose, False))
    pass
    freqs, worst_delay, false_accepts, crashes = [], 0.0, 0, 0
    end = time.monotonic() + minutes * 60
    last_count, last_t = 0, time.monotonic()
    timer, k = sf.HoldTimer(), 0
    try:
        while time.monotonic() < end:
            target, pose, valid = poses[k % len(poses)]
            if hand is not None and not use_pi:
                hand.override = pose
            else:                                            # real glove: the wearer follows these prompts
                which = "copy it exactly" if valid else f"copy it but move the {FINGERS[(k % len(poses)) - 3]} finger"
                print(f"   Pose for the next 3 s: sign {target}, {which}")
            timer.reset()
            seg_end = time.monotonic() + 3.0
            while time.monotonic() < seg_end:
                t0 = time.monotonic()
                try:
                    title, detail, success = sf.practice_step(sampler.newest(), calibration, target, timer, t0)
                    display.message(title, detail)
                except Exception:
                    crashes += 1
                    success = False
                worst_delay = max(worst_delay, (time.monotonic() - t0) * 1000)
                if success:
                    false_accepts += not valid
                if time.monotonic() - last_t >= 1.0:
                    now = time.monotonic()
                    freqs.append((reads["n"] - last_count) / (now - last_t))
                    last_count, last_t = reads["n"], now
                time.sleep(DT)
            k += 1
    finally:
        sampler.stop()
        adc.close()
    corrupted = int(cal.load_calibration(cal_file) is None)
    freqs = freqs[1:] or freqs
    lines = [f"Runtime: {minutes} min",
             f"Minimum Frequency: {min(freqs):.1f} Hz", f"Maximum Frequency: {max(freqs):.1f} Hz",
             f"Maximum Delay: {worst_delay:.1f} ms", f"False Acceptances: {false_accepts}",
             f"Crashes: {crashes}", f"Corrupted Files: {corrupted}"]
    ok = (minutes >= 30 and min(freqs) >= 198 and max(freqs) <= 202 and worst_delay <= 100
          and false_accepts == 0 and crashes == 0 and corrupted == 0)
    if minutes < 30:
        lines.append("(Shortened run: the form needs >= 30 min for a PASS.)")
    verdict("3.8", "Soak run", ok, lines)


# ----------------------------------------------------------------------------- glove setup

FLAT_COUNTS = [2816, 2810, 2820, 2805, 2815]
FIST_COUNTS = [1252, 1260, 1240, 1255, 1250]
hand = None
LAST_CAL = {}


def live_glove(use_pi, quiet=False):
    """The real glove on the Pi, or a pretend hand that follows the calibration prompts."""
    global hand
    from hardware import Sampler
    folder = Path(tempfile.mkdtemp())
    if use_pi:
        from hardware import open_hardware
        adc, real_display = open_hardware()
        display = TeeDisplay(real_display)
    else:
        hand = ReplayHand()
        adc, display = hand, TeeDisplay(hand, quiet)
    sampler = Sampler(adc)
    sampler.start()
    return display, adc, sampler, folder / "calibration.json"


class TeeDisplay(RecordingDisplay):
    def __init__(self, inner, quiet=False):
        super().__init__()
        self.inner, self.quiet = inner, quiet

    def message(self, title, detail=""):
        super().message(title, detail)
        if not self.quiet:
            print(f"   [LCD] {title} {detail}")
        self.inner.message(title, detail)


class ReplayHand:
    """Pretend glove: acts as the ADC (read) and follows the LCD (message)."""

    def __init__(self):
        self.bends, self.override = [0.0] * 5, None

    def message(self, title, detail=""):
        if title == "CALIBRATING":
            self.bends = [1.0] * 5 if "FIST" in detail else [0.0] * 5

    def read(self, channel):
        bends = self.override if self.override is not None else self.bends
        value = FLAT_COUNTS[channel] + bends[channel] * (FIST_COUNTS[channel] - FLAT_COUNTS[channel])
        return int(round(value + np.random.randint(-3, 4)))

    def close(self):
        pass


def main():
    parser = argparse.ArgumentParser(description="TC-DELIV-03 test runner")
    parser.add_argument("--pi", action="store_true", help="Tests 3.1, 3.2 and 3.8 use the real glove")
    parser.add_argument("--soak", type=float, default=0, help="also run Test 3.8 for this many minutes")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    print(f"TC-DELIV-03 run  {time.strftime('%Y-%m-%d %H:%M')}  ({'real glove' if args.pi else 'pretend glove'})")
    test_3_1(args.pi)
    test_3_2(args.pi)
    test_3_3()
    test_3_4()
    test_3_5()
    test_3_6()
    test_3_7()
    if args.soak:
        if args.pi:
            print("\nTest 3.8 on the Pi: the wearer cycles the poses; the pose script only drives the pretend glove.")
        test_3_8(args.soak, args.pi)
    print("\n=== Summary ===")
    for test, title, passed in RESULTS:
        print(f"   {test}  {'PASS' if passed else 'FAIL'}  {title}")


if __name__ == "__main__":
    main()
