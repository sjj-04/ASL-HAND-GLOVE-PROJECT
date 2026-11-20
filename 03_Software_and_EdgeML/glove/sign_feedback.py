"""
Sign feedback: flowchart sheets 2 and 3 in Section 6 of the Design Specification.

  normalize()      scale each finger from 0.0 (straight) to 1.0 (fully bent)
  recognize()      find the supported sign closest to the hand, and how sure it is
  evaluate()       grade the hand against the lesson's target sign
  practice_step()  put them together into the wrist display's message
"""

import numpy as np

import settings


def normalize(raw, flat, fist):
    """Scale each finger to 0.0 (straight) through 1.0 (fully bent), using the calibration.

    Example: flat = 2250, fist = 1500, reading = 1875 gives 0.5 (half bent).
    Readings slightly outside the calibrated range are clipped to 0.0 or 1.0.
    """
    return np.clip((raw - flat) / (fist - flat), 0.0, 1.0)


def recognize(bends):
    """Return the best-matching sign and a confidence score (0.0 to 1.0).

    This picks the sign whose reference is closest to the hand. 
    The trained network will replace this
    and return the same two values, so nothing else needs to change.
    """
    best_sign, best_distance = None, None
    for sign, reference in settings.REFERENCES.items():
        # Distance = average difference between the hand and this sign, across all fingers.
        distance = float(np.mean(np.abs(bends - np.array(reference))))
        if best_distance is None or distance < best_distance:
            best_sign, best_distance = sign, distance
    return best_sign, 1.0 - best_distance               # closer match = higher confidence


def recognize_with_margin(bends):
    """Return the best sign, its confidence, and how far ahead of the second-best sign it is.

    The margin rule: a hand that is nearly as close to two signs (for example a
    tightly curled C, halfway to A) is too close to call, so free practice
    shows nothing instead of guessing.
    """
    scores = sorted(((1.0 - float(np.mean(np.abs(bends - np.array(reference)))), sign)
                     for sign, reference in settings.REFERENCES.items()), reverse=True)
    best_confidence, best_sign = scores[0]
    second_confidence = scores[1][0] if len(scores) > 1 else 0.0
    return best_sign, best_confidence, best_confidence - second_confidence


def match_score(bends, target):
    """Match score, 0 to 100: 100 x (1 - average finger error). 100 is a perfect copy of the target."""
    errors = np.abs(bends - np.array(settings.REFERENCES[target]))
    return 100.0 * (1.0 - float(np.mean(errors)))


def evaluate(bends, target, recognized=None, confidence=None):
    """Grade one attempt at the target sign. Returns "CORRECT", "ADJUST: [finger]" or "TRY AGAIN".

    CORRECT needs a match score of at least ACCEPT_SCORE and every finger within
    ADJUST_TOLERANCE of the target (the TC-DELIV-03 acceptance rule).
    recognized and confidence are kept for the free-practice display only.
    """
    errors = np.abs(bends - np.array(settings.REFERENCES[target]))
    tiny = 1e-9                                          # ignore floating-point rounding at the limits
    out_of_place = [finger for finger, error in zip(settings.FINGERS, errors)
                    if error > settings.ADJUST_TOLERANCE + tiny]

    if match_score(bends, target) >= settings.ACCEPT_SCORE - tiny and not out_of_place:
        return "CORRECT"
    if len(out_of_place) == 1:                           # "One finger clearly out of place?"
        return f"ADJUST: {out_of_place[0]}"
    return "TRY AGAIN"                                   # several fingers off, or none far enough to name


class FaultGuard:
    """Catches an unplugged or shorted sensor during a lesson, and decides when scoring may resume.

    A fault starts on the first bad reading. Scoring stays off until every
    sensor has read in range for FAULT_RECOVERY_SECONDS in a row, so a wire
    that is flickering in and out can't sneak a score through.
    """

    def __init__(self):
        self.faulty = None                               # name of the faulty finger, or None
        self.valid_since = None                          # when readings became valid again

    def update(self, raw, now):
        """Return the faulty finger's name while scoring must stay off, otherwise None."""
        from startup_check import find_faulty_finger
        bad = find_faulty_finger(raw)
        if bad:
            self.faulty, self.valid_since = bad, None
            return bad
        if self.faulty is None:
            return None
        if self.valid_since is None:
            self.valid_since = now
        if now - self.valid_since >= settings.FAULT_RECOVERY_SECONDS:
            self.faulty = self.valid_since = None        # recovered: no recalibration needed
            return None
        return self.faulty


class HoldTimer:
    """Answers the flowchart question "Held long enough?" (HOLD_SECONDS).

    It reports success once per hold: to score again the user has to break
    the pose first, so holding a sign for a long time counts only once.
    """

    def __init__(self):
        self.started = None                              # when the current correct hold began
        self.fired = False                               # success already reported for this hold

    def update(self, is_correct, now):
        """Return True once, when the sign has been correct for HOLD_SECONDS in a row."""
        if not is_correct:
            self.reset()                                 # the hold was broken: start over
            return False
        if self.started is None:
            self.started = now
        if self.fired or now - self.started < settings.HOLD_SECONDS:
            return False
        self.fired = True
        return True

    def reset(self):
        self.started = None
        self.fired = False


def practice_step(raw, calibration, target, hold_timer, now, fault_guard=None):
    """One pass of the practice loop: raw readings in, the wrist display's message out.

    Returns (title, detail, success). success is True when the target sign
    has been held correctly long enough to count.
    """
    if fault_guard is not None:
        faulty = fault_guard.update(raw, now)
        if faulty:                                       # never score a disconnected sensor
            hold_timer.reset()
            return "SENSOR FAULT", f"Check {faulty}", False

    flat, fist = calibration
    bends = normalize(raw, flat, fist)
    recognized, confidence = recognize(bends)

    if target is None:                                   # "Lesson target active?" No: independent use
        shown = recognized if confidence >= settings.ACCEPT_CONFIDENCE else "--"
        return "Free practice", f"SIGN: {shown}", False  # name the sign, never call it "correct"

    result = evaluate(bends, target, recognized, confidence)
    if hold_timer.update(result == "CORRECT", now):
        return f"CORRECT: {target}", "", True
    if result == "CORRECT":
        return f"Target: {target}", "HOLD STEADY", False
    return f"Target: {target}", result, False
