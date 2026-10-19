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

    TEMPORARY: this stands in for the neural network. It picks the sign whose
    reference is closest to the hand. The trained network will replace this
    and return the same two values, so nothing else needs to change.
    """
    best_sign, best_distance = None, None
    for sign, reference in settings.REFERENCES.items():
        # Distance = average difference between the hand and this sign, across all fingers.
        distance = float(np.mean(np.abs(bends - np.array(reference))))
        if best_distance is None or distance < best_distance:
            best_sign, best_distance = sign, distance
    return best_sign, 1.0 - best_distance               # closer match = higher confidence


def evaluate(bends, target, recognized, confidence):
    """Grade one attempt at the target sign. Returns "CORRECT", "ADJUST: [finger]" or "TRY AGAIN"."""
    errors = np.abs(bends - np.array(settings.REFERENCES[target]))
    out_of_place = [finger for finger, error in zip(settings.FINGERS, errors)
                    if error >= settings.ADJUST_TOLERANCE]

    # Correct needs both checks: confident it's the target sign, AND no finger out of place.
    if recognized == target and confidence >= settings.ACCEPT_CONFIDENCE and not out_of_place:
        return "CORRECT"
    if len(out_of_place) == 1:                           # "One finger clearly out of place?"
        return f"ADJUST: {out_of_place[0]}"
    return "TRY AGAIN"                                   # several fingers off, or none far enough to name


class HoldTimer:
    """Answers the flowchart question "Held for 1 second?"."""

    def __init__(self):
        self.started = None                              # when the current correct hold began

    def update(self, is_correct, now):
        """Return True once the sign has been correct for HOLD_SECONDS in a row."""
        if not is_correct:
            self.started = None                          # the hold was broken: start over
            return False
        if self.started is None:
            self.started = now
        return now - self.started >= settings.HOLD_SECONDS

    def reset(self):
        self.started = None


def practice_step(raw, calibration, target, hold_timer, now):
    """One pass of the practice loop: raw readings in, the wrist display's message out.

    Returns (title, detail, success). success is True when the target sign
    has been held correctly long enough to count.
    """
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
