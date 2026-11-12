"""Personalized, static A/B/C matching from five flex sensors. No pretrained model."""
import json
import math
import os
import statistics
from pathlib import Path

from flex_calibration import validate_calibration

LETTERS = ("A", "B", "C")


def sensor_fault(counts):
    names = ("THUMB", "INDEX", "MIDDLE", "RING", "PINKY")
    if len(counts) != 5:
        return "ADC FRAME"
    for name, value in zip(names, counts):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return name
        if value < 50 or value > 4045:
            return name
    return None


def normalize(counts, profile):
    fault = sensor_fault(counts)
    if fault:
        raise ValueError("SENSOR FAULT " + fault)
    return tuple(max(0.0, min(1.0, (value - flat) / (bent - flat)))
                 for value, flat, bent in zip(counts, profile["flat"], profile["bent"]))


def distance(first, second):
    errors = [abs(a - b) for a, b in zip(first, second)]
    return statistics.mean(errors), max(errors)


def validate_profile(profile):
    validate_calibration(profile)
    if any(abs(b - a) < 200 for a, b in zip(profile["flat"], profile["bent"])):
        raise ValueError("Each finger needs at least 200 counts of flat-to-fist movement")
    references = profile.get("references")
    if not isinstance(references, dict) or set(references) != set(LETTERS):
        raise ValueError("Profile must contain recorded A, B and C samples")
    for letter in LETTERS:
        samples = references[letter]
        if not isinstance(samples, list) or len(samples) != 3:
            raise ValueError("Record three samples for " + letter)
        for sample in samples:
            if not isinstance(sample, list) or len(sample) != 5:
                raise ValueError("Each sign sample needs five normalized values")
            if any(isinstance(x, bool) or not isinstance(x, (float, int))
                   or not math.isfinite(x) or not 0 <= x <= 1 for x in sample):
                raise ValueError("Sign samples must contain finite values from 0 to 1")
        if max(distance(a, b)[0] for a in samples for b in samples) > 0.12:
            raise ValueError(letter + " samples varied too much; repeat setup")
    for index, first in enumerate(LETTERS):
        for second in LETTERS[index + 1:]:
            separation = min(distance(a, b)[0] for a in references[first]
                             for b in references[second])
            if separation < 0.10:
                raise ValueError(first + "/" + second + " readings are too similar; repeat setup")
    return profile


def load_profile(path):
    return validate_profile(json.loads(Path(path).read_text(encoding="utf-8")))


def save_profile(path, profile):
    validate_profile(profile)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def classify(values, profile):
    """Return a matching letter or None; distance is not a probability."""
    ranked = []
    for letter in LETTERS:
        mean_error, worst_error = min(distance(values, sample)
                                      for sample in profile["references"][letter])
        ranked.append((mean_error, worst_error, letter))
    ranked.sort()
    mean_error, worst_error, letter = ranked[0]
    if mean_error > 0.10 or worst_error > 0.20:
        return None
    if ranked[1][0] - mean_error < 0.04:
        return None
    return letter


class HoldGate:
    def __init__(self, seconds=0.5):
        self.seconds = seconds
        self.reset()

    def reset(self):
        self.candidate = None
        self.started = None
        self.last = None
        self.samples = 0

    def update(self, letter, timestamp):
        if self.last is not None and timestamp <= self.last:
            return None
        gap = self.last is not None and timestamp - self.last > 0.5
        if letter is None:
            self.reset()
            return None
        if letter != self.candidate or gap:
            self.candidate, self.started, self.samples = letter, timestamp, 0
        self.last = timestamp
        self.samples += 1
        return letter if timestamp - self.started >= self.seconds and self.samples >= 3 else None
