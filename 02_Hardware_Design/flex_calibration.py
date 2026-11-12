"""Two-position calibration; no assumed flex-sensor resistance endpoints."""

import json
import os
from pathlib import Path

from mcp3208 import FINGERS


def validate_calibration(calibration):
    if not isinstance(calibration, dict):
        raise ValueError("Calibration file must contain a JSON object")
    if calibration.get("version") != 1 or calibration.get("fingers") != list(FINGERS):
        raise ValueError("Calibration format or finger order does not match this build")
    for key in ("flat", "bent"):
        values = calibration.get(key)
        if not isinstance(values, list) or len(values) != 5:
            raise ValueError("Calibration must contain five flat and five bent counts")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not 0 <= value <= 4095
               for value in values):
            raise ValueError("Calibration counts must be finite values from 0 to 4095")
    # 50 counts is an initial usability gate, not a sensor specification.
    if any(abs(bent - flat) < 50
           for flat, bent in zip(calibration["flat"], calibration["bent"])):
        raise ValueError("A finger has less than 50 counts of movement; check wiring/poses")
    return calibration


def load_calibration(path):
    return validate_calibration(json.loads(Path(path).read_text(encoding="utf-8")))


def save_calibration(path, flat, bent):
    calibration = validate_calibration({"version": 1, "fingers": list(FINGERS),
                                        "flat": list(flat), "bent": list(bent)})
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return calibration


def bend_percent(counts, calibration):
    return tuple(max(0.0, min(100.0, 100.0 * (value - flat) / (bent - flat)))
                 for value, flat, bent in
                 zip(counts, calibration["flat"], calibration["bent"]))
