"""Five-point affine touch calibration, including axis swaps and reversals."""
import json
import math
import os
from pathlib import Path

TARGETS = ((12, 15), (115, 15), (12, 53), (115, 53), (64, 34))


def solve(matrix, vector):
    rows = [list(row) + [value] for row, value in zip(matrix, vector)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda index: abs(rows[index][col]))
        rows[col], rows[pivot] = rows[pivot], rows[col]
        scale = rows[col][col]
        if abs(scale) < 1e-10:
            raise ValueError("Touch points have insufficient separation; repeat calibration")
        rows[col] = [value / scale for value in rows[col]]
        for row in range(3):
            if row != col:
                scale = rows[row][col]
                rows[row] = [a - scale * b for a, b in zip(rows[row], rows[col])]
    return [row[3] for row in rows]


def fit(raw, targets=TARGETS):
    if len(raw) != len(targets) or len(raw) < 3:
        raise ValueError("Touch calibration requires matching point sets")
    rows = [[x / 4095.0, y / 4095.0, 1] for x, y in raw]
    matrix = [[sum(row[a] * row[b] for row in rows) for b in range(3)] for a in range(3)]
    coefficients = [solve(matrix, [sum(row[a] * target[axis] for row, target in zip(rows, targets))
                                  for a in range(3)]) for axis in range(2)]
    mapping = {"version": 1, "coefficients": coefficients}
    errors = [math.hypot(*(a - b for a, b in zip(project(point, mapping), target)))
              for point, target in zip(raw, targets)]
    if max(errors) > 6:
        raise ValueError("Touch calibration error exceeds 6 pixels; tap the crosses again")
    mapping["max_error_pixels"] = max(errors)
    return mapping


def project(raw, mapping):
    x, y = raw
    return tuple(a * x / 4095.0 + b * y / 4095.0 + c
                 for a, b, c in mapping["coefficients"])


def load_mapping(path):
    mapping = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or mapping.get("version") != 1:
        raise ValueError("Invalid touch calibration format")
    rows = mapping.get("coefficients")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("Invalid touch calibration coefficients")
    for row in rows:
        if not isinstance(row, list) or len(row) != 3 or any(
                isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x)
                for x in row):
            raise ValueError("Invalid touch calibration coefficients")
    if abs(rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0]) < 1e-8:
        raise ValueError("Touch calibration is singular; recalibrate")
    return mapping


def save_mapping(path, mapping):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    load_mapping(temporary)
    os.replace(temporary, destination)


def hit(point, rectangle):
    x, y = point
    left, top, right, bottom = rectangle
    return left <= x <= right and top <= y <= bottom
