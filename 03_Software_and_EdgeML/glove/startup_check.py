"""
Start-up check: "All five sensors in range?" (flowchart sheet 1).

An unplugged sensor reads near 0 and a shorted one reads near 4095, so a
reading close to either end means a wiring fault, not a finger position.
"""

import settings


def find_faulty_finger(readings):
    """Return the name of the first finger reading out of range, or None if all are fine."""
    for finger, value in zip(settings.FINGERS, readings):
        if value < settings.FAULT_MARGIN or value > settings.ADC_MAX - settings.FAULT_MARGIN:
            return finger
    return None
