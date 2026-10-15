"""
Pretend glove hardware, for running the calibration code without the real glove.
 
Nothing here touches the Raspberry Pi's pins, so it runs on any computer.
These are the mock classes planned in the Design Specification
(Appendix C, 7.3.3: MockMCP3208 and MockOLED).
"""
 
import random
import threading
 
# Raw readings (0-4095) a typical hand might give, Thumb to Pinky.
# Straight fingers read HIGH and bent fingers read LOWER: bending raises the
# flex sensor's resistance, which lowers the voltage out of its divider.
FLAT_READINGS = [2250, 2300, 2280, 2260, 2240]
FIST_READINGS = [1750, 1500, 1450, 1480, 1550]
NOISE = 10    # real sensors jitter a little, so each reading wobbles by up to +/- this
 
 
class PretendHand:
    """A fake hand that does whatever the wrist display tells it to."""
 
    def __init__(self, stuck_channel=None):
        self.pose = "flat"
        self.stuck_channel = stuck_channel    # this sensor never changes, like a loose wire
 
    def follow(self, title, detail):
        """Form the pose the display asks for, just like a real user would."""
        if "FLAT" in detail:
            self.pose = "flat"
        elif "FIST" in detail:
            self.pose = "fist"
 
    def reading(self, channel):
        """The raw value this hand produces on one ADC channel right now."""
        if self.pose == "fist" and channel != self.stuck_channel:
            value = FIST_READINGS[channel]
        else:
            value = FLAT_READINGS[channel]
        return value + random.randint(-NOISE, NOISE)
 
 
class MockMCP3208:
    """Stands in for the ADC: returns the pretend hand's readings."""
 
    def __init__(self, hand):
        self.hand = hand
 
    def read(self, channel):
        return self.hand.reading(channel)
 
    def close(self):
        pass
 
 
class MockOLED:
    """Stands in for the wrist OLED: prints each message instead of drawing it.
 
    It also keeps a list of every message shown, so tests can check them.
    """
 
    def __init__(self, hand):
        self.hand = hand
        self.history = []
 
    def message(self, title, detail=""):
        print(f"[OLED] {title:<12} {detail}".rstrip())
        self.history.append((title, detail))
        self.hand.follow(title, detail)       # the pretend user reads the screen
 
 
class MockButton:
    """Stands in for a gpiozero Button: pressing Enter counts as a press."""
 
    def __init__(self):
        self.when_pressed = None
        threading.Thread(target=self._watch_keyboard, daemon=True).start()
 
    def _watch_keyboard(self):
        while True:
            try:
                input()
            except EOFError:                  # no keyboard to read from
                return
            if self.when_pressed:
                self.when_pressed()
 