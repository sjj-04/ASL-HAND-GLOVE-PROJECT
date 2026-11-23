"""
The gyroscope (IMU): which way the hand points, and how much it is moving.

The board is an MPU-6050 (or MPU-6500 / MPU-9250, which talk the same way) on
the Pi's I2C bus: SDA on GPIO2 (pin 3), SCL on GPIO3 (pin 5), VCC to 3.3 V,
GND to GND. It has two sensors:

  accelerometer   measures gravity, so it knows which way is up -> TILT
  gyroscope       measures how fast the hand turns             -> MOTION

From them the glove gets six numbers, added to the five finger bends:

  tilt_along, tilt_thumb, tilt_back   which way is "up", as (along the fingers, toward the
                                      thumb, out of the back of the hand), each -1.0 to 1.0.
                                      Fingers pointing up = (1, 0, 0). G and H point sideways,
                                      P and Q point down: that's how they split from L, U and K.
  motion_along, motion_thumb,         how fast the hand turns around each of those axes,
  motion_back                         averaged over MOTION_SECONDS, 0.0 (still) to 1.0
                                      (MOTION_FULL_SCALE_DPS). J and Z are traced, so they move;
                                      every other letter is held still.

hand_features() puts the 11 numbers together. The trained network and the
recordings use them; without an IMU they are just the 5 bends, like before.

Calibration (calibration.py, steps 3 and 4: "Point fingers UP", "Thumb UP")
measures how the board sits on this person's wrist and the gyro's still
reading. orientation_from() turns those into a rotation, and the sampler uses
it, so tilt and motion mean the same thing on everyone. Until a calibration
has been done with the IMU, the rough guess in IMU_AXES is used.

Check the IMU and its axes (on the Pi):
    python3 imu.py            prints tilt and motion 5 times per second, using the saved
                              calibration if there is one. Point your fingers up: tilt should
                              read about (1, 0, 0). Thumb up: (0, 1, 0). Before the first
                              calibration with the gyroscope, IMU_AXES in settings.py is used.
"""

import threading
import time
from collections import deque

import numpy as np

import settings

BEND_NAMES = [f"bend_{f.lower()}" for f in settings.FINGERS]
TILT_NAMES = ["tilt_along", "tilt_thumb", "tilt_back"]
MOTION_NAMES = ["motion_along", "motion_thumb", "motion_back"]
IMU_FEATURE_NAMES = TILT_NAMES + MOTION_NAMES
FEATURE_NAMES = BEND_NAMES + IMU_FEATURE_NAMES     # what the network reads, in this order
STILL_STATE = (np.array([1.0, 0.0, 0.0]), np.zeros(3))   # fingers up, not moving


def hand_features(bends, imu_state=None):
    """The network's input: 5 bends, plus 3 tilt and 3 motion numbers if there is an IMU reading."""
    bends = np.asarray(bends, dtype=float)
    if imu_state is None:
        return bends
    tilt, motion = imu_state
    return np.concatenate([bends, np.asarray(tilt, dtype=float), np.asarray(motion, dtype=float)])


def split_features(features):
    """(bends, tilt or None, motion or None) from hand_features()' output."""
    features = np.asarray(features, dtype=float)
    n = len(settings.FINGERS)
    if len(features) >= n + 6:
        return features[:n], features[n:n + 3], features[n + 3:n + 6]
    return features[:n], None, None


def is_moving(motion):
    """True if the hand is turning fast enough to be tracing J or Z."""
    return motion is not None and float(np.max(motion)) >= settings.MOTION_START


def orientation_from(fingers_up, thumb_up):
    """The rotation that turns IMU readings into glove axes, from two calibration poses.

    fingers_up: the accelerometer's average while the fingers point up. Gravity runs
                along the fingers, so this is the "along the fingers" direction.
    thumb_up:   the average while the thumb points up: the "toward the thumb" direction.
    The third, out of the back of the hand, is at right angles to both.
    Returns a 3x3 matrix (rows: along, thumb, back), or None if the two poses
    point too nearly the same way (less than MIN_POSE_ANGLE apart) to tell them apart.
    """
    along = np.asarray(fingers_up, dtype=float)
    thumb = np.asarray(thumb_up, dtype=float)
    if np.linalg.norm(along) < 0.5 or np.linalg.norm(thumb) < 0.5:
        return None                                  # hardly any gravity reading: not a real pose
    along = along / np.linalg.norm(along)
    thumb = thumb / np.linalg.norm(thumb)
    angle = np.degrees(np.arccos(np.clip(np.dot(along, thumb), -1.0, 1.0)))
    if angle < MIN_POSE_ANGLE:
        return None
    thumb = thumb - np.dot(thumb, along) * along     # make it exactly at right angles to "along"
    thumb = thumb / np.linalg.norm(thumb)
    back = np.cross(along, thumb)                    # out of the back of the (right) hand
    return np.vstack([along, thumb, back])


MIN_POSE_ANGLE = 60.0          # "fingers up" and "thumb up" must point at least this far apart


def axis_map(axes=None):
    """IMU_AXES ("+x", "-z", ...) as a 3x3 matrix that turns IMU axes into glove axes."""
    axes = axes or settings.IMU_AXES
    matrix = np.zeros((3, 3))
    for row, name in enumerate(axes):
        sign = -1.0 if name.strip().startswith("-") else 1.0
        matrix[row, "xyz".index(name.strip()[-1].lower())] = sign
    return matrix


class MPU6050:
    """The real IMU, over I2C. read() returns (accel in g, gyro in degrees per second), glove axes."""

    WHO_AM_I = {0x68: "MPU-6050", 0x70: "MPU-6500", 0x71: "MPU-9250", 0x73: "MPU-9255"}

    def __init__(self, bus=None, address=None):
        try:
            from smbus2 import SMBus
        except ImportError:
            from smbus import SMBus                  # python3-smbus, from apt
        self.address = settings.IMU_ADDRESS if address is None else address
        self.bus = SMBus(settings.IMU_I2C_BUS if bus is None else bus)
        chip = self.bus.read_byte_data(self.address, 0x75)
        self.name = self.WHO_AM_I.get(chip, f"unknown chip 0x{chip:02x}")
        self.bus.write_byte_data(self.address, 0x6B, 0x00)   # wake up (it starts asleep)
        time.sleep(0.05)
        self.bus.write_byte_data(self.address, 0x1A, 0x03)   # 44 Hz low-pass filter: smooths hand tremor
        self.bus.write_byte_data(self.address, 0x1B, 0x00)   # gyro range +/-250 deg/s: 131 counts per deg/s
        self.bus.write_byte_data(self.address, 0x1C, 0x00)   # accel range +/-2 g: 16384 counts per g
        self.axes = axis_map()

    def read(self):
        raw = self.bus.read_i2c_block_data(self.address, 0x3B, 14)   # accel x,y,z, temperature, gyro x,y,z
        values = [int.from_bytes(bytes(raw[i:i + 2]), "big", signed=True) for i in range(0, 14, 2)]
        accel = np.array(values[0:3], dtype=float) / 16384.0
        gyro = np.array(values[4:7], dtype=float) / 131.0
        return self.axes @ accel, self.axes @ gyro

    def close(self):
        self.bus.close()


class ImuSampler:
    """Reads the IMU IMU_SAMPLE_HZ times per second in a background thread, like the flex-sensor Sampler.

    state() gives the (tilt, motion) that hand_features() needs.
    """

    def __init__(self, imu):
        self.imu = imu
        self.window = deque(maxlen=max(2, int(settings.IMU_SAMPLE_HZ * max(settings.MOTION_SECONDS, 1.0))))
        self.lock = threading.Lock()
        self.gyro_offset = np.zeros(3)              # the gyro reads a few deg/s even when still
        self.rotation = np.eye(3)                   # IMU axes -> glove axes, from calibration
        self.running = False
        self.errors = 0
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.running = True
        self.thread.start()
        time.sleep(0.1)                             # let the first readings arrive

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

    def run(self):
        period = 1.0 / settings.IMU_SAMPLE_HZ
        next_time = time.monotonic()
        while self.running:
            try:
                accel, gyro = self.imu.read()
                with self.lock:
                    self.window.append((time.monotonic(), np.asarray(accel, float), np.asarray(gyro, float)))
            except OSError:                         # a loose I2C wire: skip this reading
                self.errors += 1
            next_time += period
            time.sleep(max(0.0, next_time - time.monotonic()))

    def set_orientation(self, rotation, gyro_offset=None):
        """Use a calibration's hand directions (and gyro still reading) from now on."""
        self.rotation = np.asarray(rotation, dtype=float)
        if gyro_offset is not None:
            self.gyro_offset = np.asarray(gyro_offset, dtype=float)

    def recent(self, seconds):
        """The raw readings from the last `seconds`: (accel array n x 3, gyro array n x 3), IMU axes,
        before the calibration's rotation and gyro offset. Calibration measures with these."""
        now = time.monotonic()
        with self.lock:
            part = [(a, g) for t, a, g in self.window if now - t <= seconds]
        if not part:
            return np.zeros((0, 3)), np.zeros((0, 3))
        return np.array([a for a, _ in part]), np.array([g for _, g in part])

    def zero_gyro(self):
        """Make the gyro's still reading zero (it reads a few deg/s even when still).

        Only done if the last half second really was still (every axis steady within
        3 deg/s), so zeroing while the hand is moving can't spoil the motion reading.
        Returns True if it zeroed.
        """
        with self.lock:
            recent = [g for _, _, g in list(self.window)[-int(settings.IMU_SAMPLE_HZ * 0.5):]]
        if len(recent) < 5 or np.max(np.std(recent, axis=0)) > 3.0:
            return False
        self.gyro_offset = np.mean(recent, axis=0)
        return True

    def latest(self):
        """The newest reading: (accel in g, gyro in deg/s, gyro offset removed), glove axes."""
        with self.lock:
            if not self.window:
                return np.array([1.0, 0.0, 0.0]), np.zeros(3)
            _, accel, gyro = self.window[-1]
        return self.rotation @ accel, self.rotation @ (gyro - self.gyro_offset)

    def state(self):
        """(tilt, motion): tilt is the unit 'up' direction, motion each axis's turning speed, 0 to 1."""
        now = time.monotonic()
        with self.lock:
            readings = list(self.window)
        if not readings:
            return STILL_STATE
        tilt_part = [a for t, a, _ in readings if now - t <= settings.TILT_SECONDS] or [readings[-1][1]]
        up = self.rotation @ np.mean(tilt_part, axis=0)
        size = np.linalg.norm(up)
        tilt = up / size if size > 0.3 else STILL_STATE[0]        # in free fall or a bad read: assume up
        motion_part = [g for t, _, g in readings if now - t <= settings.MOTION_SECONDS]
        if not motion_part:
            return tilt, np.zeros(3)
        turning = (np.array(motion_part) - self.gyro_offset) @ self.rotation.T     # into glove axes
        speed = np.mean(np.abs(turning), axis=0)
        return tilt, np.clip(speed / settings.MOTION_FULL_SCALE_DPS, 0.0, 1.0)


def open_imu():
    """The IMU's background sampler, started, or None if USE_IMU is off or no IMU answers."""
    if not getattr(settings, "USE_IMU", False):
        return None
    try:
        imu = MPU6050()
    except Exception as error:                      # not wired, I2C off, smbus missing...
        print(f"[IMU] Gyroscope not found ({error}). Carrying on with the flex sensors only "
              "(J and Z can't be read). Check the wiring and that I2C is on.")
        return None
    print(f"[IMU] {imu.name} found at 0x{imu.address:02x}")
    sampler = ImuSampler(imu)
    sampler.start()
    return sampler


def main():
    """Print tilt and motion, to check the wiring and IMU_AXES."""
    sampler = open_imu()
    if sampler is None:
        raise SystemExit("No IMU. Is USE_IMU = True in settings.py, is I2C on (sudo raspi-config), "
                         "and does  i2cdetect -y 1  show 68?")
    from calibration import apply_saved_imu_calibration
    if apply_saved_imu_calibration(sampler, settings.CAL_FILE):
        print("Using the saved calibration's gyroscope directions.")
    time.sleep(0.6)
    if sampler.zero_gyro():
        print("Gyro still reading zeroed (keep the board still while this starts).")
    print("Point your fingers up: tilt about (1, 0, 0). Thumb up: (0, 1, 0). Back of hand up: (0, 0, 1).")
    print("Twist and wave your hand to see motion. Ctrl+C quits.")
    try:
        while True:
            tilt, motion = sampler.state()
            print(f"tilt {tilt[0]:+.2f} {tilt[1]:+.2f} {tilt[2]:+.2f}    "
                  f"motion {motion[0]:.2f} {motion[1]:.2f} {motion[2]:.2f}"
                  + ("   MOVING" if is_moving(motion) else ""))
            time.sleep(0.2)
    except KeyboardInterrupt:
        print()
    finally:
        sampler.stop()


if __name__ == "__main__":
    main()
