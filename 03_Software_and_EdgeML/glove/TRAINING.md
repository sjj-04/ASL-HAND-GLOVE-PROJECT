# Training the letter model (all 26 letters)

With the gyroscope (`imu.py`), the glove can learn **all 26 letters**. J and Z are traced in the air: the gyroscope's motion reading is what separates a traced J from a still I. Without the gyroscope it learns the 24 still letters (A to Y except J).

The model is a small neural network (`asl_model.py`) that runs on the Pi Zero with only numpy. It reads 11 numbers and returns **one letter**:

- the five finger bends from `normalize()`, after this user's flat/fist calibration
- **tilt** (3 numbers): which way is up for the hand. G and H point sideways, P and Q point down, so they split from L, U and K
- **motion** (3 numbers): how fast the hand is turning. J and Z move; every other letter is held still

A model trained without the gyroscope reads just the 5 bends, as before.

**It is off by default** (`USE_MODEL = False` in `settings.py`). Until you turn it on, the glove works exactly as before.

## What's different from the hand-typed REFERENCES

| | `settings.REFERENCES` | Trained model |
|---|---|---|
| Letters | the 8 typed in (J and Z only while moving) | all 26 recorded |
| Where the poses come from | guesses | everyone's recordings |
| Letters the sensors can't tell apart | guesses one, often wrong | shows `--` when it can't decide, never two letters |
| Hand that isn't a letter | shows the nearest letter | shows `--` (learns a NONE class, plus a distance check) |
| Lessons | only the 6 letters | any trained letter, graded against its measured pose |
| Flicker | none | averaged over 5 readings (0.25 s) |

## Look-alike letters

A flex sensor measures how far a finger bends, and nothing else. The gyroscope adds which way the hand points and how it moves, which splits:

- **G / L / Q**, **K / P** and **H / U**: same shape, pointed a different way (tilt)
- **I / J** and **Z / a pointing hand**: same shape, but J and Z are traced (motion)

Nothing on the glove measures finger spread, crossing, or where the thumb sits, so these can still read the same:

- **U / V / R**: only spread or crossing differ
- **A / S / T / M / N**: all fists; only the thumb position differs

The glove always shows **one letter at a time** (`SHOW_LOOK_ALIKE_GROUPS = False`). When it can't decide between look-alikes, the margin rule shows `--` (or keeps the last letter) rather than guessing. `train_model.py` tests the network on people it wasn't trained on and lists the letters it mixes up on 20% or more of their readings, so you know which ones to leave out of a demo.

## Steps

**1. Check the gyroscope** (`python3 imu.py`: fingers up reads about (1, 0, 0)). It must work for every recording: if any CSV has no gyroscope readings, the trainer uses the bends only for everyone.

**2. Record each person (about 9 minutes each, on the Pi with the glove on).** All five of you:

```bash
python3 collect_data.py --person zion
python3 collect_data.py --person amad
python3 collect_data.py --person milly
python3 collect_data.py --person sleytters
python3 collect_data.py --person chris
```

Use exactly these names (they're `TEAM` in `settings.py`): a typo like `zoin` counts as a different person, and `collect_data.py` warns you. `train_model.py` lists anyone on the team who hasn't been recorded yet.

The terminal says how to form each letter. For still letters: make the sign, hold still until it says Saved, then relax. **For J and Z: make the handshape and trace the letter, again and again, until it says Saved** (about a second of tracing). Point the hand the way the letter really points (G sideways, P and Q down): that's what the gyroscope learns from. At the end, move your hand around freely for 20 seconds (the "not a letter" recording). Ctrl+C keeps everything recorded so far. Record more people if you can: **the more different hands, the better it works on a stranger.**

Only want some letters, or adding to an earlier recording? `--person milly --letters J,Z`. Every CSV in `data/` is used. CSVs recorded before the gyroscope was fitted have empty tilt and motion columns: move them out of `data/` (or record those people again), or the gyroscope won't be used.

Each CSV row has the five bends, `tilt_along/thumb/back`, `motion_along/thumb/back`, the raw ADC readings, the raw gyroscope board readings (`accel_*` in g, `gyro_*` in degrees per second) and that session's calibration.

**3. Train (on a laptop, about 15 seconds; on the Pi Zero, several minutes).** Copy `data/` to a laptop with Python and numpy, then:

```bash
python3 train_model.py
```

**4. Read the results.** The row you care about is **Neural network** (one letter at a time, what the glove shows), measured on people the network was never trained on:

- **right letter shown**: higher is better
- **wrong letter shown**: should be near 0%. At a demo, a wrong letter is worse than `--`
- **letter during NONE**: how often a random hand movement shows a letter

Then check the per-letter table. A letter that mostly shows `--` needs more recordings, or is too close to the "not a letter" movement.

**5. Turn it on.** Copy `models/asl_model.npz` to the same folder on the Pi, set `USE_MODEL = True` in `settings.py`, and run `main.py` as usual. The terminal prints which letters it knows. A model trained with the gyroscope needs it: if the gyroscope isn't working, the glove falls back to `REFERENCES` and says so.

## For a demo

To show only certain letters, set `ACTIVE_LETTERS = ["A", "B", "C"]` in `settings.py`. Anything else shows `--`. Leaving out one of each look-alike pair (keep U, drop V and R) makes the others reliable.

Pick demo letters that the per-letter table shows near 100% right and 0% wrong. Don't switch the model on right before a presentation without testing it on the real glove first.

## Tuning (settings.py)

| Setting | What it does |
|---|---|
| `MODEL_MAX_DISTANCE` (0.15) | shows `--` if the hand is this far from every recording of the letter. Raise it if the trainer says it turns away more than ~10% of real letters |
| `MODEL_SMOOTHING_FRAMES` (5) | readings averaged together. Higher is steadier and slower to change |
| `ACCEPT_CONFIDENCE`, `MARGIN` | the same "sure enough" rules as before |
| `MOTION_START` (0.12) | how much turning counts as "moving" (about 30 deg/s): J and Z need more, still letters less |
| `MOTION_HOLD_SECONDS` (0.3) | how long J or Z must be read while tracing before it's shown |
| `SHOW_LOOK_ALIKE_GROUPS` (False) | True shows look-alikes together (`U/V`) instead of one letter |

`train_model.py --merge 0.3` groups less eagerly (more single letters, more mistakes). `--no-groups` turns grouping off.

## Pretend-glove data

`collect_data.py --mock` saves to `data/mock/`, which training skips unless you ask for it (`--data data/mock`). If a file in `data/` has the pretend glove's exact calibration, `train_model.py` warns you. `data/mock/full26/` is a 26-letter pretend dataset with gyroscope readings for all five of you (Zion, Amad, Sleytters, Milly, Chris), for trying the pipeline (`data/mock/full24/` is the older one without the gyroscope):

```bash
python3 train_model.py --data data/mock/full26 --no-save
```

It shows about 77% right and 4% wrong one letter at a time: G, H, J, K, L, P, Q and Z are read correctly, and the misses are U/V/R and M/N/S/T, which the pretend gyroscope can't split either. Pretend hands vary less than real ones, so expect lower numbers on the glove.

## Splitting the rest

The gyroscope handles G/L/Q, K/P, H/U, J and Z. More training can't split letters that no sensor tells apart. That takes another sensor:

- **A contact sensor (a strip of conductive fabric) between the index and middle fingers** splits U from V, and R needs one for crossing.
- **The fist letters (A/S/T/M/N)** need to know where the thumb sits, such as contact pads or a pressure sensor on the side of the index finger.

The network takes any number of inputs. Adding a sensor means adding its columns to `collect_data.py` and `imu.FEATURE_NAMES`-style inputs in `train_model.py`, then recording everyone again.
