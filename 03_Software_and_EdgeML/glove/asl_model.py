"""
The ASL letter network: a small neural network that reads the five finger bends,
and the gyroscope's tilt and motion when the glove has one (imu.py).

Why numpy only: the Raspberry Pi Zero W (ARMv6) can't install TensorFlow or
PyTorch, but it already has numpy. A network this size needs only a few
matrix multiplications per reading, so it runs in about a millisecond on the Pi.

The network (a multilayer perceptron):

    5 or 11 inputs  ->  32 neurons (tanh)  ->  32 neurons (tanh)  ->  one score per letter (softmax)

    Input:  Thumb, Index, Middle, Ring, Pinky, each 0.0 (flat) to 1.0 (fist),
            from sign_feedback.normalize() with this user's calibration.
            With the gyroscope, 6 more (imu.hand_features()): which way is up (3 numbers,
            -1 to 1) and how fast the hand turns (3 numbers, 0 to 1). Tilt splits letters
            that bend the same but point differently (G/L/Q, K/P, H/U); motion splits J
            from I and Z from a still pointing hand.
    Output: a probability for every letter it was trained on, plus "NONE"
            (a hand that isn't making any supported letter).

Look-alike groups: some letters still read the same (U and V differ only in spread,
which no sensor on the glove feels). train_model.py measures which letters the network
confuses on people it hasn't seen, and saves those as groups. By default the glove still
shows ONE letter, the most likely (settings.SHOW_LOOK_ALIKE_GROUPS = False); turned on,
it adds up a group's probabilities and shows "U/V".

Smoothing: the probabilities are averaged over the last few readings (settings.
MODEL_SMOOTHING_FRAMES), so one noisy reading can't make the shown letter flicker.

Two extra safety checks run after the network, so it shows "--" instead of guessing:
  * NONE wins, or the best letter isn't in settings.ACTIVE_LETTERS
  * Distance check: the hand is further than MODEL_MAX_DISTANCE (average finger
    error) from every recorded example of that letter. A network always picks
    *something*, even for a hand shape it has never seen; this stops that.

train_model.py builds and tests the network. This file is all the Pi needs to run it.
"""

import json
from collections import deque
from pathlib import Path

import numpy as np

NONE = "NONE"                 # the label for "not a supported letter"
HIDDEN = (32, 32)             # neurons in each hidden layer (enough for all 26 letters)
MAX_POINTS_PER_LETTER = 300   # recorded examples kept in the model file for the distance check


class GloveModel:
    """A trained network plus what it needs to run: the class names, input scaling and examples."""

    def __init__(self, weights, classes, mean, std, points, point_labels, info=None,
                 groups=None, references=None):
        self.weights = [np.asarray(w, dtype=float) for w in weights]   # W1, b1, W2, b2, W3, b3
        self.classes = [str(c) for c in classes]
        self.mean = np.asarray(mean, dtype=float)                       # input scaling, from the training data
        self.std = np.asarray(std, dtype=float)
        self.points = np.asarray(points, dtype=float)                   # recorded letter examples
        self.point_labels = np.asarray(point_labels).astype(str)
        self.info = info or {}
        # Look-alike groups, e.g. [["U", "V"]]. Every other letter is a group of its own.
        self.groups = [sorted(g) for g in (groups or []) if len(g) > 1]
        # Each letter's typical pose (median of every recording): used for the lesson's
        # match score and "ADJUST: finger" feedback, so lessons work for every trained letter.
        self.references = {k: [float(x) for x in v] for k, v in (references or {}).items()}

    def letters(self):
        return [c for c in self.classes if c != NONE]

    @property
    def n_inputs(self):
        """5 (bends only) or 11 (bends + gyroscope tilt and motion)."""
        return len(self.mean)

    def group_of(self, letter):
        """The letters shown together with this one, e.g. group_of("V") -> ["U", "V"]."""
        for group in self.groups:
            if letter in group:
                return group
        return [letter]

    # ----------------------------------------------------------------- running
    def probabilities(self, bends):
        """Each class's probability for one hand (5 values) or many hands (n x 5)."""
        x = np.atleast_2d(np.asarray(bends, dtype=float))
        _, _, probs = forward(self.weights, (x - self.mean) / self.std)
        return probs if np.ndim(bends) > 1 else probs[0]

    def nearest_distance(self, bends, letter):
        """Average error (fingers, and tilt and motion if trained with them) to the closest
        recorded example of this letter (0.0 = identical)."""
        examples = self.points[self.point_labels == letter]
        if len(examples) == 0:
            return np.inf
        return float(np.min(np.mean(np.abs(examples - np.asarray(bends, dtype=float)), axis=1)))

    def recognize_with_margin(self, bends, active_letters=None, max_distance=None, smoother=None,
                              use_groups=True):
        """Return (label or None, probability, lead over the second-best).

        Same three values as sign_feedback.recognize_with_margin(), so main.py's
        ACCEPT_CONFIDENCE and MARGIN checks work on it unchanged. The label is one
        letter ("B") or a look-alike group ("U/V"), and the probability is the
        group's total.
        None means "show --": the hand is NONE, no letter of the group is in
        active_letters, or the hand is too far from anything recorded.
        smoother: a Smoother, to average this reading with the last few.
        use_groups: False shows one letter even for look-alikes (the most likely of them).
        """
        probs = self.probabilities(bends)
        if smoother is not None:
            probs = smoother.update(probs)
        index = {c: i for i, c in enumerate(self.classes)}

        # One entry per group (or single letter), plus NONE, with its total probability.
        options, seen = [], set()
        for letter in self.letters():
            if letter in seen:
                continue
            group = self.group_of(letter) if use_groups else [letter]
            seen.update(group)
            options.append((float(sum(probs[index[g]] for g in group if g in index)), group))
        if NONE in index:
            options.append((float(probs[index[NONE]]), [NONE]))
        options.sort(key=lambda option: -option[0])

        best, group = options[0]
        margin = best - options[1][0] if len(options) > 1 else best
        if group == [NONE]:
            return None, best, margin
        if active_letters:
            group = [g for g in group if g in active_letters]
            if not group:
                return None, 0.0, 0.0
        if max_distance is not None and min(self.nearest_distance(bends, g) for g in group) > max_distance:
            return None, 0.0, 0.0
        return group_label(group), best, margin

    # ----------------------------------------------------------- saving/loading
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"w{i}": w for i, w in enumerate(self.weights)}
        temporary = path.with_name(path.stem + ".tmp.npz")             # a power cut can't leave half a file
        np.savez(temporary, classes=np.array(self.classes), mean=self.mean, std=self.std,
                 points=self.points, point_labels=self.point_labels,
                 info=np.array(json.dumps(self.info)), groups=np.array(json.dumps(self.groups)),
                 references=np.array(json.dumps(self.references)), **arrays)
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            count = sum(1 for key in data.files if key[0] == "w" and key[1:].isdigit())
            weights = [data[f"w{i}"] for i in range(count)]
            extra = {key: json.loads(str(data[key])) for key in ("groups", "references") if key in data.files}
            return cls(weights, data["classes"], data["mean"], data["std"],
                       data["points"], data["point_labels"], json.loads(str(data["info"])), **extra)


def group_label(group):
    """"U/V" for a group. Six or more letters drop the slashes ("AEMNST") to fit the 128-pixel wrist display."""
    return "/".join(group) if len(group) <= 5 else "".join(group)


def letters_in(label):
    """The letters a shown label stands for: "U/V" -> ["U", "V"], "B" -> ["B"]."""
    return list(label.replace("/", "")) if label not in (None, "--", NONE) else []


class Smoother:
    """Averages the network's probabilities over the last few readings (20 per second).

    5 readings is a quarter of a second: enough to hide one noisy reading,
    short enough that changing letters still feels instant.
    """

    def __init__(self, frames=5):
        self.recent = deque(maxlen=max(1, int(frames)))

    def update(self, probs):
        self.recent.append(np.asarray(probs, dtype=float))
        return np.mean(self.recent, axis=0)

    def reset(self):
        self.recent.clear()


# ============================================================================
# The network's maths (used by both running and training)
# ============================================================================

def forward(weights, x):
    """Run scaled inputs through the network. Returns both hidden layers and the probabilities."""
    w1, b1, w2, b2, w3, b3 = weights
    h1 = np.tanh(x @ w1 + b1)
    h2 = np.tanh(h1 @ w2 + b2)
    logits = h2 @ w3 + b3
    logits -= logits.max(axis=1, keepdims=True)                         # keeps exp() from overflowing
    exp = np.exp(logits)
    return h1, h2, exp / exp.sum(axis=1, keepdims=True)


def augment(features, rng):
    """Make slightly different copies of the recordings, so the network copes with new hands.

    Each finger gets a random gain (+/-15%) and offset (about +/-3%), which is roughly
    how much the same sign shifts between people and between calibrations. With the
    gyroscope's 6 numbers too, the tilt is nudged a little (the glove sits slightly
    differently on each wrist) and the motion's size varies by +/-25%.
    """
    features = np.asarray(features, dtype=float)
    bends = features[:, :5]
    gain = rng.uniform(0.85, 1.15, bends.shape)
    offset = rng.normal(0.0, 0.03, bends.shape)
    out = [np.clip(bends * gain + offset, 0.0, 1.0)]                    # normalize() clips the same way
    if features.shape[1] >= 11:
        tilt = features[:, 5:8] + rng.normal(0.0, 0.08, (len(features), 3))
        tilt /= np.maximum(np.linalg.norm(tilt, axis=1, keepdims=True), 1e-6)
        motion = np.clip(features[:, 8:11] * rng.uniform(0.75, 1.25, (len(features), 1)), 0.0, 1.0)
        out += [tilt, motion, features[:, 11:]]
    return np.hstack(out)


def train_network(bends, labels, classes, epochs=150, batch=128, learning_rate=0.01,
                  l2=1e-4, seed=0, log=None):
    """Train a GloveModel on recorded inputs (n x 5 bends, or n x 11 with the gyroscope) and their labels.
    Plain numpy, Adam optimizer."""
    rng = np.random.default_rng(seed)
    bends = np.asarray(bends, dtype=float)
    labels = np.asarray(labels).astype(str)
    targets = np.array([classes.index(label) for label in labels])

    # Every class counts equally, even if one (often NONE) has many more readings.
    counts = np.bincount(targets, minlength=len(classes)).astype(float)
    class_weight = len(targets) / (len(classes) * np.maximum(counts, 1.0))

    mean, std = bends.mean(axis=0), bends.std(axis=0) + 1e-6

    sizes = (bends.shape[1],) + HIDDEN + (len(classes),)
    weights = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):                      # Xavier start: small random weights
        weights += [rng.normal(0.0, np.sqrt(1.0 / n_in), (n_in, n_out)), np.zeros(n_out)]
    m = [np.zeros_like(w) for w in weights]                             # Adam's running averages
    v = [np.zeros_like(w) for w in weights]
    beta1, beta2, step = 0.9, 0.999, 0

    for epoch in range(epochs):
        order = rng.permutation(len(bends))
        for start in range(0, len(order), batch):
            index = order[start:start + batch]
            x = (augment(bends[index], rng) - mean) / std
            t = targets[index]
            sample_weight = class_weight[t] / class_weight[t].sum()

            h1, h2, probs = forward(weights, x)
            # Backpropagation of the weighted cross-entropy loss.
            d3 = probs.copy()
            d3[np.arange(len(t)), t] -= 1.0
            d3 *= sample_weight[:, None]
            d2 = (d3 @ weights[4].T) * (1.0 - h2 ** 2)
            d1 = (d2 @ weights[2].T) * (1.0 - h1 ** 2)
            grads = [x.T @ d1, d1.sum(0), h1.T @ d2, d2.sum(0), h2.T @ d3, d3.sum(0)]
            for i in (0, 2, 4):
                grads[i] += l2 * weights[i]                             # weight decay: keeps the network simple

            step += 1
            for i, g in enumerate(grads):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * g * g
                m_hat = m[i] / (1 - beta1 ** step)
                v_hat = v[i] / (1 - beta2 ** step)
                weights[i] -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

        if log and (epoch + 1) % 50 == 0:
            _, _, probs = forward(weights, (bends - mean) / std)
            accuracy = np.mean(np.argmax(probs, axis=1) == targets)
            log(f"  epoch {epoch + 1:>4}: training accuracy {accuracy:.1%}")

    # Keep a sample of each letter's recordings for the distance check.
    keep = []
    for letter in classes:
        if letter == NONE:
            continue
        index = np.flatnonzero(labels == letter)
        if len(index) > MAX_POINTS_PER_LETTER:
            index = rng.choice(index, MAX_POINTS_PER_LETTER, replace=False)
        keep.extend(index)
    keep = np.array(sorted(keep), dtype=int)
    return GloveModel(weights, classes, mean, std, bends[keep], labels[keep])
