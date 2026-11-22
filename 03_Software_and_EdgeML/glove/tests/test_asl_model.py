"""Tests for the full-alphabet network: look-alike groups, smoothing, saving, lessons and the trainer."""

import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))   # lets the tests import the files in glove/

import settings
import sign_feedback
import train_model
from alphabet import EXPECTED_POSES, HOW_TO, MOTION_LETTERS, STATIC_LETTERS
from asl_model import NONE, GloveModel, Smoother, group_label, letters_in, train_network

LETTERS = ["A", "B", "U", "V", "Y"]          # U and V are identical to a flex sensor


def dataset(letters=LETTERS, per_letter=60, seed=1):
    """Pretend recordings: each letter's expected pose plus noise, and random NONE readings."""
    rng = np.random.default_rng(seed)
    bends, labels = [], []
    for letter in letters:
        bends += list(np.clip(np.array(EXPECTED_POSES[letter]) + rng.normal(0, 0.03, (per_letter, 5)), 0, 1))
        labels += [letter] * per_letter
    bends += list(rng.uniform(0, 1, (per_letter, 5)))
    labels += [NONE] * per_letter
    return np.array(bends), np.array(labels)


class Trained(unittest.TestCase):
    """Trains one small network for every test in this file (about a second)."""

    @classmethod
    def setUpClass(cls):
        bends, labels = dataset()
        cls.model = train_network(bends, labels, LETTERS + [NONE], epochs=60)
        cls.model.groups = [["U", "V"]]
        cls.model.references = {k: EXPECTED_POSES[k] for k in LETTERS}


class AlphabetTests(unittest.TestCase):

    def test_24_static_letters_and_two_motion_letters(self):
        self.assertEqual(len(STATIC_LETTERS), 24)
        self.assertEqual(set(MOTION_LETTERS), {"J", "Z"})

    def test_every_static_letter_has_a_pose_and_a_prompt(self):
        for letter in STATIC_LETTERS:
            self.assertEqual(len(EXPECTED_POSES[letter]), 5, letter)
            self.assertIn(letter, HOW_TO)


class GroupTests(Trained):

    def test_distinct_letters_are_shown_alone(self):
        for letter in ("A", "B", "Y"):
            sign, confidence, _ = self.model.recognize_with_margin(EXPECTED_POSES[letter], max_distance=0.15)
            self.assertEqual(sign, letter)
            self.assertGreater(confidence, 0.8)

    def test_look_alikes_are_shown_together_and_confidently(self):
        sign, confidence, margin = self.model.recognize_with_margin(EXPECTED_POSES["V"], max_distance=0.15)
        self.assertEqual(sign, "U/V")
        self.assertGreater(confidence, 0.8)       # the group's total, so the U/V split doesn't matter
        self.assertGreater(margin, 0.1)

    def test_active_letters_narrow_a_group(self):
        sign, _, _ = self.model.recognize_with_margin(EXPECTED_POSES["V"], active_letters=["A", "U"])
        self.assertEqual(sign, "U")

    def test_letter_outside_active_letters_shows_nothing(self):
        self.assertIsNone(self.model.recognize_with_margin(EXPECTED_POSES["Y"], active_letters=["A", "B"])[0])

    def test_far_from_every_recording_shows_nothing(self):
        odd = [1.0, 0.0, 1.0, 0.0, 1.0]           # no letter in the training data looks like this
        self.assertIsNone(self.model.recognize_with_margin(odd, max_distance=0.15)[0])

    def test_group_of(self):
        self.assertEqual(self.model.group_of("V"), ["U", "V"])
        self.assertEqual(self.model.group_of("A"), ["A"])

    def test_fast_enough_for_the_pi(self):
        start = time.perf_counter()
        for _ in range(200):
            self.model.recognize_with_margin(EXPECTED_POSES["V"], max_distance=0.15)
        # The main loop allows 50 ms per reading; the Pi Zero is roughly 20x slower than a laptop.
        self.assertLess((time.perf_counter() - start) / 200 * 20, 0.025)


class LabelTests(unittest.TestCase):

    def test_labels_fit_the_wrist_display(self):
        self.assertEqual(group_label(["U", "V"]), "U/V")
        self.assertEqual(group_label(list("AEMNST")), "AEMNST")      # 6+ letters: no slashes

    def test_letters_in(self):
        self.assertEqual(letters_in("U/V"), ["U", "V"])
        self.assertEqual(letters_in("AEMNST"), list("AEMNST"))
        self.assertEqual(letters_in("--"), [])


class SaveLoadTests(Trained):

    def test_round_trip_keeps_groups_and_references(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.npz"
            self.model.save(path)
            loaded = GloveModel.load(path)
        self.assertEqual(loaded.groups, [["U", "V"]])
        self.assertEqual(loaded.references["B"], EXPECTED_POSES["B"])
        np.testing.assert_allclose(loaded.probabilities(EXPECTED_POSES["A"]),
                                   self.model.probabilities(EXPECTED_POSES["A"]))

    def test_a_model_saved_before_groups_still_loads(self):
        old = GloveModel(self.model.weights, self.model.classes, self.model.mean, self.model.std,
                         self.model.points, self.model.point_labels)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old.npz"
            old.save(path)
            loaded = GloveModel.load(path)
        self.assertEqual(loaded.groups, [])
        self.assertEqual(loaded.recognize_with_margin(EXPECTED_POSES["V"])[0] in ("U", "V"), True)


class SmootherTests(unittest.TestCase):

    def test_one_odd_reading_is_outvoted(self):
        smoother = Smoother(5)
        for _ in range(4):
            smoother.update([0.9, 0.1])
        self.assertGreater(smoother.update([0.1, 0.9])[0], 0.5)

    def test_only_the_last_few_count(self):
        smoother = Smoother(2)
        smoother.update([1.0, 0.0])
        smoother.update([0.0, 1.0])
        np.testing.assert_allclose(smoother.update([0.0, 1.0]), [0.0, 1.0])


class LessonReferenceTests(Trained):

    def setUp(self):
        sign_feedback._model_cache = (Path("test"), self.model)
        self.patch = mock.patch.multiple(settings, USE_MODEL=True, MODEL_FILE="test")
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        sign_feedback._model_cache = None

    def test_lessons_can_teach_a_letter_only_the_model_knows(self):
        self.assertNotIn("Y", settings.REFERENCES)
        self.assertEqual(sign_feedback.evaluate(np.array(EXPECTED_POSES["Y"]), "Y"), "CORRECT")

    def test_one_finger_off_is_named(self):
        bends = np.array(EXPECTED_POSES["Y"])
        bends[4] = 1.0                            # pinky curled instead of out
        self.assertEqual(sign_feedback.evaluate(bends, "Y"), "ADJUST: Pinky")

    def test_without_the_model_references_come_from_settings(self):
        with mock.patch.object(settings, "USE_MODEL", False):
            np.testing.assert_allclose(sign_feedback.reference_for("A"), settings.REFERENCES["A"])


class TrainerTests(unittest.TestCase):

    def test_find_groups_joins_mix_ups_and_chains_them(self):
        truth = np.array(["U"] * 10 + ["V"] * 10 + ["R"] * 10 + ["B"] * 10)
        top = np.array(["U"] * 6 + ["V"] * 4 + ["V"] * 10 + ["R"] * 7 + ["U"] * 2 + ["V"] * 1 + ["B"] * 10)
        # U->V 40%: joined. R goes 20% to U and 10% to V, 30% to the U/V group: joined too.
        self.assertEqual(train_model.find_groups(truth, top, ["B", "R", "U", "V"]), [["R", "U", "V"]])

    def test_rare_mix_ups_stay_separate(self):
        truth = np.array(["A"] * 20 + ["B"] * 20)
        top = np.array(["A"] * 19 + ["B"] + ["B"] * 20)        # 5%: below MERGE_CONFUSION
        self.assertEqual(train_model.find_groups(truth, top, ["A", "B"]), [])

    def test_a_group_label_counts_as_right(self):
        right, wrong, _ = train_model.score(["V", "V", "V", "V"], ["U/V", "V", "U", None])
        self.assertEqual((right, wrong), (0.5, 0.25))

    def test_pretend_glove_data_is_spotted(self):
        mock_row = {f"flat_{f.lower()}": str(v) for f, v in zip(settings.FINGERS, [2250.7, 2300, 2279, 2260, 2241])}
        real_row = {f"flat_{f.lower()}": str(v) for f, v in zip(settings.FINGERS, [2410, 2190, 2302, 2255, 2280])}
        self.assertTrue(train_model.looks_like_mock(mock_row))
        self.assertFalse(train_model.looks_like_mock(real_row))


if __name__ == "__main__":
    unittest.main()
