"""Tests for lesson.py: starting, advancing and finishing a lesson."""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parents[1]))   # lets the tests import the files in glove/

from lesson import LessonManager


class LessonManagerTests(unittest.TestCase):

    def setUp(self):
        quiet = mock.patch("builtins.print")             # the lesson prints what the teaching screen shows
        quiet.start()
        self.addCleanup(quiet.stop)
        self.lesson = LessonManager(["A", "B"])

    def test_no_target_until_the_lesson_starts(self):
        self.assertIsNone(self.lesson.target)            # independent use

    def test_selection_starts_the_lesson_then_skips_ahead(self):
        self.lesson.select_next()
        self.assertEqual(self.lesson.target, "A")
        self.lesson.select_next()
        self.assertEqual(self.lesson.target, "B")

    def test_success_moves_on_and_the_lesson_ends_in_free_practice(self):
        self.lesson.select_next()
        self.lesson.report_success()
        self.assertEqual(self.lesson.target, "B")
        self.lesson.report_success()
        self.assertIsNone(self.lesson.target)

    def test_selection_after_the_lesson_ends_starts_it_again(self):
        for _ in range(3):                               # start, skip to B, skip past the end
            self.lesson.select_next()
        self.assertIsNone(self.lesson.target)
        self.lesson.select_next()
        self.assertEqual(self.lesson.target, "A")


if __name__ == "__main__":
    unittest.main()
