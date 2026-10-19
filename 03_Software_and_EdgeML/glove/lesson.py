"""
The lesson: which sign the user is practicing, and what the Selection button does.

Stand-in for the teaching app on the HDMI monitor. That app will draw the
prompts and demonstrations; for now this keeps track of the target sign and
prints what the teaching screen would show.
"""

import settings


class LessonManager:

    def __init__(self, signs=settings.LESSON):
        self.signs = list(signs)
        self.index = None                                # None = no lesson running (independent use)

    @property                                            # read it like a variable: lesson.target
    def target(self):
        """The sign being practiced, or None when no lesson is running."""
        return None if self.index is None else self.signs[self.index]

    def select_next(self):
        """Selection button: start the lesson, or go to the next sign."""
        self.index = 0 if self.index is None else self.index + 1
        if self.index == len(self.signs):
            self.index = None                            # past the last sign: back to free practice
            print("[LESSON] Lesson complete. Free practice.")
        else:
            print(f"[LESSON] Form the sign: {self.target}")

    def report_success(self):
        """Success event from the glove: mark the sign complete and set the next target."""
        print(f"[LESSON] {self.target} complete")
        self.select_next()
