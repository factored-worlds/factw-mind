import unittest
from copy import deepcopy

from factw.mind.examples.wordle_provider.demo import EXAMPLE_INPUTS, run_example


class WordleProviderTests(unittest.TestCase):
    def test_public_evidence_produces_an_exact_next_guess(self):
        result = run_example()["plan"]
        self.assertEqual(len(result["actions"]), 1)
        self.assertEqual(sum(p["count"] for p in result["predictions"]), len(result["candidates"]))

    def test_known_win_has_no_more_actions(self):
        public = deepcopy(EXAMPLE_INPUTS["knowledge"]["public"])
        public.update(guesses=["APPLE"], feedbacks=[[2] * 5], attempts=1, won=True, terminated=True)
        self.assertEqual(
            run_example(public)["plan"],
            {
                "actions": [],
                "candidates": ["APPLE"],
                "predictions": [],
            },
        )

    def test_privileged_answer_is_not_an_input(self):
        public = {**EXAMPLE_INPUTS["knowledge"]["public"], "target": "APPLE"}
        with self.assertRaises(ValueError):
            run_example(public)
