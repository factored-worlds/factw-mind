import pytest

from factw.mind.baselines.wordle import Belief, ContradictionError, MinimaxPolicy
from factw.mind.baselines.wordle.rules import feedback

WORDS = ("APPLE", "CRANE", "GRAPE", "LEMON", "MELON", "PEACH", "PLANT", "RAISE")


def test_minimax_partition_and_lexical_tie():
    policy = MinimaxPolicy(tuple(reversed(WORDS)))
    assert policy.choose(Belief(WORDS, ())) == {"guess": "APPLE"}


def test_minimax_finishes_known_answer():
    assert MinimaxPolicy(WORDS).choose(Belief(("RAISE",), ())) == {"guess": "RAISE"}


def test_minimax_rejects_empty_belief():
    with pytest.raises(ContradictionError):
        MinimaxPolicy(WORDS).choose(Belief((), ()))


@pytest.mark.parametrize("answer", WORDS)
def test_minimax_reaches_answer_from_exact_feedback(answer):
    policy = MinimaxPolicy(WORDS)
    candidates = WORDS
    for _ in range(6):
        guess = policy.choose(Belief(candidates, ()))["guess"]
        if guess == answer:
            return
        codes = feedback(guess, answer)
        candidates = tuple(w for w in candidates if feedback(guess, w) == codes)
    pytest.fail("minimax failed to finish within the native guess limit")
