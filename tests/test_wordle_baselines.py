import itertools
import json

import pytest

from factw.mind.baselines.wordle import (
    ContradictionError,
    ExactBeliefEstimator,
    InformationPolicy,
    RandomPolicy,
)
from factw.mind.baselines.wordle.rules import feedback
from factw.mind.baselines.wordle.session import WordleSession
from factw.mind.representations import ObservedTransition

WORDS = ("APPLE", "ALLEY", "CRANE", "GRAPE", "LEMON")


def public(guesses=(), codes=(), attempts=None, limit=6):
    won = bool(codes and tuple(codes[-1]) == (2,) * 5)
    return {
        "guesses": list(guesses),
        "feedbacks": [list(c) for c in codes],
        "attempts": len(guesses) if attempts is None else attempts,
        "max_guesses": limit,
        "max_attempts": 60,
        "won": won,
        "terminated": won or len(guesses) == limit,
        "truncated": False,
    }


def oracle(guess, target):
    greens = {i for i in range(5) if guess[i] == target[i]}
    result = [2 if i in greens else 0 for i in range(5)]
    for letter in set(guess):
        available = sum(target[i] == letter for i in range(5) if i not in greens)
        unmatched = [i for i in range(5) if i not in greens and guess[i] == letter]
        for i in unmatched[:available]:
            result[i] = 1
    return tuple(result)


def test_exact_belief_equals_independent_oracle_exhaustively():
    words = tuple("".join(w) for w in itertools.product("AB", repeat=5))
    estimator = ExactBeliefEstimator(words)
    initial = estimator.initialize(public())
    for target in words:
        for guess in words:
            codes = oracle(guess, target)
            expected = tuple(w for w in words if oracle(guess, w) == codes)
            state = estimator.initialize(public([guess], [codes]))
            assert state.candidates == expected
            assert target in state.candidates
            assert set(state.candidates) <= set(initial.candidates)
            assert feedback(guess, target) == codes


def test_history_intersection_is_monotonic_and_predictions_do_not_advance_progress():
    estimator = ExactBeliefEstimator(WORDS)
    session = WordleSession(estimator, InformationPolicy(WORDS), public())
    before = session.belief
    progress = session.progress
    predicted = session.predictor.predict(before, {"guess": "ALLEY"})
    assert session.belief is before and session.progress is progress
    assert sum(p.probability for p in predicted) == pytest.approx(1)
    state = before
    guesses, codes = [], []
    for guess in ("ALLEY", "GRAPE", "APPLE"):
        guesses.append(guess)
        codes.append(oracle(guess, "APPLE"))
        after = public(guesses, codes)
        updated = estimator.update(
            state,
            ObservedTransition({"guess": guess}, "won" if after["won"] else "guessed", after, 0),
        )
        assert set(updated.candidates) <= set(state.candidates)
        assert updated.candidates == tuple(
            w
            for w in sorted(WORDS)
            if all(oracle(g, w) == c for g, c in zip(guesses, codes, strict=True))
        )
        state = updated
    assert state.candidates == ("APPLE",)


def test_contradictions_and_privileged_input_are_explicit_errors():
    estimator = ExactBeliefEstimator(("APPLE",))
    with pytest.raises(ContradictionError):
        estimator.initialize(public(["APPLE"], [(0,) * 5]))
    with pytest.raises(ValueError, match="public"):
        estimator.initialize({**public(), "target": "APPLE"})


def test_rejected_action_keeps_belief_and_changes_only_observed_progress():
    estimator = ExactBeliefEstimator(WORDS)

    class InvalidPolicy:
        def choose(self, state):
            return {"guess": "ZZZZZ"}

    session = WordleSession(estimator, InvalidPolicy(), public())
    original = session.belief
    trace = session.step(
        lambda action: ObservedTransition(action, "illegal", public(attempts=1), 0)
    )
    assert session.belief is original
    assert session.progress.attempts == 1 and session.progress.accepted_guesses == 0
    assert not trace["postcondition_checked"]


def test_closed_loop_uses_only_feedback_and_checks_observed_postconditions():
    estimator = ExactBeliefEstimator(WORDS)
    session = WordleSession(estimator, InformationPolicy(WORDS), public())
    guesses, codes = [], []

    def execute(action):
        guesses.append(action["guess"])
        codes.append(oracle(action["guess"], "APPLE"))
        observed = public(guesses, codes)
        return ObservedTransition(
            action,
            "won" if observed["won"] else "lost" if observed["terminated"] else "guessed",
            observed,
            0,
        )

    while not session.progress.terminated:
        trace = session.step(execute)
        assert trace["postcondition_checked"]
    assert session.progress.won
    assert session.belief.candidates == ("APPLE",)


def test_wrong_executor_binding_does_not_advance_live_session():
    session = WordleSession(ExactBeliefEstimator(WORDS), InformationPolicy(WORDS), public())
    initial = session.progress
    with pytest.raises(ValueError, match="different bound action"):
        session.step(
            lambda action: ObservedTransition({"guess": "ZZZZZ"}, "illegal", public(attempts=1), 0)
        )
    assert session.progress == initial


@pytest.mark.parametrize("consistent", [False, True])
def test_random_policy_saved_rng_reproduces_continuation(consistent):
    policy = RandomPolicy(WORDS, seed=193, consistent=consistent)
    belief = ExactBeliefEstimator(WORDS).initialize(public())
    policy.choose(belief)
    state = json.loads(json.dumps(policy.snapshot()))
    expected = [policy.choose(belief) for _ in range(12)]
    restored = RandomPolicy(WORDS, seed=2, consistent=consistent)
    restored.restore(state)
    assert [restored.choose(belief) for _ in range(12)] == expected
