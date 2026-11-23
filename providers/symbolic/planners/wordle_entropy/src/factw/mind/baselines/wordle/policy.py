import math
import random
from collections import defaultdict
from dataclasses import dataclass

from .belief import Belief, ContradictionError
from .rules import feedback, vocabulary, word


@dataclass(frozen=True)
class PredictedFeedback:
    feedback: tuple[int, ...]
    candidates: tuple[str, ...]
    probability: float


class FeedbackPredictor:
    """Exact supplied model over a uniform candidate belief; never executes guesses."""

    def predict(self, state: Belief, action: dict) -> tuple[PredictedFeedback, ...]:
        if type(action) is not dict or set(action) != {"guess"}:
            raise ValueError("Expected a guess command")
        guess = word(action["guess"])
        if not state.candidates:
            raise ContradictionError("Cannot predict from an empty belief")
        groups = defaultdict(list)
        for candidate in state.candidates:
            groups[feedback(guess, candidate)].append(candidate)
        return tuple(
            PredictedFeedback(codes, tuple(candidates), len(candidates) / len(state.candidates))
            for codes, candidates in sorted(groups.items())
        )


class InformationPolicy:
    def __init__(self, allowed_guesses: tuple[str, ...]):
        self.allowed_guesses = vocabulary(allowed_guesses)
        self.predictor = FeedbackPredictor()

    def choose(self, state: Belief) -> dict:
        def rank(guess):
            outcomes = self.predictor.predict(state, {"guess": guess})
            entropy = -sum(p.probability * math.log2(p.probability) for p in outcomes)
            return (-entropy, guess not in state.candidates, guess)

        return {"guess": min(self.allowed_guesses, key=rank)}


class MinimaxPolicy:
    """Minimize the largest feedback partition; finish a known answer directly."""

    def __init__(self, allowed_guesses: tuple[str, ...]):
        self.allowed_guesses = vocabulary(allowed_guesses)
        self.predictor = FeedbackPredictor()

    def choose(self, state: Belief) -> dict:
        if len(state.candidates) == 1 and state.candidates[0] in self.allowed_guesses:
            return {"guess": state.candidates[0]}

        def rank(guess):
            outcomes = self.predictor.predict(state, {"guess": guess})
            return (max(len(part.candidates) for part in outcomes), guess)

        return {"guess": min(self.allowed_guesses, key=rank)}


class RandomPolicy:
    def __init__(
        self, allowed_guesses: tuple[str, ...], *, consistent: bool = False, seed: int = 0
    ):
        self.allowed_guesses = vocabulary(allowed_guesses)
        if type(consistent) is not bool:
            raise ValueError("consistent must be Boolean")
        self.consistent = consistent
        self.reset(seed=seed)

    def reset(self, *, seed: int = 0) -> None:
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.rng = random.Random(seed)

    def choose(self, state: Belief) -> dict:
        candidates = (
            tuple(w for w in state.candidates if w in self.allowed_guesses)
            if self.consistent
            else self.allowed_guesses
        )
        if not candidates:
            raise ContradictionError("No permitted candidate can be selected")
        return {"guess": self.rng.choice(candidates)}

    def snapshot(self) -> dict:
        version, values, gaussian = self.rng.getstate()
        return {
            "version": 1,
            "allowed_guesses": list(self.allowed_guesses),
            "consistent": self.consistent,
            "rng": [version, list(values), gaussian],
        }

    def restore(self, snapshot: dict) -> None:
        if type(snapshot) is not dict or set(snapshot) != {
            "version",
            "allowed_guesses",
            "consistent",
            "rng",
        }:
            raise ValueError("Invalid policy snapshot")
        if (
            snapshot["version"] != 1
            or tuple(snapshot["allowed_guesses"]) != self.allowed_guesses
            or snapshot["consistent"] is not self.consistent
        ):
            raise ValueError("Policy snapshot configuration mismatch")
        version, values, gaussian = snapshot["rng"]
        rng = random.Random()
        rng.setstate((version, tuple(values), gaussian))
        self.rng = rng
