from copy import deepcopy
from dataclasses import asdict, dataclass

from factw.mind.representations import ObservedTransition

from .belief import ExactBeliefEstimator, public_history
from .policy import FeedbackPredictor


@dataclass(frozen=True)
class TaskProgress:
    attempts: int
    accepted_guesses: int
    won: bool
    terminated: bool
    truncated: bool

    @classmethod
    def observed(cls, public: dict):
        public_history(public)
        return cls(
            public["attempts"],
            len(public["guesses"]),
            public["won"],
            public["terminated"],
            public["truncated"],
        )


class WordleSession:
    """Estimate, predict, select, execute and check one completed public transition."""

    def __init__(self, estimator: ExactBeliefEstimator, policy, initial: dict):
        self.estimator, self.policy = estimator, policy
        self.belief = estimator.initialize(initial)
        self.progress = TaskProgress.observed(initial)
        self.public = deepcopy(initial)
        self.predictor = FeedbackPredictor()

    def step(self, execute) -> dict:
        if self.progress.terminated or self.progress.truncated:
            raise RuntimeError("The observed task has ended")
        before = self.belief
        action = self.policy.choose(before)
        predictions = self.predictor.predict(before, action)
        transition: ObservedTransition = execute(deepcopy(action))
        if transition.action != action:
            raise ValueError("Executor returned a different bound action")
        public = transition.observation
        after = self.estimator.update(before, transition)
        progress = TaskProgress.observed(public)
        if progress.attempts != self.progress.attempts + 1:
            raise ValueError("Executor did not complete exactly one attempt")
        if any(public[name] != self.public[name] for name in ("max_attempts", "max_guesses")):
            raise ValueError("Executor changed task limits")
        if transition.outcome != "illegal":
            observed = tuple(public["feedbacks"][-1])
            matching = [p for p in predictions if p.feedback == observed]
            if len(matching) != 1 or matching[0].candidates != after.candidates:
                raise ValueError("Observed postcondition disagrees with the candidate prediction")
        self.belief, self.progress, self.public = after, progress, deepcopy(public)
        return {
            "action": action,
            "predictions": [asdict(p) for p in predictions],
            "outcome": transition.outcome,
            "belief_before": list(before.candidates),
            "belief_after": list(after.candidates),
            "progress": asdict(progress),
            "postcondition_checked": transition.outcome != "illegal",
        }
