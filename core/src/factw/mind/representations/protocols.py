from collections.abc import Iterable
from typing import Protocol

from .state import ObservedTransition


class StateEstimator[Observation, Action, Outcome, State](Protocol):
    def initialize(self, observation: Observation) -> State: ...

    def update(
        self, state: State, transition: ObservedTransition[Observation, Action, Outcome]
    ) -> State: ...


class Predictor[State, Action, Prediction](Protocol):
    def predict(self, state: State, action: Action) -> Prediction: ...


def estimate_state[Observation, Action, Outcome, State](
    estimator: StateEstimator[Observation, Action, Outcome, State],
    initial: Observation,
    transitions: Iterable[ObservedTransition[Observation, Action, Outcome]],
) -> State:
    state = estimator.initialize(initial)
    for transition in transitions:
        state = estimator.update(state, transition)
    return state
