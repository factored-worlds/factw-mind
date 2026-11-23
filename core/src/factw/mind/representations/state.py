from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class StateEstimate[World, Memory]:
    world: World
    memory: Memory


@dataclass(frozen=True)
class ObservedTransition[Observation, Action, Outcome]:
    action: Action
    outcome: Outcome
    observation: Observation
    elapsed_seconds: float

    def __post_init__(self):
        if (
            type(self.elapsed_seconds) not in (int, float)
            or not isfinite(self.elapsed_seconds)
            or self.elapsed_seconds < 0
        ):
            raise ValueError("Elapsed time must be finite and nonnegative")
