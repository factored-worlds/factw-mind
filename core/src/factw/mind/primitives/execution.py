from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from ..representations import ObservedTransition
from .types import PrimitiveInvocation


class ExecutionStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TRUNCATED = "truncated"


@dataclass(frozen=True)
class Completion:
    status: ExecutionStatus
    reason: str


@dataclass(frozen=True)
class CommandRequest[Command]:
    command: Command


@dataclass(frozen=True)
class ExecutionFeedback[Observation, Command, Outcome]:
    transition: ObservedTransition[Observation, Command, Outcome]
    terminated: bool = False
    truncated: bool = False

    def __post_init__(self):
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise ValueError("Execution boundaries must be Boolean")
        if self.terminated and self.truncated:
            raise ValueError("Execution cannot terminate and truncate together")


type Decision[Command] = CommandRequest[Command] | Completion


class PrimitiveExecution[Observation, Command, Outcome](Protocol):
    def start(self) -> Decision[Command]: ...

    def advance(
        self, feedback: ExecutionFeedback[Observation, Command, Outcome]
    ) -> Decision[Command]: ...


class PrimitiveExecutor[Observation, Command, Outcome](Protocol):
    def start(
        self, invocation: PrimitiveInvocation, observation: Observation
    ) -> PrimitiveExecution[Observation, Command, Outcome]: ...


@dataclass(frozen=True)
class ExecutionResult[Observation, Command, Outcome]:
    completion: Completion
    feedback: tuple[ExecutionFeedback[Observation, Command, Outcome], ...]


def execute_primitive[Observation, Command, Outcome](
    executor: PrimitiveExecutor[Observation, Command, Outcome],
    invocation: PrimitiveInvocation,
    observation: Observation,
    step: Callable[[Command], ExecutionFeedback[Observation, Command, Outcome]],
    *,
    max_commands: int,
) -> ExecutionResult[Observation, Command, Outcome]:
    if type(max_commands) is not int or max_commands < 1:
        raise ValueError("Command budget must be a positive integer")
    execution = executor.start(invocation, observation)
    decision = execution.start()
    feedback = []
    while isinstance(decision, CommandRequest):
        if len(feedback) == max_commands:
            decision = Completion(ExecutionStatus.TRUNCATED, "command_budget_exhausted")
            break
        observed = step(decision.command)
        feedback.append(observed)
        decision = execution.advance(observed)
        if isinstance(decision, CommandRequest):
            if observed.truncated:
                decision = Completion(ExecutionStatus.TRUNCATED, "environment_truncated")
            elif observed.terminated:
                decision = Completion(ExecutionStatus.FAILED, "environment_terminated")
    if not isinstance(decision, Completion):
        raise TypeError("An execution must return a command request or completion")
    return ExecutionResult(decision, tuple(feedback))
