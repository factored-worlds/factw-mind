from collections.abc import Callable
from dataclasses import dataclass

from .execution import CommandRequest, Completion, ExecutionFeedback
from .types import PrimitiveInvocation, PrimitiveType


@dataclass
class _DirectExecution[Observation, Command, Outcome]:
    command: Command
    classify: Callable[[ExecutionFeedback[Observation, Command, Outcome]], Completion]
    started: bool = False
    finished: bool = False

    def start(self) -> CommandRequest[Command]:
        if self.started:
            raise RuntimeError("Execution already started")
        self.started = True
        return CommandRequest(self.command)

    def advance(self, feedback: ExecutionFeedback[Observation, Command, Outcome]) -> Completion:
        if not self.started or self.finished:
            raise RuntimeError("Execution has no pending command")
        self.finished = True
        return self.classify(feedback)


class DirectExecutor[Observation, Command, Outcome]:
    def __init__(
        self,
        primitive: PrimitiveType,
        command: Callable[[PrimitiveInvocation, Observation], Command],
        classify: Callable[[ExecutionFeedback[Observation, Command, Outcome]], Completion],
    ):
        self.primitive = primitive
        self.command = command
        self.classify = classify

    def start(
        self, invocation: PrimitiveInvocation, observation: Observation
    ) -> _DirectExecution[Observation, Command, Outcome]:
        if invocation.primitive != self.primitive:
            raise ValueError("Executor does not implement this primitive type")
        return _DirectExecution(self.command(invocation, observation), self.classify)
