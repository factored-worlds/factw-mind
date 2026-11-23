import pytest

from factw.mind.primitives import (
    CommandRequest,
    Completion,
    DirectExecutor,
    ExecutionFeedback,
    ExecutionStatus,
    PrimitiveType,
    execute_primitive,
)
from factw.mind.representations import ObservedTransition


def feedback(command, outcome="passed", observation=1, **boundary):
    return ExecutionFeedback(ObservedTransition(command, outcome, observation, 0.25), **boundary)


def classify(observed):
    outcome = observed.transition.outcome
    status = ExecutionStatus.SUCCEEDED if outcome == "passed" else ExecutionStatus.FAILED
    return Completion(status, outcome)


@pytest.mark.parametrize("outcome,status", [("passed", "succeeded"), ("denied", "failed")])
def test_direct_execution_uses_observed_outcome_not_command_intent(outcome, status):
    primitive = PrimitiveType("inspect")
    executor = DirectExecutor(primitive, lambda call, initial: "inspect-command", classify)
    result = execute_primitive(
        executor,
        primitive.bind({}),
        0,
        lambda command: feedback(command, outcome),
        max_commands=3,
    )
    assert result.completion.status == status
    assert len(result.feedback) == 1
    assert result.feedback[0].transition.action == "inspect-command"


class CounterExecution:
    def start(self):
        return CommandRequest("advance")

    def advance(self, observed):
        if observed.transition.observation >= 3:
            return Completion(ExecutionStatus.SUCCEEDED, "target_reached")
        return CommandRequest("advance")


class CounterExecutor:
    def start(self, invocation, observation):
        return CounterExecution()


def test_multistep_execution_waits_for_feedback_and_preserves_command_trace():
    observations = iter([1, 2, 3])
    result = execute_primitive(
        CounterExecutor(),
        PrimitiveType("reach").bind({}),
        0,
        lambda command: feedback(command, observation=next(observations)),
        max_commands=5,
    )
    assert result.completion == Completion(ExecutionStatus.SUCCEEDED, "target_reached")
    assert [item.transition.observation for item in result.feedback] == [1, 2, 3]


@pytest.mark.parametrize(
    "boundary,status,reason",
    [
        ({"terminated": True}, "failed", "environment_terminated"),
        ({"truncated": True}, "truncated", "environment_truncated"),
        ({}, "truncated", "command_budget_exhausted"),
    ],
)
def test_execution_stops_on_boundaries_or_budget_without_claiming_success(boundary, status, reason):
    result = execute_primitive(
        CounterExecutor(),
        PrimitiveType("reach").bind({}),
        0,
        lambda command: feedback(command, **boundary),
        max_commands=1,
    )
    assert result.completion == Completion(ExecutionStatus(status), reason)
    assert len(result.feedback) == 1


def test_environment_exceptions_propagate():
    def failing_step(command):
        raise OSError("connection lost")

    with pytest.raises(OSError, match="connection lost"):
        execute_primitive(
            CounterExecutor(), PrimitiveType("reach").bind({}), 0, failing_step, max_commands=1
        )


def test_direct_execution_lifecycle_and_wrong_primitive_are_errors():
    primitive = PrimitiveType("inspect")
    executor = DirectExecutor(primitive, lambda call, initial: "inspect", classify)
    with pytest.raises(ValueError, match="primitive type"):
        executor.start(PrimitiveType("release").bind({}), 0)
    execution = executor.start(primitive.bind({}), 0)
    with pytest.raises(RuntimeError, match="pending"):
        execution.advance(feedback("inspect"))
    execution.start()
    execution.advance(feedback("inspect"))
    with pytest.raises(RuntimeError, match="pending"):
        execution.advance(feedback("inspect"))
    with pytest.raises(RuntimeError, match="already started"):
        execution.start()


def test_execution_feedback_rejects_conflicting_boundaries():
    with pytest.raises(ValueError, match="terminate and truncate"):
        feedback("read", terminated=True, truncated=True)
