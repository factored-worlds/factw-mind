import pytest

from factw.mind.planning import SearchLimitError, breadth_first_search


def test_shortest_path_and_caller_order_with_cycles():
    graph = {
        "start": [("left", "a"), ("right", "b")],
        "a": [("back", "start"), ("finish_left", "goal")],
        "b": [("finish_right", "goal")],
    }
    result = breadth_first_search(
        "start", "goal", lambda state: graph[state], max_depth=2, max_states=4
    )
    assert result.actions == ("left", "finish_left")
    assert result.expanded_states == 2


def test_satisfied_goal_never_requests_successors():
    def forbidden(state):
        raise AssertionError("A satisfied goal needs no successor computation")

    result = breadth_first_search((1, 2), (1, 2), forbidden, max_depth=0, max_states=1)
    assert result.actions == ()
    assert result.expanded_states == 0


@pytest.mark.parametrize(
    "limits", [{"max_depth": 1, "max_states": 3}, {"max_depth": 2, "max_states": 2}]
)
def test_state_and_depth_limits_are_explicit(limits):
    with pytest.raises(SearchLimitError):
        breadth_first_search(0, 2, lambda state: [("advance", state + 1)], **limits)


def test_unreachable_and_successor_failure_are_not_fallback_answers():
    with pytest.raises(SearchLimitError):
        breadth_first_search(0, 1, lambda state: [], max_depth=3, max_states=3)

    def failed(state):
        raise RuntimeError("transition failed")

    with pytest.raises(RuntimeError, match="transition failed"):
        breadth_first_search(0, 1, failed, max_depth=3, max_states=3)


@pytest.mark.parametrize("depth,states", [(-1, 1), (True, 1), (1, 0), (1, True)])
def test_invalid_bounds(depth, states):
    with pytest.raises(ValueError):
        breadth_first_search(0, 0, lambda state: [], max_depth=depth, max_states=states)
