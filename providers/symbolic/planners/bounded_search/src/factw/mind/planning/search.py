"""Breadth-first search; the caller supplies states, actions and transitions."""

from collections import deque
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass


class SearchLimitError(RuntimeError):
    """No path was found within the supplied state and depth bounds."""


@dataclass(frozen=True)
class SearchResult[Action]:
    actions: tuple[Action, ...]
    expanded_states: int


def breadth_first_search[State: Hashable, Action](
    start: State,
    goal: State,
    successors: Callable[[State], Iterable[tuple[Action, State]]],
    *,
    max_depth: int,
    max_states: int,
) -> SearchResult[Action]:
    """Return a shortest path, preserving the caller's successor order for ties."""
    if type(max_depth) is not int or max_depth < 0:
        raise ValueError("max_depth must be a nonnegative integer")
    if type(max_states) is not int or max_states < 1:
        raise ValueError("max_states must be a positive integer")
    if start == goal:
        return SearchResult((), 0)
    queue = deque([(start, 0)])
    parents = {start: None}
    expanded = 0
    while queue:
        state, depth = queue.popleft()
        expanded += 1
        if depth == max_depth:
            continue
        for action, successor in successors(state):
            if successor in parents:
                continue
            if len(parents) == max_states:
                raise SearchLimitError("Search exhausted max_states before finding a path")
            parents[successor] = (state, action)
            if successor == goal:
                actions = []
                while parents[successor] is not None:
                    successor, chosen = parents[successor]
                    actions.append(chosen)
                return SearchResult(tuple(reversed(actions)), expanded)
            queue.append((successor, depth + 1))
    raise SearchLimitError("No path found within max_depth")
