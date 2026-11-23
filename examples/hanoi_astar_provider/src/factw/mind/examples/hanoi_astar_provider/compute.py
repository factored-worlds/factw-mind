"""Heap A-star with an admissible misplaced-disk lower bound."""

from heapq import heappop, heappush

from factw.mind.examples.hanoi_provider.moves import (
    bind_move,
    command_for,
    describe_move,
    is_legal,
)

from factw.mind.planning import SearchLimitError


def distance(state, goal):
    return sum(peg != destination for peg, destination in zip(state, goal, strict=True))


def search(start, goal, *, max_states, max_depth):
    if start == goal:
        return (), 0
    queue = [(distance(start, goal), distance(start, goal), 0, start)]
    costs, parents, expanded = {start: 0}, {start: None}, set()
    while queue:
        _, _, depth, state = heappop(queue)
        if state in expanded or depth != costs[state]:
            continue
        if state == goal:
            actions = []
            while parents[state] is not None:
                state, action = parents[state]
                actions.append(action)
            return tuple(reversed(actions)), len(expanded)
        expanded.add(state)
        if depth == max_depth:
            continue
        for disk in range(3):
            for target in range(3):
                if not is_legal(state, disk, target):
                    continue
                successor = (*state[:disk], target, *state[disk + 1 :])
                cost, lower_bound = depth + 1, distance(successor, goal)
                if cost + lower_bound > max_depth or cost >= costs.get(successor, max_depth + 1):
                    continue
                if successor not in costs and len(costs) == max_states:
                    raise SearchLimitError("A-star exhausted max_states before finding a path")
                costs[successor], parents[successor] = cost, (state, (disk, target))
                heappush(queue, (cost + lower_bound, lower_bound, cost, successor))
    raise SearchLimitError("A-star found no path within max_depth")


def compute(inputs, config):
    start = tuple(inputs["knowledge"]["public"]["pegs"])
    goal = tuple(inputs["goal"]["pegs"])
    actions, expanded = search(start, goal, **config)
    invocation = bind_move(*actions[0]) if actions else None
    return {
        "plan": {
            "actions": [command_for(invocation)] if actions else [],
            "invocation": describe_move(invocation) if actions else None,
            "remaining_moves": len(actions),
            "expanded_states": expanded,
        }
    }
