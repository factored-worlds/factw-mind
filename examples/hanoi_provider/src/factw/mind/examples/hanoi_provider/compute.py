from factw.mind.planning import SearchLimitError, breadth_first_search

from .moves import bind_move, command_for, describe_move, is_legal

__all__ = ["SearchLimitError", "compute"]


def successors(state):
    for disk in range(3):
        for target in range(3):
            if is_legal(state, disk, target):
                yield (disk, target), (*state[:disk], target, *state[disk + 1 :])


def compute(inputs, config):
    start = tuple(inputs["knowledge"]["public"]["pegs"])
    goal = tuple(inputs["goal"]["pegs"])
    result = breadth_first_search(start, goal, successors, **config)
    if not result.actions:
        return {
            "plan": {"actions": [], "invocation": None, "remaining_moves": 0, "expanded_states": 0}
        }
    invocation = bind_move(*result.actions[0])
    return {
        "plan": {
            "actions": [command_for(invocation)],
            "invocation": describe_move(invocation),
            "remaining_moves": len(result.actions),
            "expanded_states": result.expanded_states,
        }
    }
