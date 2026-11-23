from .moves import bind_move, describe_move, is_legal


def _pegs(value):
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(type(peg) is int and 0 <= peg <= 2 for peg in value)
    )


def validate_inputs(inputs):
    if not isinstance(inputs, dict) or set(inputs) != {"knowledge", "goal"}:
        raise ValueError("Hanoi inputs require exactly knowledge and goal")
    knowledge, goal = inputs["knowledge"], inputs["goal"]
    if not isinstance(knowledge, dict) or set(knowledge) != {"public"}:
        raise ValueError("knowledge requires only the granted public observation")
    public = knowledge["public"]
    if not isinstance(public, dict) or set(public) != {
        "pegs",
        "goal",
        "steps",
        "terminated",
        "truncated",
    }:
        raise ValueError("public observation fields do not match the Hanoi profile")
    if not _pegs(public["pegs"]):
        raise ValueError("pegs must identify exactly three disk positions")
    if not isinstance(goal, dict) or set(goal) != {"pegs"} or goal["pegs"] != [2, 2, 2]:
        raise ValueError("this scenario's goal is exactly [2, 2, 2]")
    if not _pegs(goal["pegs"]) or not _pegs(public["goal"]) or public["goal"] != goal["pegs"]:
        raise ValueError("public goal must agree with the selected three-disk goal")
    if type(public["steps"]) is not int or not 0 <= public["steps"] <= 100:
        raise ValueError("steps must be an integer from 0 to 100")
    if any(type(public[field]) is not bool for field in ("terminated", "truncated")):
        raise ValueError("terminal flags must be Boolean")
    if public["terminated"] != (public["pegs"] == goal["pegs"]):
        raise ValueError("termination must agree with the observed goal")
    if public["truncated"]:
        raise ValueError("a truncated World cannot accept another planned move")


def validate_completion(outputs, inputs):
    if not isinstance(outputs, dict) or set(outputs) != {"plan"}:
        raise ValueError("completion requires exactly plan")
    plan = outputs["plan"]
    if not isinstance(plan, dict) or set(plan) != {
        "actions",
        "invocation",
        "remaining_moves",
        "expanded_states",
    }:
        raise ValueError("plan fields do not match the Hanoi profile")
    for field, maximum in (("remaining_moves", 7), ("expanded_states", 27)):
        if type(plan[field]) is not int or not 0 <= plan[field] <= maximum:
            raise ValueError(f"invalid {field}")
    public = inputs["knowledge"]["public"]
    if public["pegs"] == inputs["goal"]["pegs"]:
        if plan != {"actions": [], "invocation": None, "remaining_moves": 0, "expanded_states": 0}:
            raise ValueError("observed completion requires an empty plan")
        return
    actions = plan["actions"]
    if not isinstance(actions, list) or len(actions) != 1 or not isinstance(actions[0], dict):
        raise ValueError("an unfinished task requires exactly one next command")
    action = actions[0]
    if set(action) != {"disk", "target"} or any(
        type(action[field]) is not int or not 0 <= action[field] <= 2
        for field in ("disk", "target")
    ):
        raise ValueError("a command requires an integer disk and target from 0 to 2")
    if not is_legal(public["pegs"], action["disk"], action["target"]):
        raise ValueError("the proposed move is illegal in the observed configuration")
    expected = describe_move(bind_move(action["disk"], action["target"]))
    if plan["invocation"] != expected:
        raise ValueError("reusable move bindings must identify the proposed command")
    if plan["remaining_moves"] == 0 or plan["expanded_states"] == 0:
        raise ValueError("an unfinished task requires a nonempty searched plan")
