"""Public-history planning profiles; State Language owns record envelopes."""

import json
from importlib.resources import files

from .belief import ExactBeliefEstimator, public_history
from .policy import FeedbackPredictor
from .rules import vocabulary, word

PROFILE_NAMES = frozenset({"public_history", "configuration", "knowledge", "goal", "plan"})


def profile_bytes(name):
    if name not in PROFILE_NAMES:
        raise ValueError("unknown Wordle provider profile")
    return (
        files("factw.mind")
        .joinpath("baselines", "wordle", "schemas", name + ".schema.json")
        .read_bytes()
    )


def profile_schema(name):
    return json.loads(profile_bytes(name))


def object_fields(value, names, label):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError(f"{label} requires exactly {names}")


def words(value, label):
    if type(value) is not list:
        raise ValueError(f"{label} must be an array")
    return vocabulary(value)


def validate_config(config):
    object_fields(config, {"answers", "allowed_guesses"}, "configuration")
    answers = words(config["answers"], "answers")
    allowed = words(config["allowed_guesses"], "allowed_guesses")
    if not set(answers) <= set(allowed):
        raise ValueError("every declared answer must be an allowed guess")
    return {"answers": list(answers), "allowed_guesses": list(allowed)}


def validate_inputs(values):
    object_fields(values, {"knowledge", "goal"}, "inputs")
    object_fields(values["knowledge"], {"public"}, "knowledge")
    public_history(values["knowledge"]["public"])
    object_fields(values["goal"], {"won"}, "goal")
    if values["goal"]["won"] is not True:
        raise ValueError("the fixed Wordle goal must be won=true")


def predicted_partitions(state, action):
    return [
        {
            "feedback": list(item.feedback),
            "candidates": list(item.candidates),
            "count": len(item.candidates),
            "total": len(state.candidates),
        }
        for item in FeedbackPredictor().predict(state, action)
    ]


def validate_outputs(values):
    object_fields(values, {"plan"}, "outputs")
    plan = values["plan"]
    object_fields(plan, {"actions", "candidates", "predictions"}, "plan")
    words(plan["candidates"], "candidates")
    if type(plan["actions"]) is not list or len(plan["actions"]) > 1:
        raise ValueError("plan must contain at most one next guess")
    for action in plan["actions"]:
        object_fields(action, {"guess"}, "guess action")
        word(action["guess"])
    if type(plan["predictions"]) is not list:
        raise ValueError("predictions must be an array")
    for prediction in plan["predictions"]:
        object_fields(prediction, {"feedback", "candidates", "count", "total"}, "prediction")
        codes = prediction["feedback"]
        if (
            type(codes) is not list
            or len(codes) != 5
            or any(type(code) is not int or code not in (0, 1, 2) for code in codes)
        ):
            raise ValueError("feedback must contain five integer codes in 0..2")
        words(prediction["candidates"], "predicted candidates")
        for name in ("count", "total"):
            if type(prediction[name]) is not int or prediction[name] < 1:
                raise ValueError(f"{name} must be a positive integer")


def validate_completion(values, inputs, config):
    validate_inputs(inputs)
    validate_outputs(values)
    config = validate_config(config)
    public = inputs["knowledge"]["public"]
    if not set(public["guesses"]) <= set(config["allowed_guesses"]):
        raise ValueError("accepted history contains a guess outside the declared action vocabulary")
    belief = ExactBeliefEstimator(tuple(config["answers"])).initialize(public)
    plan = values["plan"]
    if plan["candidates"] != list(belief.candidates):
        raise ValueError("plan candidates disagree with the supplied public evidence")
    if public["won"]:
        if plan["actions"] or plan["predictions"]:
            raise ValueError("a won task requires no further action or prediction")
        return
    if public["terminated"] or public["truncated"]:
        raise ValueError("the observed task ended without satisfying the goal")
    if len(plan["actions"]) != 1:
        raise ValueError("an unfinished task requires one next guess")
    action = plan["actions"][0]
    if action["guess"] not in config["allowed_guesses"]:
        raise ValueError("planned guess is outside the declared action vocabulary")
    if plan["predictions"] != predicted_partitions(belief, action):
        raise ValueError("predicted partitions disagree with the exact feedback model")
