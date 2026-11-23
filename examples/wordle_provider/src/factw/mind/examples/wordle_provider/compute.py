"""Invoke the canonical exact Wordle methods using supplied public history."""

from factw.mind.baselines.wordle import ExactBeliefEstimator, InformationPolicy
from factw.mind.baselines.wordle.provider_contract import predicted_partitions


def compute(values, config, *, policy_factory=InformationPolicy):
    public = values["knowledge"]["public"]
    belief = ExactBeliefEstimator(tuple(config["answers"])).initialize(public)
    if public["won"]:
        return {"plan": {"actions": [], "candidates": list(belief.candidates), "predictions": []}}
    if public["terminated"] or public["truncated"]:
        raise ValueError("the observed task ended without satisfying the goal")
    action = policy_factory(tuple(config["allowed_guesses"])).choose(belief)
    return {
        "plan": {
            "actions": [action],
            "candidates": list(belief.candidates),
            "predictions": predicted_partitions(belief, action),
        }
    }
