"""Adapt one deterministic algorithm to the State Language provider protocol."""

from copy import deepcopy

from factw.mind.baselines.wordle import InformationPolicy

from .compute import compute
from .config import validate_config
from .contracts import validate_completion, validate_inputs

CONTRACT = "symbolic.plan"
INPUT_PORTS = {"knowledge": "knowledge", "goal": "goal"}
OUTPUT_PORTS = {"plan": "plan"}


class Provider:
    policy_factory = staticmethod(InformationPolicy)

    def __init__(self, config):
        self.config = validate_config(config)
        self._closed = False

    def invoke(self, call):
        if self._closed:
            raise RuntimeError("provider is closed")
        if call["contract"] != CONTRACT:
            raise ValueError("provider received an unsupported callable contract")
        values = {port: record["value"] for port, record in call["inputs"].items()}
        validate_inputs(values)
        outputs = compute(values, self.config, policy_factory=self.policy_factory)
        validate_completion(outputs, values, self.config)
        sequence = max(record["sequence"] for record in call["inputs"].values())
        return {
            port: {
                "schema": OUTPUT_PORTS[port],
                "value": value,
                "basis": deepcopy(call["basis"]),
                "sequence": sequence,
            }
            for port, value in outputs.items()
        }

    def cancel(self, request_id):
        return False

    def close(self):
        self._closed = True
