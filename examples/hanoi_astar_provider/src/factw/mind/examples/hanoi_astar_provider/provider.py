from copy import deepcopy

from factw.mind.examples.hanoi_provider.config import validate_config
from factw.mind.examples.hanoi_provider.contracts import validate_completion, validate_inputs

from .compute import compute

CONTRACT = "symbolic.plan"


class Provider:
    def __init__(self, config):
        self.config = validate_config(config)
        self._closed = False

    def invoke(self, call):
        if self._closed:
            raise RuntimeError("provider is closed")
        if call["contract"] != CONTRACT:
            raise ValueError("provider received an unsupported callable contract")
        inputs = {port: record["value"] for port, record in call["inputs"].items()}
        validate_inputs(inputs)
        output = compute(inputs, self.config)
        validate_completion(output, inputs)
        return {
            "plan": {
                "schema": "plan",
                "value": output["plan"],
                "basis": deepcopy(call["basis"]),
                "sequence": max(record["sequence"] for record in call["inputs"].values()),
            }
        }

    def cancel(self, request_id):
        return False

    def close(self):
        self._closed = True
