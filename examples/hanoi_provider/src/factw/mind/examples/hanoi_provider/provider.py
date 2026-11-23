from copy import deepcopy

from .compute import compute
from .config import validate_config
from .contracts import validate_completion, validate_inputs

CONTRACT = "symbolic.plan"
INPUT_PORTS = {"knowledge": "knowledge", "goal": "goal"}
OUTPUT_PORTS = {"plan": "plan"}


class Provider:
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
        outputs = compute(values, self.config)
        validate_completion(outputs, values)
        return {
            port: {
                "schema": OUTPUT_PORTS[port],
                "value": value,
                "basis": deepcopy(call["basis"]),
                "sequence": max(record["sequence"] for record in call["inputs"].values()),
            }
            for port, value in outputs.items()
        }

    def cancel(self, request_id):
        return False

    def close(self):
        self._closed = True
