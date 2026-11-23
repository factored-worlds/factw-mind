from copy import deepcopy
from importlib.resources import files

import smsl

from .contracts import validate_completion
from .provider import CONTRACT, INPUT_PORTS
from .registration import bind_services

PROVIDER_ID = "hanoi_tutorial.planner"
DEFAULT_CONFIG = {"max_states": 27, "max_depth": 7}
EXAMPLE_INPUTS = {
    "knowledge": {
        "public": {
            "pegs": [0, 0, 0],
            "goal": [2, 2, 2],
            "steps": 0,
            "terminated": False,
            "truncated": False,
        }
    },
    "goal": {"pegs": [2, 2, 2]},
}


def run_example(public=None):
    definition = smsl.SystemDefinition.loads(
        files("factw.mind.examples.hanoi_provider")
        .joinpath("smsl/provider_system.smsl")
        .read_text()
    )
    registry = smsl.ProviderRegistry().register(PROVIDER_ID, {CONTRACT}, bind_services({}))
    values = deepcopy(EXAMPLE_INPUTS)
    if public is not None:
        values["knowledge"]["public"] = deepcopy(public)
    basis = {"revision": "hanoi_provider_example", "tick": values["knowledge"]["public"]["steps"]}
    records = {
        port: {"schema": INPUT_PORTS[port], "value": value, "basis": deepcopy(basis), "sequence": 1}
        for port, value in values.items()
    }
    with definition.session(registry, basis) as session:
        outputs = session.invoke(
            "planner.next_move",
            "next_move",
            inputs=records,
            current_basis=lambda: basis,
            validate_outputs=lambda result, invocation: validate_completion(
                {port: record["value"] for port, record in result.items()}, values
            ),
        )
    return {port: record["value"] for port, record in outputs.items()}
