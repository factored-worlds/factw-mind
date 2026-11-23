from copy import deepcopy
from importlib.resources import files

import smsl
from factw.mind.examples.hanoi_provider.contracts import validate_completion
from factw.mind.examples.hanoi_provider.demo import EXAMPLE_INPUTS

from .registration import bind_services

PROVIDER_ID = "hanoi_astar.planner"


def run_example(public=None):
    definition = smsl.SystemDefinition.loads(
        files("factw.mind.examples.hanoi_astar_provider")
        .joinpath("smsl/provider_system.smsl")
        .read_text()
    )
    registry = smsl.ProviderRegistry().register(PROVIDER_ID, {"symbolic.plan"}, bind_services({}))
    inputs = deepcopy(EXAMPLE_INPUTS)
    if public is not None:
        inputs["knowledge"]["public"] = deepcopy(public)
    basis = {"revision": "hanoi_astar_example", "tick": inputs["knowledge"]["public"]["steps"]}
    records = {
        port: {"schema": port, "value": value, "basis": deepcopy(basis), "sequence": 1}
        for port, value in inputs.items()
    }
    with definition.session(registry, basis) as session:
        result = session.invoke(
            "planner.next_move",
            "next_move",
            inputs=records,
            current_basis=lambda: basis,
            validate_outputs=lambda output, call: validate_completion(
                {port: record["value"] for port, record in output.items()}, inputs
            ),
        )
    return {port: record["value"] for port, record in result.items()}
