"""Run one next-guess plan through the actual SDK without a world or harness."""

from copy import deepcopy
from importlib.resources import files

import smsl

from .contracts import validate_completion
from .provider import CONTRACT, INPUT_PORTS
from .registration import bind_services

PROVIDER_ID = "wordle.exact_information_policy"
WORDS = ["APPLE", "ALLEY", "CRANE", "GRAPE", "LEMON"]
DEFAULT_CONFIG = {"answers": WORDS, "allowed_guesses": WORDS}
EXAMPLE_INPUTS = {
    "knowledge": {
        "public": {
            "guesses": [],
            "feedbacks": [],
            "attempts": 0,
            "max_guesses": 6,
            "max_attempts": 60,
            "won": False,
            "terminated": False,
            "truncated": False,
        }
    },
    "goal": {"won": True},
}


def validate_result(outputs, inputs, config=DEFAULT_CONFIG):
    validate_completion({port: record["value"] for port, record in outputs.items()}, inputs, config)


def run_example(public=None):
    definition = smsl.SystemDefinition.loads(
        files("factw.mind.examples.wordle_provider")
        .joinpath("smsl", "provider_system.smsl")
        .read_text()
    )
    registry = smsl.ProviderRegistry().register(PROVIDER_ID, {CONTRACT}, bind_services({}))
    values = deepcopy(EXAMPLE_INPUTS)
    if public is not None:
        values["knowledge"]["public"] = deepcopy(public)
    basis = {
        "revision": "wordle_provider_example",
        "tick": values["knowledge"]["public"]["attempts"],
    }
    records = {
        port: {"schema": INPUT_PORTS[port], "value": value, "basis": deepcopy(basis), "sequence": 1}
        for port, value in values.items()
    }
    with definition.session(registry, basis) as session:
        result = session.invoke(
            "planner.next_guess",
            "next_guess",
            inputs=records,
            current_basis=lambda: basis,
            validate_outputs=lambda outputs, invocation: validate_result(outputs, values),
        )
    return {port: record["value"] for port, record in result.items()}
