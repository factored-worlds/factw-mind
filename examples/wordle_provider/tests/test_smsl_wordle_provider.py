"""Exercise the exact-method adapter through the actual State Language SDK."""

import json
import tomllib
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import smsl

from factw.mind.baselines.wordle import (
    ContradictionError,
    ExactBeliefEstimator,
    FeedbackPredictor,
    InformationPolicy,
    RandomPolicy,
)
from factw.mind.baselines.wordle.provider_contract import (
    predicted_partitions,
    profile_bytes,
    profile_schema,
    validate_completion,
)
from factw.mind.examples.wordle_provider import demo, provider, registration, replacement


@pytest.fixture
def adapter():
    return SimpleNamespace(
        root=Path(__file__).resolve().parents[1],
        provider=provider,
        registration=registration,
        demo=demo,
    )


def definition(adapter, configs=None):
    configs = {"next_guess": adapter.demo.DEFAULT_CONFIG} if configs is None else configs
    blocks = "\n".join(
        f'block {name}: symbolic.plan {{ provider "wordle.exact_information_policy"; '
        "config { "
        + " ".join(f"{key} = {json.dumps(value)};" for key, value in config.items())
        + " }; }"
        for name, config in configs.items()
    )
    return smsl.SystemDefinition.loads(
        'smsl "0.2"; system ExactWordle { component planner: "symbolic-planner" { '
        + blocks
        + " } }"
    )


def registry(adapter, factory=None):
    return smsl.ProviderRegistry().register(
        "wordle.exact_information_policy",
        {"symbolic.plan"},
        adapter.registration.bind_services(MappingProxyType({})) if factory is None else factory,
    )


def records(values, basis, sequence=1):
    return {
        port: {
            "schema": port,
            "value": deepcopy(value),
            "basis": deepcopy(basis),
            "sequence": sequence,
        }
        for port, value in values.items()
    }


def invoke(adapter, values=None, *, config=None, factory=None):
    values = deepcopy(adapter.demo.EXAMPLE_INPUTS if values is None else values)
    config = adapter.demo.DEFAULT_CONFIG if config is None else config
    basis = {"revision": "wordle_adapter_test", "tick": 0}
    with definition(adapter, {"next_guess": config}).session(
        registry(adapter, factory), basis
    ) as session:
        result = session.invoke(
            "planner.next_guess",
            "guess",
            inputs=records(values, basis),
            current_basis=lambda: basis,
            validate_outputs=lambda outputs, call: adapter.demo.validate_result(
                outputs, values, config
            ),
        )
    return {port: record["value"] for port, record in result.items()}


def narrowed(adapter):
    values = deepcopy(adapter.demo.EXAMPLE_INPUTS)
    values["knowledge"]["public"].update(
        guesses=["ALLEY"],
        feedbacks=[[2, 1, 0, 1, 0]],
        attempts=1,
    )
    return values


def test_real_sdk_calls_canonical_methods_without_mutating_inputs(adapter, monkeypatch):
    calls = []
    for owner, name in [
        (ExactBeliefEstimator, "initialize"),
        (InformationPolicy, "choose"),
        (FeedbackPredictor, "predict"),
    ]:
        original = getattr(owner, name)

        def tracked(self, *args, _name=name, _original=original):
            calls.append(_name)
            return _original(self, *args)

        monkeypatch.setattr(owner, name, tracked)
    values = narrowed(adapter)
    before = deepcopy(values)
    assert invoke(adapter, values)["plan"] == {
        "actions": [{"guess": "APPLE"}],
        "candidates": ["APPLE"],
        "predictions": [{"feedback": [2] * 5, "candidates": ["APPLE"], "count": 1, "total": 1}],
    }
    assert values == before
    assert {"initialize", "choose", "predict"} <= set(calls)


def test_full_history_replay_has_no_accumulated_private_belief(adapter):
    basis = {"revision": "public_history", "tick": 0}
    values = [adapter.demo.EXAMPLE_INPUTS, narrowed(adapter), adapter.demo.EXAMPLE_INPUTS]
    outputs = []
    with definition(adapter).session(registry(adapter), basis) as session:
        for tick, value in enumerate(values):
            current = {**basis, "tick": tick}
            if tick:
                session.advance(current)
            result = session.invoke(
                "planner.next_guess",
                f"request_{tick}",
                inputs=records(value, current, tick + 1),
                current_basis=lambda current=current: current,
                validate_outputs=lambda output, call, value=value: adapter.demo.validate_result(
                    output, value
                ),
            )
            assert result["plan"]["basis"] == current
            assert result["plan"]["sequence"] == tick + 1
            outputs.append(result["plan"]["value"])
    assert outputs[0] == outputs[2] == invoke(adapter)["plan"]
    assert outputs[1]["candidates"] == ["APPLE"]
    assert len(outputs[0]["candidates"]) == 5


def test_same_registration_owns_distinct_blocks_and_sessions(adapter):
    created = []

    def create(config):
        instance = adapter.provider.Provider(config)
        created.append(instance)
        return instance

    narrow = {"answers": ["APPLE"], "allowed_guesses": ["APPLE"]}
    model = definition(
        adapter, {"next_guess": adapter.demo.DEFAULT_CONFIG, "single_answer": narrow}
    )
    basis = {"revision": "per_instance", "tick": 0}
    for _ in range(2):
        with model.session(registry(adapter, create), basis) as session:
            for block, expected in [("next_guess", 5), ("single_answer", 1)]:
                result = session.invoke(
                    "planner." + block,
                    block,
                    inputs=records(adapter.demo.EXAMPLE_INPUTS, basis),
                )
                assert len(result["plan"]["value"]["candidates"]) == expected
    assert len({id(item) for item in created}) == 4
    assert all(item._closed for item in created)
    assert created[0].config is not created[2].config


@pytest.mark.parametrize(
    "mutation",
    [
        "target",
        "unknown",
        "goal_false",
        "goal_integer",
        "feedback_boolean",
        "progress",
        "extra_knowledge",
    ],
)
def test_invalid_player_values_are_rejected_through_sdk(adapter, mutation):
    values = narrowed(adapter)
    public = values["knowledge"]["public"]
    if mutation == "target":
        public["target"] = "APPLE"
    elif mutation == "unknown":
        values["knowledge"]["public"] = None
    elif mutation == "goal_false":
        values["goal"]["won"] = False
    elif mutation == "goal_integer":
        values["goal"]["won"] = 1
    elif mutation == "feedback_boolean":
        public["feedbacks"][0][0] = True
    elif mutation == "progress":
        public["won"] = True
    else:
        values["knowledge"]["reference"] = {"answer": "APPLE"}
    with pytest.raises(ValueError):
        invoke(adapter, values)


@pytest.mark.parametrize(
    "config",
    [
        {"answers": [], "allowed_guesses": ["APPLE"]},
        {"answers": ["APPLE", "APPLE"], "allowed_guesses": ["APPLE"]},
        {"answers": ["APPLE"], "allowed_guesses": ["ALLEY"]},
        {"answers": ["APPLE"], "allowed_guesses": ["APPLE"], "seed": 1},
    ],
)
def test_actual_block_configuration_is_validated(adapter, config):
    with pytest.raises(ValueError):
        invoke(adapter, config=config)


def test_accepted_public_history_must_use_the_declared_action_vocabulary(adapter):
    values = narrowed(adapter)
    config = {"answers": ["APPLE"], "allowed_guesses": ["APPLE"]}
    with pytest.raises(ValueError, match="accepted history.*action vocabulary"):
        invoke(adapter, values, config=config)


def test_legal_repeated_guesses_and_rejected_attempts_remain_valid_history(adapter):
    values = narrowed(adapter)
    public = values["knowledge"]["public"]
    public.update(
        guesses=["ALLEY", "ALLEY"],
        feedbacks=[[2, 1, 0, 1, 0], [2, 1, 0, 1, 0]],
        attempts=3,
    )
    config = {"answers": ["APPLE"], "allowed_guesses": ["APPLE", "ALLEY"]}
    assert invoke(adapter, values, config=config)["plan"]["actions"] == [{"guess": "APPLE"}]


def test_contradiction_failure_does_not_poison_a_later_public_request(adapter):
    basis = {"revision": "failure", "tick": 0}
    values = narrowed(adapter)
    values["knowledge"]["public"].update(guesses=["APPLE"], feedbacks=[[0] * 5])
    config = {"answers": ["APPLE"], "allowed_guesses": ["APPLE"]}
    with definition(adapter, {"next_guess": config}).session(registry(adapter), basis) as session:
        with pytest.raises(ContradictionError):
            session.invoke("planner.next_guess", "bad", inputs=records(values, basis))
        assert session.inspect()["pending"] == []
        result = session.invoke(
            "planner.next_guess",
            "good",
            inputs=records(adapter.demo.EXAMPLE_INPUTS, basis, 2),
        )
        assert result["plan"]["value"]["actions"] == [{"guess": "APPLE"}]


@pytest.mark.parametrize("status", ["won", "lost", "truncated"])
def test_observed_terminal_state_controls_goal_behavior(adapter, status):
    values = deepcopy(adapter.demo.EXAMPLE_INPUTS)
    public = values["knowledge"]["public"]
    if status == "won":
        public.update(guesses=["APPLE"], feedbacks=[[2] * 5], attempts=1, won=True, terminated=True)
        assert invoke(adapter, values)["plan"] == {
            "actions": [],
            "candidates": ["APPLE"],
            "predictions": [],
        }
    else:
        if status == "lost":
            public.update(
                guesses=["ALLEY"],
                feedbacks=[[2, 1, 0, 1, 0]],
                attempts=1,
                max_guesses=1,
                terminated=True,
            )
        else:
            public.update(attempts=60, truncated=True)
        with pytest.raises(ValueError, match="ended"):
            invoke(adapter, values)


@pytest.mark.parametrize("mutation", ["missing", "nominal", "basis"])
def test_sdk_envelope_rejects_before_creating_provider(adapter, mutation):
    created = []

    def create(config):
        created.append(config)
        return adapter.provider.Provider(config)

    basis = {"revision": "envelope", "tick": 0}
    supplied = records(adapter.demo.EXAMPLE_INPUTS, basis)
    if mutation == "missing":
        del supplied["goal"]
    elif mutation == "nominal":
        supplied["knowledge"]["schema"] = "observation"
    else:
        supplied["knowledge"]["basis"]["revision"] = "another_model"
    with definition(adapter).session(registry(adapter, create), basis) as session:
        with pytest.raises(smsl.SmslError):
            session.invoke("planner.next_guess", "invalid", inputs=supplied)
    assert created == []


def test_host_completion_rejects_tampered_exact_partitions(adapter):
    class Tampered(adapter.provider.Provider):
        def invoke(self, call):
            output = super().invoke(call)
            output["plan"]["value"]["predictions"][0]["count"] += 1
            return output

    with pytest.raises(ValueError, match="partitions"):
        invoke(adapter, factory=Tampered)


def test_contract_allows_a_different_legal_policy_with_exact_predictions(adapter):
    values = deepcopy(adapter.demo.EXAMPLE_INPUTS)
    belief = ExactBeliefEstimator(tuple(adapter.demo.WORDS)).initialize(
        values["knowledge"]["public"]
    )
    action = {"guess": "ALLEY"}
    result = {
        "plan": {
            "actions": [action],
            "candidates": list(belief.candidates),
            "predictions": predicted_partitions(belief, action),
        }
    }
    validate_completion(result, values, adapter.demo.DEFAULT_CONFIG)


def test_alternate_registration_invokes_the_existing_consistent_policy(adapter, monkeypatch):
    calls = []
    choose = RandomPolicy.choose

    def tracked(self, belief):
        calls.append(self.consistent)
        return choose(self, belief)

    monkeypatch.setattr(RandomPolicy, "choose", tracked)
    factory = replacement.bind_services(MappingProxyType({}))
    result = invoke(adapter, factory=factory)
    assert result == invoke(adapter, factory=factory)
    assert result["plan"]["actions"] != invoke(adapter)["plan"]["actions"]
    assert calls == [True, True]


def test_alternate_metadata_changes_implementation_without_changing_profiles(adapter):
    primary = tomllib.loads((adapter.root / "extension.toml").read_text())
    alternate = tomllib.loads((adapter.root / "replacement_extension.toml").read_text())
    assert alternate["id"] == "wordle.consistent_policy"
    assert alternate["resources"] == primary["resources"]
    for key in ("contracts", "config_schema", "information", "lifecycle"):
        assert alternate["provider"][key] == primary["provider"][key]
    assert alternate["provider"]["registration"] != primary["provider"]["registration"]
    assert alternate["provider"]["implementation"] != primary["provider"]["implementation"]


def test_closed_provider_rejects_calls_and_binding_has_no_services(adapter):
    factory = adapter.registration.bind_services(MappingProxyType({}))
    instance = factory(adapter.demo.DEFAULT_CONFIG)
    basis = {"revision": "closed", "tick": 0}
    call = {
        "contract": "symbolic.plan",
        "basis": basis,
        "inputs": records(adapter.demo.EXAMPLE_INPUTS, basis),
    }
    assert instance.invoke(call)["plan"]["value"]["actions"]
    assert instance.cancel("not_pending") is False
    instance.close()
    instance.close()
    with pytest.raises(RuntimeError, match="closed"):
        instance.invoke(call)
    for operations in ([], None, {"execute": object()}):
        with pytest.raises(ValueError):
            adapter.registration.bind_services(operations)


def test_payload_resources_have_one_method_owner(adapter):
    import hashlib

    descriptor = tomllib.loads((adapter.root / "extension.toml").read_text())
    resources = descriptor["resources"]
    assert descriptor["provider"]["information"]["requires_fields"] == [
        {"view": "player_wordle", "pointer": "/public"}
    ]
    for identity, name in [
        ("configuration", "configuration"),
        ("inputs_knowledge", "knowledge"),
        ("inputs_goal", "goal"),
        ("outputs_plan", "plan"),
        ("wordle_public_history", "public_history"),
    ]:
        resource = resources[identity]
        assert resource["distribution"] == "factw-mind"
        assert resource["package"] == "factw.mind"
        assert resource["sha256"] == hashlib.sha256(profile_bytes(name)).hexdigest()
    assert profile_schema("knowledge")["properties"]["public"] == {"$ref": "wordle_public_history"}
    assert profile_schema("public_history")["$id"] == "wordle_public_history"
    assert not (adapter.root / "src/factw/mind/examples/wordle_provider/contracts").exists()
