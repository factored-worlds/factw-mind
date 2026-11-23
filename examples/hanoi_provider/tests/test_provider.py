import unittest
from copy import deepcopy
from itertools import product

from factw.mind.examples.hanoi_provider import SearchLimitError
from factw.mind.examples.hanoi_provider.config import validate_config
from factw.mind.examples.hanoi_provider.contracts import validate_completion
from factw.mind.examples.hanoi_provider.demo import DEFAULT_CONFIG, EXAMPLE_INPUTS, run_example
from factw.mind.examples.hanoi_provider.provider import CONTRACT, Provider
from factw.mind.examples.hanoi_provider.registration import bind_services


def call_for(public):
    basis = {"revision": "hanoi_test", "tick": public["steps"]}
    return {
        "contract": CONTRACT,
        "basis": basis,
        "inputs": {
            "knowledge": {
                "schema": "knowledge",
                "value": {"public": deepcopy(public)},
                "basis": deepcopy(basis),
                "sequence": public["steps"] + 1,
            },
            "goal": {
                "schema": "goal",
                "value": {"pegs": [2, 2, 2]},
                "basis": deepcopy(basis),
                "sequence": 0,
            },
        },
    }


class HanoiProviderTests(unittest.TestCase):
    def test_sdk_invocation_proposes_bound_move_without_changing_observation(self):
        public = deepcopy(EXAMPLE_INPUTS["knowledge"]["public"])
        before = deepcopy(public)
        plan = run_example(public)["plan"]
        self.assertEqual(plan["actions"], [{"disk": 0, "target": 2}])
        self.assertEqual(plan["remaining_moves"], 7)
        self.assertEqual(plan["invocation"]["primitive"], "move")
        self.assertEqual(
            plan["invocation"]["bindings"],
            {
                "disk": {"kind": "disk", "key": "disk_0"},
                "destination": {"kind": "peg", "key": "peg_2"},
            },
        )
        self.assertEqual(public, before)

    def test_observed_positions_drive_each_next_move(self):
        provider = Provider(DEFAULT_CONFIG)
        public = deepcopy(EXAMPLE_INPUTS["knowledge"]["public"])
        expected = [(0, 2), (1, 1), (0, 1), (2, 2), (0, 0), (1, 2), (0, 2)]
        for index, (disk, target) in enumerate(expected):
            call = call_for(public)
            before = deepcopy(call)
            record = provider.invoke(call)["plan"]
            plan = record["value"]
            self.assertEqual(plan["actions"], [{"disk": disk, "target": target}])
            self.assertEqual(plan["remaining_moves"], 7 - index)
            self.assertEqual(record["basis"], call["basis"])
            self.assertEqual(record["sequence"], index + 1)
            self.assertEqual(call, before)
            public["pegs"][disk] = target
            public["steps"] += 1
        public["terminated"] = True
        self.assertEqual(
            provider.invoke(call_for(public))["plan"]["value"],
            {"actions": [], "invocation": None, "remaining_moves": 0, "expanded_states": 0},
        )
        provider.close()

    def test_search_limits_are_explicit_failures(self):
        for config in ({"max_states": 1, "max_depth": 7}, {"max_states": 27, "max_depth": 6}):
            with self.subTest(config=config):
                provider = Provider(config)
                call = call_for(EXAMPLE_INPUTS["knowledge"]["public"])
                before = deepcopy(call)
                with self.assertRaises(SearchLimitError):
                    provider.invoke(call)
                self.assertEqual(call, before)
                provider.close()

    def test_every_three_disk_board_reaches_the_goal_using_legal_next_moves(self):
        provider = Provider(DEFAULT_CONFIG)
        for board in product(range(3), repeat=3):
            with self.subTest(board=board):
                public = deepcopy(EXAMPLE_INPUTS["knowledge"]["public"])
                public["pegs"] = list(board)
                previous_remaining = None
                while public["pegs"] != [2, 2, 2]:
                    plan = provider.invoke(call_for(public))["plan"]["value"]
                    if previous_remaining is not None:
                        self.assertEqual(plan["remaining_moves"], previous_remaining - 1)
                    previous_remaining = plan["remaining_moves"]
                    disk, target = (plan["actions"][0][key] for key in ("disk", "target"))
                    source = public["pegs"][disk]
                    self.assertNotEqual(source, target)
                    for smaller_peg in public["pegs"][:disk]:
                        self.assertNotIn(smaller_peg, (source, target))
                    public["pegs"][disk] = target
                    public["steps"] += 1
                    self.assertLessEqual(public["steps"], 7)
                public["terminated"] = True
                self.assertEqual(provider.invoke(call_for(public))["plan"]["value"]["actions"], [])
        provider.close()

    def test_ungranted_or_inconsistent_information_fails(self):
        for changed in (
            {"hidden_state": [0, 0, 0]},
            {"pegs": [True, 0, 0]},
            {"terminated": True},
            {"truncated": True},
            {"steps": True},
            {"goal": [1, 1, 1]},
            {"goal": [2.0, 2, 2]},
        ):
            with self.subTest(changed=changed):
                public = {**EXAMPLE_INPUTS["knowledge"]["public"], **changed}
                provider = Provider(DEFAULT_CONFIG)
                with self.assertRaises(ValueError):
                    provider.invoke(call_for(public))
                provider.close()

    def test_completion_cannot_substitute_an_illegal_or_differently_bound_command(self):
        inputs = deepcopy(EXAMPLE_INPUTS)
        valid = run_example()
        for change in (
            {"actions": [{"disk": 2, "target": 2}]},
            {"invocation": None},
            {"remaining_moves": 0},
            {"actions": []},
        ):
            with self.subTest(change=change):
                output = {"plan": {**valid["plan"], **change}}
                with self.assertRaises(ValueError):
                    validate_completion(output, inputs)

    def test_contract_lifecycle_and_services(self):
        provider = Provider(DEFAULT_CONFIG)
        call = call_for(EXAMPLE_INPUTS["knowledge"]["public"])
        with self.assertRaises(ValueError):
            provider.invoke({**call, "contract": "symbolic.predict"})
        self.assertFalse(provider.cancel("request"))
        provider.close()
        with self.assertRaises(RuntimeError):
            provider.invoke(call)
        with self.assertRaises(ValueError):
            bind_services({"native": object()})

    def test_configuration_is_explicit_and_bounded(self):
        for config in (
            {},
            {"max_states": True, "max_depth": 7},
            {"max_states": 28, "max_depth": 7},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_config(config)


if __name__ == "__main__":
    unittest.main()
