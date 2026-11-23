import unittest
from collections import deque
from copy import deepcopy
from itertools import product

from factw.mind.examples.hanoi_astar_provider import SearchLimitError
from factw.mind.examples.hanoi_astar_provider.demo import run_example
from factw.mind.examples.hanoi_astar_provider.provider import Provider
from factw.mind.examples.hanoi_astar_provider.registration import bind_services
from factw.mind.examples.hanoi_provider.contracts import validate_completion
from factw.mind.examples.hanoi_provider.demo import DEFAULT_CONFIG, EXAMPLE_INPUTS


def observation(board, steps=0):
    return {
        "pegs": list(board),
        "goal": [2, 2, 2],
        "steps": steps,
        "terminated": list(board) == [2, 2, 2],
        "truncated": False,
    }


def call_for(public):
    basis = {"revision": "astar_test", "tick": public["steps"]}
    return {
        "contract": "symbolic.plan",
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


def distances():
    goal = (2, 2, 2)
    result, pending = {goal: 0}, deque([goal])
    while pending:
        board = pending.popleft()
        top = {peg: next((disk for disk in range(3) if board[disk] == peg), 3) for peg in range(3)}
        for source in range(3):
            disk = top[source]
            for target in range(3):
                if disk == 3 or source == target or disk >= top[target]:
                    continue
                successor = tuple(
                    target if index == disk else peg for index, peg in enumerate(board)
                )
                if successor not in result:
                    result[successor] = result[board] + 1
                    pending.append(successor)
    return result


class HanoiAstarTests(unittest.TestCase):
    def test_every_board_has_exact_shortest_remaining_distance_and_legal_bound_moves(self):
        oracle = distances()
        self.assertEqual(len(oracle), 27)
        provider = Provider(DEFAULT_CONFIG)
        self.addCleanup(provider.close)
        for board in product(range(3), repeat=3):
            with self.subTest(board=board):
                public = observation(board)
                while not public["terminated"]:
                    call = call_for(public)
                    original = deepcopy(call)
                    record = provider.invoke(call)["plan"]
                    plan = record["value"]
                    self.assertEqual(plan["remaining_moves"], oracle[tuple(public["pegs"])])
                    self.assertGreater(plan["expanded_states"], 0)
                    self.assertLessEqual(plan["expanded_states"], 27)
                    action = plan["actions"][0]
                    disk, target = action["disk"], action["target"]
                    source = public["pegs"][disk]
                    self.assertNotEqual(source, target)
                    self.assertTrue(
                        all(peg not in (source, target) for peg in public["pegs"][:disk])
                    )
                    self.assertEqual(
                        plan["invocation"]["bindings"],
                        {
                            "disk": {"kind": "disk", "key": f"disk_{disk}"},
                            "destination": {"kind": "peg", "key": f"peg_{target}"},
                        },
                    )
                    self.assertEqual(call, original)
                    self.assertEqual(record["basis"], call["basis"])
                    self.assertEqual(record["sequence"], public["steps"] + 1)
                    public["pegs"][disk] = target
                    public["steps"] += 1
                    public["terminated"] = public["pegs"] == [2, 2, 2]
                    self.assertLessEqual(public["steps"], 7)
                self.assertEqual(public["steps"], oracle[board])
                self.assertEqual(
                    provider.invoke(call_for(public))["plan"]["value"],
                    {
                        "actions": [],
                        "invocation": None,
                        "remaining_moves": 0,
                        "expanded_states": 0,
                    },
                )

    def test_actual_sdk_invocation_and_unchanged_semantic_validator(self):
        inputs = deepcopy(EXAMPLE_INPUTS)
        output = run_example(inputs["knowledge"]["public"])
        self.assertEqual(output["plan"]["actions"], [{"disk": 0, "target": 2}])
        self.assertEqual(output["plan"]["remaining_moves"], 7)
        validate_completion(output, inputs)
        for changed in (
            {"actions": [{"disk": 2, "target": 2}]},
            {"invocation": None},
            {"remaining_moves": 0},
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_completion({"plan": {**output["plan"], **changed}}, inputs)

    def test_state_and_depth_limits_fail_without_output_or_input_changes(self):
        for config in (
            {"max_states": 1, "max_depth": 7},
            {"max_states": 27, "max_depth": 6},
            {"max_states": 27, "max_depth": 0},
        ):
            with self.subTest(config=config):
                provider = Provider(config)
                self.addCleanup(provider.close)
                call = call_for(observation((0, 0, 0)))
                before = deepcopy(call)
                with self.assertRaises(SearchLimitError):
                    provider.invoke(call)
                self.assertEqual(call, before)

    def test_goal_and_one_move_boundary_use_exact_budgets(self):
        provider = Provider({"max_states": 2, "max_depth": 1})
        self.addCleanup(provider.close)
        self.assertEqual(
            provider.invoke(call_for(observation((0, 2, 2))))["plan"]["value"]["remaining_moves"], 1
        )
        solved = Provider({"max_states": 1, "max_depth": 0})
        self.addCleanup(solved.close)
        self.assertEqual(
            solved.invoke(call_for(observation((2, 2, 2))))["plan"]["value"]["expanded_states"], 0
        )

    def test_ungranted_or_inconsistent_observations_are_rejected(self):
        provider = Provider(DEFAULT_CONFIG)
        self.addCleanup(provider.close)
        for changed in (
            {"hidden_state": [0, 0, 0]},
            {"pegs": [True, 0, 0]},
            {"terminated": True},
            {"truncated": True},
            {"steps": True},
            {"goal": [1, 1, 1]},
            {"goal": [2.0, 2, 2]},
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                provider.invoke(call_for({**observation((0, 0, 0)), **changed}))

    def test_configuration_contract_and_lifecycle_remain_explicit(self):
        for config in (
            {},
            {"max_states": True, "max_depth": 7},
            {"max_states": 28, "max_depth": 7},
            {"max_states": 27, "max_depth": 8},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                Provider(config)
        provider = bind_services({})(DEFAULT_CONFIG)
        self.assertFalse(provider.cancel("request"))
        call = call_for(observation((0, 0, 0)))
        with self.assertRaises(ValueError):
            provider.invoke({**call, "contract": "symbolic.predict"})
        with self.assertRaises(ValueError):
            bind_services({"native": object()})
        provider.close()
        with self.assertRaises(RuntimeError):
            provider.invoke(call)


if __name__ == "__main__":
    unittest.main()
