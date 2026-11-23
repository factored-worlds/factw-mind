"""Compatible planner minimizing the largest exact feedback partition."""

from collections.abc import Mapping

from factw.mind.baselines.wordle import MinimaxPolicy

from .provider import Provider


class MinimaxProvider(Provider):
    policy_factory = MinimaxPolicy


def bind_services(operations):
    if not isinstance(operations, Mapping) or operations:
        raise ValueError("this provider requires an empty operation binding")
    return MinimaxProvider
