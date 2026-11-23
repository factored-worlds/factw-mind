"""A separate compatible registration using the canonical consistent policy."""

from collections.abc import Mapping

from factw.mind.baselines.wordle import RandomPolicy

from .provider import Provider


def consistent_policy(allowed_guesses):
    return RandomPolicy(allowed_guesses, consistent=True, seed=0)


class ConsistentProvider(Provider):
    policy_factory = staticmethod(consistent_policy)


def bind_services(operations):
    if not isinstance(operations, Mapping) or operations:
        raise ValueError("this provider requires an empty operation binding")
    return ConsistentProvider
