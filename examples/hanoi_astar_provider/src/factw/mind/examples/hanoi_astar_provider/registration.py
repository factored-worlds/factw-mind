from collections.abc import Mapping

from .provider import Provider


def bind_services(operations):
    if not isinstance(operations, Mapping) or operations:
        raise ValueError("the pure Hanoi planner requires an empty operation binding")
    return Provider
