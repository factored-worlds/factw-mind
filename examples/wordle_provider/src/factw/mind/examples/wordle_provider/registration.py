"""Bind only declared host operations; this example requires none."""

from collections.abc import Mapping

from .provider import Provider


def bind_services(operations):
    if not isinstance(operations, Mapping) or operations:
        raise ValueError("this provider requires an empty operation binding")
    return Provider
