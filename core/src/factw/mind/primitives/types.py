from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Literal

type ParameterValue = str | bool | int | float


@dataclass(frozen=True)
class EntityRef:
    kind: str
    key: str


@dataclass(frozen=True)
class BindingSlot:
    name: str
    entity_kind: str


@dataclass(frozen=True)
class Parameter:
    name: str
    kind: Literal["number", "integer", "boolean", "string"]
    unit: str | None = None

    def accepts(self, value: ParameterValue) -> bool:
        match self.kind:
            case "number":
                return type(value) is int or (type(value) is float and isfinite(value))
            case "integer":
                return type(value) is int
            case "boolean":
                return type(value) is bool
            case "string":
                return type(value) is str
            case _:
                raise ValueError(f"Unknown parameter kind: {self.kind}")


@dataclass(frozen=True)
class PrimitiveType:
    name: str
    bindings: tuple[BindingSlot, ...] = ()
    parameters: tuple[Parameter, ...] = ()

    def __post_init__(self):
        names = [item.name for item in (*self.bindings, *self.parameters)]
        if not self.name or any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError("Primitive and argument names must be nonempty and unambiguous")

    def bind(
        self,
        bindings: Mapping[str, EntityRef],
        parameters: Mapping[str, ParameterValue] | None = None,
    ) -> PrimitiveInvocation:
        return PrimitiveInvocation(self, bindings, {} if parameters is None else parameters)


@dataclass(frozen=True)
class PrimitiveInvocation:
    primitive: PrimitiveType
    bindings: Mapping[str, EntityRef]
    parameters: Mapping[str, ParameterValue] = field(default_factory=dict)

    def __post_init__(self):
        definition = self.primitive
        if set(self.bindings) != {slot.name for slot in definition.bindings}:
            raise ValueError("Bindings must match the primitive's declared roles")
        for slot in definition.bindings:
            entity = self.bindings[slot.name]
            if not isinstance(entity, EntityRef) or entity.kind != slot.entity_kind:
                raise ValueError(f"Wrong entity kind for binding {slot.name}")
        if set(self.parameters) != {parameter.name for parameter in definition.parameters}:
            raise ValueError("Parameters must match the primitive's declaration")
        for parameter in definition.parameters:
            if not parameter.accepts(self.parameters[parameter.name]):
                raise ValueError(f"Invalid value for parameter {parameter.name}")
        object.__setattr__(self, "bindings", MappingProxyType(dict(self.bindings)))
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
