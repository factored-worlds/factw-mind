import pytest

from factw.mind.primitives import BindingSlot, EntityRef, Parameter, PrimitiveType


def move_type():
    return PrimitiveType(
        "move",
        (BindingSlot("object", "disk"), BindingSlot("destination", "peg")),
        (Parameter("speed", "number", "m/s"),),
    )


def test_type_is_reused_across_bindings_and_parameters():
    primitive = move_type()
    bindings = {"object": EntityRef("disk", "small"), "destination": EntityRef("peg", "left")}
    parameters = {"speed": 0.1}
    first = primitive.bind(bindings, parameters)
    bindings["destination"] = EntityRef("peg", "right")
    parameters["speed"] = 0.2
    second = primitive.bind(bindings, parameters)
    assert first.primitive is second.primitive
    assert first.bindings["destination"].key == "left"
    assert second.bindings["destination"].key == "right"
    assert first.parameters["speed"] == 0.1
    with pytest.raises(TypeError):
        first.bindings["object"] = EntityRef("disk", "large")


@pytest.mark.parametrize("speed", [True, "fast", float("nan"), float("inf")])
def test_numeric_parameter_is_finite_and_not_a_boolean(speed):
    with pytest.raises(ValueError, match="speed"):
        move_type().bind(
            {"object": EntityRef("disk", "small"), "destination": EntityRef("peg", "left")},
            {"speed": speed},
        )


def test_entity_roles_and_parameters_must_match_declarations():
    primitive = move_type()
    with pytest.raises(ValueError, match="declared roles"):
        primitive.bind({}, {"speed": 0.1})
    with pytest.raises(ValueError, match="entity kind"):
        primitive.bind(
            {"object": EntityRef("peg", "left"), "destination": EntityRef("peg", "right")},
            {"speed": 0.1},
        )
    with pytest.raises(ValueError, match="Parameters"):
        primitive.bind(
            {"object": EntityRef("disk", "small"), "destination": EntityRef("peg", "left")}
        )
    with pytest.raises(ValueError, match="unambiguous"):
        PrimitiveType("move", (BindingSlot("x", "disk"),), (Parameter("x", "integer"),))


def test_arbitrary_precision_integer_parameters_are_finite():
    assert Parameter("size", "number").accepts(10**1000)
