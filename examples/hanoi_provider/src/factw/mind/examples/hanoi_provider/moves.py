from factw.mind.primitives import BindingSlot, EntityRef, PrimitiveType

MOVE = PrimitiveType("move", (BindingSlot("disk", "disk"), BindingSlot("destination", "peg")))


def is_legal(pegs, disk, target):
    source = pegs[disk]
    return source != target and all(peg not in (source, target) for peg in pegs[:disk])


def bind_move(disk, target):
    return MOVE.bind(
        {
            "disk": EntityRef("disk", f"disk_{disk}"),
            "destination": EntityRef("peg", f"peg_{target}"),
        }
    )


def describe_move(invocation):
    return {
        "primitive": invocation.primitive.name,
        "bindings": {
            name: {"kind": entity.kind, "key": entity.key}
            for name, entity in invocation.bindings.items()
        },
    }


def command_for(invocation):
    disks = {EntityRef("disk", f"disk_{index}"): index for index in range(3)}
    pegs = {EntityRef("peg", f"peg_{index}"): index for index in range(3)}
    return {
        "disk": disks[invocation.bindings["disk"]],
        "target": pegs[invocation.bindings["destination"]],
    }
