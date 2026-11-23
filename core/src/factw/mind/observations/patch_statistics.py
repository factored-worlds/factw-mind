"""Integer RGB patch summaries; these features do not assert semantic state."""

import base64
from dataclasses import dataclass


def unsigned(value, label, *, maximum=None, minimum=0):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"invalid unsigned integer {label}")


def validate_rgb(rgb):
    if type(rgb) is not dict or set(rgb) != {"width", "height", "encoding", "pixels"}:
        raise ValueError("RGB requires exactly width, height, encoding and pixels")
    for name in ("width", "height"):
        unsigned(rgb[name], name, minimum=1)
    if rgb["encoding"] != "rgb_u8_base64" or type(rgb["pixels"]) is not str:
        raise ValueError("RGB requires explicit rgb_u8_base64 encoding")
    pixels = base64.b64decode(rgb["pixels"], validate=True)
    if len(pixels) != rgb["width"] * rgb["height"] * 3:
        raise ValueError("RGB pixel count differs from its declared row-major shape")
    return pixels


@dataclass(frozen=True)
class RGBPatch:
    row: int
    column: int
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self):
        for name in ("row", "column", "x", "y"):
            unsigned(getattr(self, name), name)
        for name in ("width", "height"):
            unsigned(getattr(self, name), name, minimum=1)


def summarize_rgb_patches(rgb, patches):
    """Summarize supplied RGB pixels with exact sums, without resize or learned weights."""
    pixels = validate_rgb(rgb)
    return summarize_patches(pixels, rgb["width"], rgb["height"], patches)


def summarize_patches(pixels, width, height, patches):
    cells = []
    for patch in patches:
        if patch.x + patch.width > width or patch.y + patch.height > height:
            raise ValueError("declared visual patch lies outside the supplied RGB frame")
        sums = [0, 0, 0]
        for y in range(patch.y, patch.y + patch.height):
            for x in range(patch.x, patch.x + patch.width):
                start = (y * width + x) * 3
                for channel in range(3):
                    sums[channel] += pixels[start + channel]
        cells.append(
            {
                "row": patch.row,
                "column": patch.column,
                "count": patch.width * patch.height,
                "rgb_sum": sums,
            }
        )
    return {"kind": "rgb_patch_sums/1", "cells": cells}
