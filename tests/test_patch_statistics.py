import base64

import pytest

from factw.mind.observations.patch_statistics import RGBPatch, summarize_rgb_patches


@pytest.fixture
def rgb():
    pixels = bytes([
        1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 250, 251, 252, 253, 254, 255,
    ])
    return {
        "width": 3,
        "height": 2,
        "encoding": "rgb_u8_base64",
        "pixels": base64.b64encode(pixels).decode("ascii"),
    }


def test_patch_sums_use_frame_stride_and_preserve_patch_order(rgb):
    patches = [
        RGBPatch(row=1, column=0, x=0, y=1, width=1, height=1),
        RGBPatch(row=0, column=1, x=1, y=0, width=2, height=2),
    ]
    assert summarize_rgb_patches(rgb, patches) == {
        "kind": "rgb_patch_sums/1",
        "cells": [
            {"row": 1, "column": 0, "count": 1, "rgb_sum": [10, 11, 12]},
            {"row": 0, "column": 1, "count": 4, "rgb_sum": [514, 518, 522]},
        ],
    }


@pytest.mark.parametrize(
    "field,value",
    [("width", True), ("height", 0), ("encoding", "png"), ("pixels", "!"), ("pixels", "AA==")],
)
def test_invalid_frame_is_rejected_even_without_patches(rgb, field, value):
    with pytest.raises(ValueError):
        summarize_rgb_patches({**rgb, field: value}, [])


@pytest.mark.parametrize("x,width", [(-1, 1), (True, 1), (0, 0)])
def test_patch_requires_nonnegative_coordinates_and_positive_size(x, width):
    with pytest.raises(ValueError):
        RGBPatch(row=0, column=0, x=x, y=0, width=width, height=1)


@pytest.mark.parametrize("x,y", [(2, 0), (0, 1)])
def test_patch_extending_past_either_edge_is_rejected(rgb, x, y):
    with pytest.raises(ValueError, match="outside"):
        summarize_rgb_patches(rgb, [RGBPatch(0, 0, x, y, width=2, height=2)])
