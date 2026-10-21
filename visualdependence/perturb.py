"""Erasure and translation."""
from __future__ import annotations
import numpy as np


def body_mean(image: np.ndarray, body: np.ndarray) -> float:
    """The mean intensity of the body: the fill value of every erasure."""
    return float(np.asarray(image, float)[np.asarray(body, bool)].mean())


def erase(image: np.ndarray, region: np.ndarray, fill: float) -> np.ndarray:
    """x_O: x with region O filled by `fill`."""
    out = np.asarray(image, float).copy()
    out[np.asarray(region, bool)] = fill
    return out


def translate(image: np.ndarray, delta: int, axis: int, fill: float) -> np.ndarray:
    """x^(delta): x shifted by delta along `axis`; the vacated strip is filled with `fill`."""
    image = np.asarray(image, float)
    out = np.full_like(image, fill)
    src, dst = [slice(None)] * image.ndim, [slice(None)] * image.ndim
    n = image.shape[axis]
    if delta >= 0:
        src[axis], dst[axis] = slice(0, n - delta), slice(delta, n)
    else:
        src[axis], dst[axis] = slice(-delta, n), slice(0, n + delta)
    out[tuple(dst)] = image[tuple(src)]
    return out


def translate_mask(mask: np.ndarray, delta: int, axis: int) -> np.ndarray:
    """The mask shifted with the image."""
    return translate(np.asarray(mask, float), delta, axis, 0.0) > 0.5
