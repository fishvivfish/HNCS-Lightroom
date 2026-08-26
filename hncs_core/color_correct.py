"""Recovered HNCS ColorCorrect numerical operator used by the WB model."""
from __future__ import annotations

from typing import Any
import numpy as np

from .colour import apply_matrix


def as_samples(rgb: Any) -> tuple[np.ndarray, tuple[int, ...]]:
    values = np.asarray(rgb, dtype=np.float64)
    if values.shape == (3,):
        return values.reshape(1, 3), values.shape
    if values.ndim < 1 or values.shape[-1] != 3:
        raise ValueError("RGB input must have shape (3,) or (..., 3)")
    return values.reshape(-1, 3), values.shape


def restore_samples(samples: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    return samples.reshape(shape)


def sample_float2(texture: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    height, width, _ = texture.shape
    px, py = x - 0.5, y - 0.5
    x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
    fx, fy = px - x0, py - y0

    def take(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        return texture[np.clip(iy, 0, height - 1), np.clip(ix, 0, width - 1)]

    c00 = take(x0, y0)
    c10 = take(x0 + 1, y0)
    c01 = take(x0, y0 + 1)
    c11 = take(x0 + 1, y0 + 1)
    a = c00 * (1.0 - fx[:, None]) + c10 * fx[:, None]
    b = c01 * (1.0 - fx[:, None]) + c11 * fx[:, None]
    return a * (1.0 - fy[:, None]) + b * fy[:, None]


def apply_color_correct(
    rgb: np.ndarray,
    params: dict,
    texture: np.ndarray,
    *,
    clip_input: bool = True,
    weighting_rgb: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float64)
    rgb16 = (np.maximum(values, 0.0) if clip_input else values) * 65535.0
    weighting16 = (
        rgb16
        if weighting_rgb is None
        else np.maximum(np.asarray(weighting_rgb, dtype=np.float64), 0.0) * 65535.0
    )
    average = np.maximum(np.sum(weighting16, axis=1) / 3.0, 1e-5)
    chroma_distance = np.max(np.abs(weighting16 - average[:, None]), axis=1) / average
    low, high = params["desat_gray"]
    if high > low:
        gray_factor = np.clip((chroma_distance - low) / (high - low), 0.0, 1.0)
    else:
        gray_factor = np.ones_like(chroma_distance)

    ycc = apply_matrix(rgb16, np.asarray(params["input_matrix"], dtype=np.float64))
    y, cb, cr = ycc[:, 0], ycc[:, 1], ycc[:, 2]
    safe_y = np.maximum(y, 1e-5)
    dark_x, dark_a, dark_b, dark_c = params["dark_params"]
    dark_limit = np.where(y < dark_x, dark_a * y * y + dark_b * y + dark_c, 1.0)
    chroma_scale = np.minimum(gray_factor, dark_limit)

    division = params["div_factor"]
    start_cb, start_cr = params["start_cbcr"]
    limit_cb, limit_cr = params["cbcr_limits"]
    cc_x = np.clip(cb * (division / safe_y) - start_cb + 0.5, 0.5, limit_cb - 0.5)
    cc_y = np.clip(cr * (division / safe_y) - start_cr + 0.5, 0.5, limit_cr - 0.5)
    lut = sample_float2(texture, cc_x, cc_y)
    corrected_ycc = np.column_stack(
        (safe_y, chroma_scale * safe_y * lut[:, 0], chroma_scale * safe_y * lut[:, 1])
    )
    output = apply_matrix(corrected_ycc, np.asarray(params["output_matrix"], dtype=np.float64))
    return np.maximum(output / 65535.0, 0.0)
