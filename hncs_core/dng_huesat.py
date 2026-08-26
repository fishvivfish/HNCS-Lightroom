"""Minimal DNG HueSatMap HSV/interpolation helpers used by HNCS Color."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GridSpec:
    hue: int
    saturation: int
    value: int

    def __post_init__(self) -> None:
        if self.hue < 1 or self.saturation < 2 or self.value < 1:
            raise ValueError(f"invalid DNG HueSatMap dimensions: {self.label}")

    @property
    def nodes(self) -> int:
        return self.hue * self.saturation * self.value

    @property
    def label(self) -> str:
        return f"{self.hue}x{self.saturation}x{self.value}"


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(rgb, dtype=np.float64), 0.0)
    maximum = np.max(values, axis=-1)
    minimum = np.min(values, axis=-1)
    gap = maximum - minimum
    hue = np.zeros_like(maximum)
    nonzero = gap > 0.0
    red = nonzero & (values[..., 0] == maximum)
    green = nonzero & ~red & (values[..., 1] == maximum)
    blue = nonzero & ~red & ~green
    hue[red] = (values[..., 1][red] - values[..., 2][red]) / gap[red]
    hue[red & (hue < 0.0)] += 6.0
    hue[green] = 2.0 + (values[..., 2][green] - values[..., 0][green]) / gap[green]
    hue[blue] = 4.0 + (values[..., 0][blue] - values[..., 1][blue]) / gap[blue]
    saturation = np.zeros_like(maximum)
    saturation[nonzero] = gap[nonzero] / maximum[nonzero]
    return np.stack((hue, saturation, maximum), axis=-1)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    values = np.asarray(hsv, dtype=np.float64)
    hue, saturation, value = values[..., 0], values[..., 1], values[..., 2]
    result = np.repeat(value[..., None], 3, axis=-1)
    active = saturation > 0.0
    if not np.any(active):
        return result
    h = np.mod(hue[active], 6.0)
    sector = h.astype(np.int32)
    fraction = h - sector
    s, v = saturation[active], value[active]
    p = v * (1.0 - s)
    q = v * (1.0 - s * fraction)
    t = v * (1.0 - s * (1.0 - fraction))
    choices = np.stack(
        (
            np.stack((v, t, p), axis=-1),
            np.stack((q, v, p), axis=-1),
            np.stack((p, v, t), axis=-1),
            np.stack((p, q, v), axis=-1),
            np.stack((t, p, v), axis=-1),
            np.stack((v, p, q), axis=-1),
        ),
        axis=1,
    )
    result[active] = choices[np.arange(len(sector)), sector % 6]
    return result


def apply_payload_rgb(spec: GridSpec, payload: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Apply a DNG HueSatMap by trilinear interpolation."""
    hsv = rgb_to_hsv(np.asarray(rgb, dtype=np.float64))
    hue_scaled = hsv[:, 0] * spec.hue / 6.0
    sat_scaled = np.clip(hsv[:, 1], 0.0, 1.0) * (spec.saturation - 1)
    hbase = np.floor(hue_scaled)
    h0 = hbase.astype(np.int64) % spec.hue
    h1 = (h0 + 1) % spec.hue
    s0 = np.minimum(np.floor(sat_scaled).astype(np.int64), spec.saturation - 2)
    s1 = s0 + 1
    hf = hue_scaled - hbase
    sf = sat_scaled - s0
    if spec.value > 1:
        val_scaled = np.clip(hsv[:, 2], 0.0, 1.0) * (spec.value - 1)
        v0 = np.minimum(np.floor(val_scaled).astype(np.int64), spec.value - 2)
        v1 = v0 + 1
        vf = val_scaled - v0
    else:
        v0 = v1 = np.zeros(len(hsv), dtype=np.int64)
        vf = np.zeros(len(hsv), dtype=np.float64)
    grid = np.asarray(payload, dtype=np.float64).reshape(
        spec.value, spec.hue, spec.saturation, 3
    )
    delta = np.zeros((len(hsv), 3), dtype=np.float64)
    for vi, vw in ((v0, 1.0 - vf), (v1, vf)):
        for hi, hw in ((h0, 1.0 - hf), (h1, hf)):
            for si, sw in ((s0, 1.0 - sf), (s1, sf)):
                delta += grid[vi, hi, si] * (vw * hw * sw)[:, None]
    output_hsv = np.empty_like(hsv)
    output_hsv[:, 0] = hsv[:, 0] + delta[:, 0] / 60.0
    output_hsv[:, 1] = np.minimum(hsv[:, 1] * delta[:, 1], 1.0)
    output_hsv[:, 2] = np.clip(hsv[:, 2] * delta[:, 2], 0.0, 1.0)
    return hsv_to_rgb(output_hsv)
