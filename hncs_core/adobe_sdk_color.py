"""Narrow Adobe DNG SDK colour-spec simulator used by the triple-carrier study."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adobe_triple_illuminant import (
    IlluminantDescriptor,
    triple_illuminant_weights,
    xy_to_temperature_tint,
)
from .colour import D50_WHITE_XY, xy_to_xyz


_BRADFORD = np.asarray(
    ((0.8951, 0.2664, -0.1614),
     (-0.7502, 1.7135, 0.0367),
     (0.0389, -0.0685, 1.0296)),
    dtype=np.float64,
)
_PCS_XYZ = xy_to_xyz(D50_WHITE_XY)


def map_white_matrix(white1: tuple[float, float], white2: tuple[float, float]) -> np.ndarray:
    """Port of DNG SDK ``MapWhiteMatrix`` (linear Bradford, scales pinned)."""
    w1 = np.maximum(_BRADFORD @ xy_to_xyz(white1), 0.0)
    w2 = np.maximum(_BRADFORD @ xy_to_xyz(white2), 0.0)
    scale = np.where(w1 > 0.0, w2 / w1, 10.0)
    scale = np.clip(scale, 0.1, 10.0)
    return np.linalg.inv(_BRADFORD) @ np.diag(scale) @ _BRADFORD


def normalize_color_matrix(matrix: np.ndarray) -> np.ndarray:
    """Port of ``NormalizeColorMatrix`` including four-decimal storage rounding."""
    result = np.asarray(matrix, dtype=np.float64).copy()
    maximum = float(np.max(result @ _PCS_XYZ))
    if maximum > 0.0 and (maximum < 0.99 or maximum > 1.01):
        result /= maximum
    return np.rint(result * 10000.0) / 10000.0


def stored_forward_matrix(matrix: np.ndarray) -> np.ndarray:
    """Apply SetForwardMatrix four-decimal storage rounding."""
    return np.rint(np.asarray(matrix, dtype=np.float64) * 10000.0) / 10000.0


def normalize_forward_matrix(matrix: np.ndarray) -> np.ndarray:
    """Port of ``NormalizeForwardMatrix`` called by ``dng_color_spec``."""
    result = np.asarray(matrix, dtype=np.float64)
    mapped_one = result @ np.ones(3, dtype=np.float64)
    if np.any(mapped_one == 0.0):
        raise ValueError("ForwardMatrix maps camera one to zero")
    return np.diag(_PCS_XYZ / mapped_one) @ result


@dataclass(frozen=True)
class ColorState:
    white_xy: tuple[float, float]
    camera_white: np.ndarray
    pcs_to_camera: np.ndarray
    camera_to_pcs: np.ndarray
    color_matrix: np.ndarray
    forward_matrix: np.ndarray
    weights: np.ndarray


class DualColorSpec:
    """Three-channel dual-illuminant profile with identity calibration/analog balance."""

    def __init__(
        self,
        white1_xy: tuple[float, float],
        white2_xy: tuple[float, float],
        color1: np.ndarray,
        color2: np.ndarray,
        forward1: np.ndarray,
        forward2: np.ndarray,
    ) -> None:
        self.white1_xy = white1_xy
        self.white2_xy = white2_xy
        self.temperature1 = xy_to_temperature_tint(*white1_xy)[0]
        self.temperature2 = xy_to_temperature_tint(*white2_xy)[0]
        self.colors = (np.asarray(color1, dtype=np.float64), np.asarray(color2, dtype=np.float64))
        self.forwards = (
            normalize_forward_matrix(forward1),
            normalize_forward_matrix(forward2),
        )

    def weights(self, white_xy: tuple[float, float]) -> np.ndarray:
        temperature = xy_to_temperature_tint(*white_xy)[0]
        if temperature <= self.temperature1:
            g = 1.0
        elif temperature >= self.temperature2:
            g = 0.0
        else:
            inverse = 1.0 / temperature
            g = ((inverse - 1.0 / self.temperature2) /
                 (1.0 / self.temperature1 - 1.0 / self.temperature2))
        return np.asarray((g, 1.0 - g), dtype=np.float64)

    def matrices(self, white_xy: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        weights = self.weights(white_xy)
        color = weights[0] * self.colors[0] + weights[1] * self.colors[1]
        forward = weights[0] * self.forwards[0] + weights[1] * self.forwards[1]
        return color, forward, weights

    def set_white_xy(self, white_xy: tuple[float, float]) -> ColorState:
        color, forward, weights = self.matrices(white_xy)
        return _set_white(white_xy, color, forward, weights)

    def neutral_to_xy(self, neutral: np.ndarray) -> tuple[float, float]:
        return _neutral_to_xy(self, neutral)


class TripleColorSpec:
    """Three-channel DNG 1.6 triple-illuminant profile."""

    def __init__(
        self,
        descriptors: tuple[IlluminantDescriptor, IlluminantDescriptor, IlluminantDescriptor],
        colors: np.ndarray,
        forwards: np.ndarray,
    ) -> None:
        self.descriptors = descriptors
        self.colors = np.asarray(colors, dtype=np.float64)
        self.forwards = np.asarray(
            [normalize_forward_matrix(item) for item in np.asarray(forwards, dtype=np.float64)],
            dtype=np.float64,
        )

    def weights(self, white_xy: tuple[float, float]) -> np.ndarray:
        temperature, tint = xy_to_temperature_tint(*white_xy)
        return triple_illuminant_weights(temperature, self.descriptors, tint=tint)

    def matrices(self, white_xy: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        weights = np.asarray(self.weights(white_xy), dtype=np.float64)
        return (
            np.tensordot(weights, self.colors, axes=1),
            np.tensordot(weights, self.forwards, axes=1),
            weights,
        )

    def set_white_xy(self, white_xy: tuple[float, float]) -> ColorState:
        color, forward, weights = self.matrices(white_xy)
        return _set_white(white_xy, color, forward, weights)

    def neutral_to_xy(self, neutral: np.ndarray) -> tuple[float, float]:
        return _neutral_to_xy(self, neutral)


def _set_white(
    white_xy: tuple[float, float],
    color: np.ndarray,
    forward: np.ndarray,
    weights: np.ndarray,
) -> ColorState:
    camera_white = color @ xy_to_xyz(white_xy)
    maximum = float(np.max(camera_white))
    if maximum == 0.0:
        raise ValueError("ColorMatrix produces zero camera white")
    camera_white = np.clip(camera_white / maximum, 0.001, 1.0)
    pcs_to_camera = color @ map_white_matrix(D50_WHITE_XY, white_xy)
    scale = float(np.max(pcs_to_camera @ _PCS_XYZ))
    if scale == 0.0:
        raise ValueError("PCS-to-camera white scale is zero")
    pcs_to_camera = pcs_to_camera / scale
    camera_to_pcs = forward @ np.diag(1.0 / camera_white)
    return ColorState(
        white_xy=white_xy,
        camera_white=camera_white,
        pcs_to_camera=pcs_to_camera,
        camera_to_pcs=camera_to_pcs,
        color_matrix=color,
        forward_matrix=forward,
        weights=np.asarray(weights, dtype=np.float64),
    )


def _neutral_to_xy(spec: DualColorSpec | TripleColorSpec, neutral: np.ndarray) -> tuple[float, float]:
    """Exact 30-pass fixed-point structure of DNG SDK ``NeutralToXY``."""
    last = tuple(float(value) for value in D50_WHITE_XY)
    vector = np.asarray(neutral, dtype=np.float64)
    for pass_index in range(30):
        color, _, _ = spec.matrices(last)
        xyz = np.linalg.inv(color) @ vector
        total = float(np.sum(xyz))
        if total == 0.0:
            raise ValueError("NeutralToXY produced zero XYZ sum")
        next_xy = (float(xyz[0] / total), float(xyz[1] / total))
        if abs(next_xy[0] - last[0]) + abs(next_xy[1] - last[1]) < 1.0e-7:
            return next_xy
        if pass_index == 29:
            next_xy = ((last[0] + next_xy[0]) * 0.5, (last[1] + next_xy[1]) * 0.5)
        last = next_xy
    return last


def adobe_line_slots(
    color1: np.ndarray,
    color2: np.ndarray,
    forward1: np.ndarray,
    forward2: np.ndarray,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct stored CM/FM slots on the original Adobe A-D65 line."""
    colors = []
    forwards = []
    for coefficient in np.asarray(coefficients, dtype=np.float64):
        colors.append(normalize_color_matrix(coefficient * color1 + (1.0 - coefficient) * color2))
        forwards.append(stored_forward_matrix(coefficient * forward1 + (1.0 - coefficient) * forward2))
    return np.asarray(colors), np.asarray(forwards)
