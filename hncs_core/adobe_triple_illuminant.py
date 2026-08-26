"""Adobe DNG SDK 1.7 triple-illuminant HueSatMap carrier semantics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Wyszecki & Stiles table copied from the public DNG SDK dng_temperature.cpp.
_TEMP_TABLE = np.asarray(
    [
        (0, .18006, .26352, -.24341), (10, .18066, .26589, -.25479),
        (20, .18133, .26846, -.26876), (30, .18208, .27119, -.28539),
        (40, .18293, .27407, -.30470), (50, .18388, .27709, -.32675),
        (60, .18494, .28021, -.35156), (70, .18611, .28342, -.37915),
        (80, .18740, .28668, -.40955), (90, .18880, .28997, -.44278),
        (100, .19032, .29326, -.47888), (125, .19462, .30141, -.58204),
        (150, .19962, .30921, -.70471), (175, .20525, .31647, -.84901),
        (200, .21142, .32312, -1.0182), (225, .21807, .32909, -1.2168),
        (250, .22511, .33439, -1.4512), (275, .23247, .33904, -1.7298),
        (300, .24010, .34308, -2.0637), (325, .24702, .34655, -2.4681),
        (350, .25591, .34951, -2.9641), (375, .26400, .35200, -3.5814),
        (400, .27218, .35407, -4.3633), (425, .28039, .35577, -5.3762),
        (450, .28863, .35714, -6.7262), (475, .29685, .35823, -8.5955),
        (500, .30505, .35907, -11.324), (525, .31320, .35968, -15.628),
        (550, .32129, .36011, -23.325), (575, .32931, .36038, -40.770),
        (600, .33724, .36051, -116.45),
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class IlluminantDescriptor:
    """Legal DNG 1.6 ``lsOther`` white described by temperature and tint."""

    temperature: float
    tint: float = 0.0

    @property
    def white_xy(self) -> tuple[float, float]:
        return temperature_tint_to_xy(self.temperature, self.tint)


def temperature_tint_to_xy(temperature: float, tint: float = 0.0) -> tuple[float, float]:
    """Exact port of ``dng_temperature::Get_xy_coord``."""
    r = 1.0e6 / float(temperature)
    offset = float(tint) / -3000.0
    for index in range(30):
        if r < _TEMP_TABLE[index + 1, 0] or index == 29:
            first, second = _TEMP_TABLE[index], _TEMP_TABLE[index + 1]
            fraction = (second[0] - r) / (second[0] - first[0])
            u = first[1] * fraction + second[1] * (1.0 - fraction)
            v = first[2] * fraction + second[2] * (1.0 - fraction)
            vector1 = np.array((1.0, first[3]), dtype=np.float64)
            vector2 = np.array((1.0, second[3]), dtype=np.float64)
            vector1 /= np.linalg.norm(vector1)
            vector2 /= np.linalg.norm(vector2)
            vector3 = vector1 * fraction + vector2 * (1.0 - fraction)
            vector3 /= np.linalg.norm(vector3)
            u += vector3[0] * offset
            v += vector3[1] * offset
            denominator = u - 4.0 * v + 2.0
            return 1.5 * u / denominator, v / denominator
    raise AssertionError("unreachable")


def xy_to_temperature_tint(x: float, y: float) -> tuple[float, float]:
    """Exact port of ``dng_temperature::Set_xy_coord``."""
    denominator = 1.5 - x + 6.0 * y
    u = 2.0 * x / denominator
    v = 3.0 * y / denominator
    last_dt = last_du = last_dv = 0.0
    for index in range(1, 31):
        du, dv = 1.0, _TEMP_TABLE[index, 3]
        length = np.hypot(du, dv)
        du, dv = du / length, dv / length
        uu = u - _TEMP_TABLE[index, 1]
        vv = v - _TEMP_TABLE[index, 2]
        dt = -uu * dv + vv * du
        if dt <= 0.0 or index == 30:
            dt = -min(dt, 0.0)
            fraction = 0.0 if index == 1 else dt / (last_dt + dt)
            reciprocal = (
                _TEMP_TABLE[index - 1, 0] * fraction
                + _TEMP_TABLE[index, 0] * (1.0 - fraction)
            )
            temperature = 1.0e6 / reciprocal
            uu = u - (
                _TEMP_TABLE[index - 1, 1] * fraction
                + _TEMP_TABLE[index, 1] * (1.0 - fraction)
            )
            vv = v - (
                _TEMP_TABLE[index - 1, 2] * fraction
                + _TEMP_TABLE[index, 2] * (1.0 - fraction)
            )
            du = du * (1.0 - fraction) + last_du * fraction
            dv = dv * (1.0 - fraction) + last_dv * fraction
            length = np.hypot(du, dv)
            du, dv = du / length, dv / length
            tint = (uu * du + vv * dv) * -3000.0
            return float(temperature), float(tint)
        last_dt, last_du, last_dv = dt, du, dv
    raise AssertionError("unreachable")


def _smooth_step(values: np.ndarray) -> np.ndarray:
    return values * values * (3.0 - 2.0 * values)


def triple_illuminant_weights(
    temperatures: np.ndarray | float,
    descriptors: tuple[IlluminantDescriptor, IlluminantDescriptor, IlluminantDescriptor],
    tint: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Port of DNG SDK ``CalculateTripleIlluminantWeights``.

    Returned weights are float64. ``interpolate_payloads`` performs the SDK's
    subsequent float32 conversion used by ``dng_hue_sat_map::Interpolate``.
    """
    temperature = np.asarray(temperatures, dtype=np.float64)
    query_tint = np.broadcast_to(np.asarray(tint, dtype=np.float64), temperature.shape)
    query = np.stack(
        (query_tint / 200.0, np.minimum(1500.0 / temperature, 1.0)), axis=-1
    )
    centers = np.asarray(
        [
            (item.tint / 200.0, min(1500.0 / item.temperature, 1.0))
            for item in descriptors
        ],
        dtype=np.float64,
    )
    distances = np.sum((query[..., None, :] - centers) ** 2, axis=-1)
    weights = 1.0 / (distances + 1.0e-8)
    weights /= np.sum(weights, axis=-1, keepdims=True)
    weights = _smooth_step(weights)
    weights = np.clip((weights - 0.02) / 0.98, 0.0, 1.0)
    renormalizer = np.sum(weights, axis=-1)
    w1 = weights[..., 0] / renormalizer
    w2 = weights[..., 1] / renormalizer
    w3 = np.maximum(1.0 - w1 - w2, 0.0)
    return np.stack((w1, w2, w3), axis=-1)


def interpolate_payloads(weights: np.ndarray, payloads: np.ndarray) -> np.ndarray:
    """Blend three HueSatMaps using the SDK's float32 arithmetic contract."""
    w1 = np.asarray(weights[..., 0], dtype=np.float32)
    w2 = np.asarray(weights[..., 1], dtype=np.float32)
    w3 = np.float32(1.0) - (w1 + w2)
    maps = np.asarray(payloads, dtype=np.float32)
    return (
        w1[..., None, None] * maps[0]
        + w2[..., None, None] * maps[1]
        + w3[..., None, None] * maps[2]
    ).astype(np.float32)
