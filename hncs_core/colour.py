"""Small, explicit colour-matrix helpers used by the HNCS pipeline."""

from __future__ import annotations

import numpy as np


D65_WHITE_XY = (0.3127, 0.3290)
D50_WHITE_XY = (0.3457, 0.3585)
V_GAMUT_PRIMARIES = (0.730, 0.280, 0.165, 0.840, 0.100, -0.030)
REC709_PRIMARIES = (0.640, 0.330, 0.300, 0.600, 0.150, 0.060)
PROPHOTO_PRIMARIES = (0.7347, 0.2653, 0.1596, 0.8404, 0.0366, 0.0001)


def xy_to_xyz(white_xy: tuple[float, float]) -> np.ndarray:
    x, y = white_xy
    return np.array([x / y, 1.0, (1.0 - x - y) / y], dtype=np.float64)


def rgb_to_xyz_matrix(
    primaries: tuple[float, float, float, float, float, float],
    white_xy: tuple[float, float],
) -> np.ndarray:
    xr, yr, xg, yg, xb, yb = primaries
    columns = np.column_stack(
        (xy_to_xyz((xr, yr)), xy_to_xyz((xg, yg)), xy_to_xyz((xb, yb)))
    )
    scale = np.linalg.solve(columns, xy_to_xyz(white_xy))
    return columns * scale[np.newaxis, :]


def chromatic_adaptation_bradford(
    source_white_xy: tuple[float, float], destination_white_xy: tuple[float, float]
) -> np.ndarray:
    bradford = np.array(
        (
            (0.8951, 0.2664, -0.1614),
            (-0.7502, 1.7135, 0.0367),
            (0.0389, -0.0685, 1.0296),
        ),
        dtype=np.float64,
    )
    source_lms = bradford @ xy_to_xyz(source_white_xy)
    destination_lms = bradford @ xy_to_xyz(destination_white_xy)
    return np.linalg.inv(bradford) @ np.diag(destination_lms / source_lms) @ bradford


def apply_matrix(rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a conventional column-vector matrix to RGB values on the last axis."""

    return np.asarray(rgb, dtype=np.float64) @ np.asarray(matrix, dtype=np.float64).T



def prophoto_to_xyz_d65(rgb: np.ndarray) -> np.ndarray:
    """Convert linear ProPhoto RGB (D50) to XYZ D65."""
    prophoto_to_xyz_d50 = rgb_to_xyz_matrix(PROPHOTO_PRIMARIES, D50_WHITE_XY)
    xyz_d50 = apply_matrix(rgb, prophoto_to_xyz_d50)
    adaptation = chromatic_adaptation_bradford(D50_WHITE_XY, D65_WHITE_XY)
    return apply_matrix(xyz_d50, adaptation)


def xyz_d65_to_oklab(xyz: np.ndarray) -> np.ndarray:
    """Convert XYZ D65 to OKLab using the frozen project definition."""
    m1 = np.asarray(
        (
            (0.8190224379967030, 0.3619062600528904, -0.1288737815209879),
            (0.0329836539323885, 0.9292868615863434, 0.0361446663506424),
            (0.0481771893596242, 0.2642395317527308, 0.6335478284694309),
        ),
        dtype=np.float64,
    )
    m2 = np.asarray(
        (
            (0.2104542553, 0.7936177850, -0.0040720468),
            (1.9779984951, -2.4285922050, 0.4505937099),
            (0.0259040371, 0.7827717662, -0.8086757660),
        ),
        dtype=np.float64,
    )
    return apply_matrix(np.cbrt(apply_matrix(xyz, m1)), m2)
