"""Phocus 4.0.1 X2D 100C Standard WB-dependent ColorCorrect model."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .color_correct import as_samples, restore_samples, apply_color_correct


@dataclass(frozen=True)
class Phocus401WBModel:
    """Callable reproduction of the recovered Kelvin-indexed colour stage."""

    metadata: dict
    lut_temperatures: np.ndarray
    standard_luts: np.ndarray
    matrix_temperatures: np.ndarray
    camera_matrices: np.ndarray
    rgb_to_ycc: np.ndarray
    hasselblad_rgb_inverse: np.ndarray
    output_matrix: np.ndarray

    @classmethod
    def load(cls, directory: str | Path) -> "Phocus401WBModel":
        root = Path(directory)
        metadata = json.loads((root / "wb_model.json").read_text(encoding="utf-8"))
        with np.load(root / "records.npz") as records:
            return cls(
                metadata=metadata,
                lut_temperatures=records["lut_temperatures"].astype(np.float64),
                standard_luts=records["standard_luts"].astype(np.float64),
                matrix_temperatures=records["matrix_temperatures"].astype(np.float64),
                camera_matrices=records["camera_matrices"].astype(np.float64),
                rgb_to_ycc=records["rgb_to_ycc"].astype(np.float64),
                hasselblad_rgb_inverse=records["hasselblad_rgb_inverse"].astype(np.float64),
                output_matrix=records["output_matrix"].astype(np.float64),
            )

    @staticmethod
    def _piecewise(values: np.ndarray, anchors: np.ndarray, kelvin: int) -> np.ndarray:
        upper = int(np.searchsorted(anchors, kelvin, side="right"))
        upper = min(max(upper, 1), len(anchors) - 1)
        lower = upper - 1
        alpha = (kelvin - anchors[lower]) / (anchors[upper] - anchors[lower])
        return values[lower] + alpha * (values[upper] - values[lower])

    def parameters_at_kelvin(self, kelvin: float) -> tuple[dict, np.ndarray]:
        """Generate the exact matrix/LUT parameterization selected by Phocus."""
        temperature = int(kelvin)  # cvttsd2si: truncate toward zero
        lut_temperature = min(max(temperature, 2000), 10000)
        matrix_temperature = min(max(temperature, 2000), 8000)

        texture = self._piecewise(
            self.standard_luts, self.lut_temperatures, lut_temperature
        ).astype(np.float32).astype(np.float64)
        camera_matrix = self._piecewise(
            self.camera_matrices, self.matrix_temperatures, matrix_temperature
        )
        input_matrix = self.rgb_to_ycc @ self.hasselblad_rgb_inverse @ camera_matrix

        fixed = self.metadata["fixed_color_correct"]
        normalization_maximum = float(fixed["normalization_maximum"])
        dark_x = 15500.0 / normalization_maximum
        dark_a = -0.123 / (dark_x * dark_x)
        dark_b = -2.0 * dark_a * dark_x
        dark_c = 1.0 + dark_a * dark_x * dark_x
        params = {
            "start_cbcr": fixed["start_cbcr"],
            "cbcr_limits": fixed["cbcr_limits"],
            "div_factor": fixed["div_factor"],
            "input_matrix": input_matrix,
            "output_matrix": self.output_matrix,
            "desat_gray": fixed["desat_gray"],
            "dark_params": [dark_x, dark_a, dark_b, dark_c],
            "apply_ev": True,
            "apply_cc": True,
        }
        return params, texture

    def color_correct(self, rgb: Any, kelvin: float) -> np.ndarray:
        """Evaluate only the recovered ColorCorrect stage for ``(..., 3)`` RGB."""
        samples, shape = as_samples(rgb)
        params, texture = self.parameters_at_kelvin(kelvin)
        return restore_samples(apply_color_correct(samples, params, texture), shape)
