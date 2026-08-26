#!/usr/bin/env python3
"""Adobe-line triple-illuminant carrier model used by the final HNCS solver.

The three formal DCP slots are constrained to the original Adobe A<->D65
ColorMatrix/ForwardMatrix line. This supplies three WB interpolation weights
without introducing a third arbitrary camera-colour direction.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hncs_core.adobe_sdk_color import DualColorSpec, TripleColorSpec, adobe_line_slots
from hncs_core.adobe_triple_illuminant import IlluminantDescriptor
from hncs_core.colour import D50_WHITE_XY, D65_WHITE_XY, PROPHOTO_PRIMARIES, rgb_to_xyz_matrix
from hncs_core.dcp_tags import read_dcp_tags
from hncs_core.dng_huesat import GridSpec, apply_payload_rgb

EXPECTED_REFERENCE_DCP_SIZE = 120692
SAMPLE_SEED = 20260826
EXPOSURES = (-3, -2, -1, 0, 1, 2, 3)
ADOBE_BASE_EPSILON = 0.004

# Numerical search bounds, not recovered physical illuminant parameters.
SEARCH_BOUNDS = {
    "temperatures_K": [[1667.0, 5000.0], [3000.0, 8000.0], [6000.0, 25000.0]],
    "tints": [[-200.0, 200.0], [-200.0, 200.0], [-200.0, 200.0]],
    "adobe_line_coefficients": [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
    "classification": "NUMERICAL SEARCH BOUNDS",
}

def _table(tags: dict[str, object], data_name: str) -> np.ndarray:
    dims = tuple(int(item) for item in tags["ProfileHueSatMapDims"])
    return np.asarray(tags[data_name], dtype=np.float32).reshape(np.prod(dims), 3)

class Experiment:
    def __init__(self, dcp_path: Path, sample_count: int = 40000, verify_reference: bool = True) -> None:
        payload = Path(dcp_path).read_bytes()
        if verify_reference and len(payload) != EXPECTED_REFERENCE_DCP_SIZE:
            raise RuntimeError(f"unexpected reference DCP size: {len(payload)}")
        self.dcp_path = Path(dcp_path).resolve()
        self.tags = read_dcp_tags(self.dcp_path)
        required = {
            "ColorMatrix1", "ColorMatrix2", "ForwardMatrix1", "ForwardMatrix2",
            "ProfileHueSatMapData1", "ProfileHueSatMapData2",
            "ProfileLookTableData",
        }
        missing = sorted(required.difference(self.tags))
        if missing:
            raise RuntimeError(f"reference DCP is missing required tags: {missing}")
        self.cm1 = np.asarray(self.tags["ColorMatrix1"], dtype=np.float64).reshape(3, 3)
        self.cm2 = np.asarray(self.tags["ColorMatrix2"], dtype=np.float64).reshape(3, 3)
        self.fm1 = np.asarray(self.tags["ForwardMatrix1"], dtype=np.float64).reshape(3, 3)
        self.fm2 = np.asarray(self.tags["ForwardMatrix2"], dtype=np.float64).reshape(3, 3)
        self.original_hsm = np.stack((
            _table(self.tags, "ProfileHueSatMapData1"),
            _table(self.tags, "ProfileHueSatMapData2"),
        ))
        dims = tuple(int(item) for item in self.tags["ProfileHueSatMapDims"])
        self.base_hsm_spec = GridSpec(*dims)
        look_dims = tuple(int(item) for item in self.tags["ProfileLookTableDims"])
        self.look_spec = GridSpec(*look_dims)
        self.look = np.asarray(self.tags["ProfileLookTableData"], dtype=np.float32).reshape(-1, 3)
        self.prophoto_from_pcs = np.linalg.inv(rgb_to_xyz_matrix(PROPHOTO_PRIMARIES, D50_WHITE_XY))
        self.original = DualColorSpec(
            (0.44757, 0.40745),
            tuple(float(item) for item in D65_WHITE_XY),
            self.cm1, self.cm2, self.fm1, self.fm2,
        )
        rng = np.random.default_rng(SAMPLE_SEED)
        self.unit_samples = rng.random((sample_count, 3), dtype=np.float64)

    def _render(self, state, hsm: np.ndarray, camera_rgb: np.ndarray) -> np.ndarray:
        prophoto = camera_rgb @ state.camera_to_pcs.T @ self.prophoto_from_pcs.T
        prophoto = np.clip(prophoto, 0.0, 1.0)
        prophoto = apply_payload_rgb(self.base_hsm_spec, hsm, prophoto)
        return apply_payload_rgb(self.look_spec, self.look, prophoto)

    @staticmethod
    def unpack(values: np.ndarray) -> tuple[tuple[IlluminantDescriptor, ...], np.ndarray]:
        values = np.asarray(values, dtype=np.float64)
        descriptors = tuple(
            IlluminantDescriptor(float(values[i]), float(values[i + 3]))
            for i in range(3)
        )
        return descriptors, values[6:9]

    def candidate(self, values: np.ndarray):
        descriptors, coefficients = self.unpack(values)
        colors, forwards = adobe_line_slots(self.cm1, self.cm2, self.fm1, self.fm2, coefficients)
        hsm_slots = np.asarray([
            c * self.original_hsm[0] + (1.0 - c) * self.original_hsm[1]
            for c in coefficients
        ], dtype=np.float32)
        return TripleColorSpec(descriptors, colors, forwards), hsm_slots, colors, forwards
