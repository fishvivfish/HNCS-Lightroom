from __future__ import annotations

import numpy as np

from hncs_core.adobe_triple_illuminant import temperature_tint_to_xy, xy_to_temperature_tint
from hncs_core.colour import prophoto_to_xyz_d65
from hncs_core.dng_huesat import GridSpec, apply_payload_rgb
from tools.hncs_full_probe import enc_over, dec_over


def test_temperature_tint_roundtrip() -> None:
    for temperature, tint in ((2411.4606, 22.5696), (5550.0, 0.0), (6922.6205, -44.6572), (16689.8298, 7.2105)):
        xy = temperature_tint_to_xy(temperature, tint)
        recovered_temperature, recovered_tint = xy_to_temperature_tint(*xy)
        assert abs(recovered_temperature - temperature) < 0.05
        assert abs(recovered_tint - tint) < 1.0e-3


def test_overrange_roundtrip() -> None:
    values = np.linspace(0.0, 16.0, 257, dtype=np.float64)
    recovered = dec_over(enc_over(values))
    assert np.max(np.abs(recovered - values)) < 2.0e-12


def test_identity_huesatmap() -> None:
    spec = GridSpec(12, 8, 4)
    payload = np.empty((spec.value, spec.hue, spec.saturation, 3), dtype=np.float32)
    payload[...] = (0.0, 1.0, 1.0)
    rng = np.random.default_rng(20260826)
    rgb = rng.random((256, 3))
    output = apply_payload_rgb(spec, payload, rgb)
    assert np.max(np.abs(output - rgb)) < 2.0e-15


def test_prophoto_white_maps_to_d65_white() -> None:
    xyz = prophoto_to_xyz_d65(np.ones((1, 3), dtype=np.float64))[0]
    expected = np.asarray([0.9504559270516716, 1.0, 1.0890577507598784])
    assert np.max(np.abs(xyz - expected)) < 3.0e-4
