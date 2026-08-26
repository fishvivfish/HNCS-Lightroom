#!/usr/bin/env python3
"""Exact 1-K audit of the WB-dependent HNCS final profile pipeline.

Scientific definition
---------------------
* Ground truth is the recovered Phocus 4.0.1 WB-dependent HNCS model at the
  current integer Kelvin, not the fixed 5550 K/01 profile.
* final profile is evaluated with the actual stored triple-illuminant matrices,
  illuminant chromaticities, and 72x32x32 HSM payloads read back from the DCP.
* The fixed downstream 5550 K HNCS stage and DNG 1.7 HDR/overrange path are the
  same persisted multi-exposure definition used by the 72x32x32 checkpoint.
* Every integer Kelvin in [2400, 10000] is evaluated at EV -3..+3 by default.
* Adobe-base preservation is audited independently using the original Adobe
  HSM trajectory implied by the recovered Adobe-line coefficients; the HNCS
  HSM payload is intentionally NOT used for this base-preservation metric.

No scientific parameter is silently inferred. reference carrier carrier coefficients
come from its persisted build report; the stored DCP itself supplies the final
matrices, illuminants, and HSM payloads.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hncs_core.adobe_sdk_color import TripleColorSpec
from hncs_core.adobe_triple_illuminant import IlluminantDescriptor, temperature_tint_to_xy, xy_to_temperature_tint
from hncs_core.colour import prophoto_to_xyz_d65, xyz_d65_to_oklab
from tools.carrier_model import Experiment, EXPOSURES
from tools.hncs_full_probe import (
    FullProbe,
    HSM_SPEC,
    apply_map_hdr,
    test_grid,
)


EXPECTED_CANDIDATE_HSM_DIMS = (72, 32, 32)
EXPECTED_P = 0.53
DEFAULT_START = 2400
DEFAULT_END = 10000
DEFAULT_STEP = 1
DEFAULT_TEST_SEED = 260237


def _rat_matrix(value: object) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float64).reshape(-1, 2)
    return (raw[:, 0] / raw[:, 1]).reshape(3, 3)


def _decode_illuminant_data(value: object) -> tuple[int, tuple[float, float]]:
    raw = bytes(value)
    kind, nx, dx, ny, dy = struct.unpack("<Hiiii", raw)
    if dx == 0 or dy == 0:
        raise ValueError("invalid IlluminantData denominator")
    return int(kind), (float(nx / dx), float(ny / dy))


def _stats(delta: np.ndarray) -> dict[str, float]:
    x = np.asarray(delta, dtype=np.float64)
    return {
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def _oklab_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = xyz_d65_to_oklab(prophoto_to_xyz_d65(np.asarray(a, dtype=np.float64)))
    bb = xyz_d65_to_oklab(prophoto_to_xyz_d65(np.asarray(b, dtype=np.float64)))
    return np.linalg.norm(aa - bb, axis=1)

# Auxiliary reporting metric only. The scientific optimization objective remains
# the frozen OKLab Euclidean distance used throughout the project.
_D65_XYZ = np.asarray([0.95047, 1.0, 1.08883], dtype=np.float64)

def _xyz_d65_to_lab(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64) / _D65_XYZ[None, :]
    delta = 6.0 / 29.0
    d3 = delta ** 3
    f = np.where(xyz > d3, np.cbrt(xyz), xyz / (3.0 * delta * delta) + 4.0 / 29.0)
    L = 116.0 * f[:, 1] - 16.0
    a = 500.0 * (f[:, 0] - f[:, 1])
    b = 200.0 * (f[:, 1] - f[:, 2])
    return np.column_stack((L, a, b))

def _ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    # Sharma, Wu & Dalal (2005), kL=kC=kH=1. Vectorized, degrees internally.
    lab1 = np.asarray(lab1, dtype=np.float64); lab2 = np.asarray(lab2, dtype=np.float64)
    L1,a1,b1 = lab1.T; L2,a2,b2 = lab2.T
    C1 = np.hypot(a1,b1); C2 = np.hypot(a2,b2); Cbar = 0.5*(C1+C2)
    G = 0.5*(1.0 - np.sqrt((Cbar**7)/(Cbar**7 + 25.0**7)))
    ap1=(1.0+G)*a1; ap2=(1.0+G)*a2
    Cp1=np.hypot(ap1,b1); Cp2=np.hypot(ap2,b2)
    hp1=(np.degrees(np.arctan2(b1,ap1))+360.0)%360.0
    hp2=(np.degrees(np.arctan2(b2,ap2))+360.0)%360.0
    hp1=np.where((Cp1==0.0),0.0,hp1); hp2=np.where((Cp2==0.0),0.0,hp2)
    dLp=L2-L1; dCp=Cp2-Cp1
    dh=hp2-hp1
    dhp=np.where((Cp1*Cp2)==0.0,0.0,dh)
    dhp=np.where(((Cp1*Cp2)!=0.0)&(dh>180.0),dh-360.0,dhp)
    dhp=np.where(((Cp1*Cp2)!=0.0)&(dh<-180.0),dh+360.0,dhp)
    dHp=2.0*np.sqrt(Cp1*Cp2)*np.sin(np.radians(dhp/2.0))
    Lbar=0.5*(L1+L2); Cpbar=0.5*(Cp1+Cp2)
    hsum=hp1+hp2; hdiff=np.abs(hp1-hp2)
    hpbar=np.where((Cp1*Cp2)==0.0,hsum,0.5*hsum)
    hpbar=np.where(((Cp1*Cp2)!=0.0)&(hdiff>180.0)&(hsum<360.0),0.5*(hsum+360.0),hpbar)
    hpbar=np.where(((Cp1*Cp2)!=0.0)&(hdiff>180.0)&(hsum>=360.0),0.5*(hsum-360.0),hpbar)
    T=(1.0 - 0.17*np.cos(np.radians(hpbar-30.0)) + 0.24*np.cos(np.radians(2.0*hpbar))
       + 0.32*np.cos(np.radians(3.0*hpbar+6.0)) - 0.20*np.cos(np.radians(4.0*hpbar-63.0)))
    dtheta=30.0*np.exp(-((hpbar-275.0)/25.0)**2)
    RC=2.0*np.sqrt((Cpbar**7)/(Cpbar**7 + 25.0**7))
    SL=1.0 + (0.015*(Lbar-50.0)**2)/np.sqrt(20.0+(Lbar-50.0)**2)
    SC=1.0+0.045*Cpbar; SH=1.0+0.015*Cpbar*T
    RT=-np.sin(np.radians(2.0*dtheta))*RC
    return np.sqrt((dLp/SL)**2 + (dCp/SC)**2 + (dHp/SH)**2 + RT*(dCp/SC)*(dHp/SH))

def _de00_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    la=_xyz_d65_to_lab(prophoto_to_xyz_d65(np.asarray(a,dtype=np.float64)))
    lb=_xyz_d65_to_lab(prophoto_to_xyz_d65(np.asarray(b,dtype=np.float64)))
    return _ciede2000(la,lb)


def _read_candidate_dcp(path: Path) -> tuple[TripleColorSpec, np.ndarray, dict[str, object]]:
    payload = path.read_bytes()
    if payload[:4] != b"IIRC":
        raise RuntimeError(f"candidate DCP magic mismatch: {payload[:4]!r}")

    with tifffile.TiffFile(path) as tf:
        tags = tf.pages[0].tags
        dims = tuple(int(x) for x in tags[50937].value)
        if dims != EXPECTED_CANDIDATE_HSM_DIMS:
            raise RuntimeError(f"candidate HSM dims mismatch: {dims}")

        colors = np.stack([_rat_matrix(tags[code].value) for code in (50721, 50722, 52531)])
        forwards = np.stack([_rat_matrix(tags[code].value) for code in (50964, 50965, 52532)])

        decoded = [_decode_illuminant_data(tags[code].value) for code in (52533, 52534, 52535)]
        if [kind for kind, _ in decoded] != [0, 0, 0]:
            raise RuntimeError(f"candidate IlluminantData types are not chromaticity: {decoded}")
        stored_xy = [xy for _, xy in decoded]
        if len({(round(x, 10), round(y, 10)) for x, y in stored_xy}) != 3:
            raise RuntimeError("candidate does not contain three distinct illuminant chromaticities")

        descriptors = tuple(
            IlluminantDescriptor(*xy_to_temperature_tint(float(x), float(y)))
            for x, y in stored_xy
        )
        spec = TripleColorSpec(descriptors, colors, forwards)

        bases = np.stack(
            [
                np.asarray(tags[code].value, dtype=np.float32).reshape(
                    dims[2], dims[0], dims[1], 3
                )
                for code in (50938, 50939, 52537)
            ]
        )

    info = {
        "path": str(path.resolve()),
        "bytes": len(payload),        "hsm_dims": list(dims),
        "stored_xy": [list(xy) for xy in stored_xy],
        "stored_descriptors": [
            {"temperature_K": float(d.temperature), "tint": float(d.tint)} for d in descriptors
        ],
    }
    return spec, bases, info


def _read_coefficients(report_path: Path) -> np.ndarray:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    dims = tuple(report["numerical_checkpoint"]["hsm_dims"])
    if dims != EXPECTED_CANDIDATE_HSM_DIMS:
        raise RuntimeError(f"build report HSM dims mismatch: {dims}")
    p = float(report["numerical_checkpoint"]["p"])
    if not math.isclose(p, EXPECTED_P, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(f"build report p mismatch: {p}")
    coefficients = np.asarray(report["carrier"]["coefficients"], dtype=np.float64)
    if coefficients.shape != (3,):
        raise RuntimeError(f"invalid Adobe-line coefficient shape: {coefficients.shape}")
    return coefficients


def _candidate_state(probe: FullProbe, temperature: int):
    original_white = temperature_tint_to_xy(float(temperature), 0.0)
    original_state = probe.exp.original.set_white_xy(original_white)
    candidate_white = probe.spec.neutral_to_xy(original_state.camera_white)
    candidate_state = probe.spec.set_white_xy(candidate_white)
    return original_white, original_state, candidate_white, candidate_state


def _evaluate_hncs_temperature(
    probe: FullProbe,
    bases: np.ndarray,
    samples: np.ndarray,
    temperature: int,
    exposures: tuple[int, ...],
) -> tuple[list[dict[str, float | int]], np.ndarray, dict[str, object]]:
    _, _, candidate_white, candidate_state = _candidate_state(probe, temperature)
    hsm_weights = np.asarray(candidate_state.weights, dtype=np.float32)
    payload = np.tensordot(hsm_weights, bases, axes=1).astype(np.float32)

    # Compute the pre-exposure branches once. Then batch all seven EV states through
    # the unchanged downstream and exact Phocus target for this Kelvin.
    candidate_pre = apply_map_hdr(HSM_SPEC, payload, np.clip(samples, 0.0, 1.0))
    original_pre = probe.original_post_hsm(samples, temperature)

    candidate_stack = np.vstack([candidate_pre * (2.0 ** ev) for ev in exposures])
    target_input_stack = np.vstack(
        [probe.adobe_look_extended(np.maximum(original_pre, 0.0) * (2.0 ** ev)) for ev in exposures]
    )
    candidate_stack = probe.downstream(candidate_stack)
    target_stack = probe.projector.target(target_input_stack, int(temperature))

    n = len(samples)
    rows: list[dict[str, float | int]] = []
    p95 = np.empty(len(exposures), dtype=np.float64)
    for index, ev in enumerate(exposures):
        sl = slice(index * n, (index + 1) * n)
        delta = _oklab_error(candidate_stack[sl], target_stack[sl])
        de00 = _de00_error(candidate_stack[sl], target_stack[sl])
        s = _stats(delta); d = _stats(de00)
        p95[index] = s["p95"]
        rows.append(
            {
                "temperature_K": int(temperature),
                "EV": int(ev),
                "median": s["median"],
                "p95": s["p95"],
                "max": s["max"],
                "de00_median": d["median"],
                "de00_p95": d["p95"],
                "de00_max": d["max"],
            }
        )

    diag = {
        "candidate_white_xy": [float(candidate_white[0]), float(candidate_white[1])],
        "hsm_weights": [float(x) for x in hsm_weights],
    }
    return rows, p95, diag


def _evaluate_adobe_base_temperature(
    probe: FullProbe,
    coefficients: np.ndarray,
    temperature: int,
    sample_count: int,
) -> dict[str, float | list[float]]:
    white = temperature_tint_to_xy(float(temperature), 0.0)
    original_state = probe.exp.original.set_white_xy(white)
    original_weights = probe.exp.original.weights(white)
    original_hsm = (
        original_weights[0] * probe.exp.original_hsm[0]
        + original_weights[1] * probe.exp.original_hsm[1]
    ).astype(np.float32)

    camera_rgb = probe.exp.unit_samples[:sample_count] * original_state.camera_white[None, :]
    reference = probe.exp._render(original_state, original_hsm, camera_rgb)

    candidate_white = probe.spec.neutral_to_xy(original_state.camera_white)
    candidate_state = probe.spec.set_white_xy(candidate_white)
    adobe_hsm_slots = np.asarray(
        [
            c * probe.exp.original_hsm[0] + (1.0 - c) * probe.exp.original_hsm[1]
            for c in coefficients
        ],
        dtype=np.float32,
    )
    candidate_hsm = np.tensordot(
        np.asarray(candidate_state.weights, dtype=np.float32), adobe_hsm_slots, axes=1
    ).astype(np.float32)
    output = probe.exp._render(candidate_state, candidate_hsm, camera_rgb)
    delta = _oklab_error(output, reference)
    s = _stats(delta)
    return {
        **s,
        "white_xy_error_max": float(np.max(np.abs(np.asarray(candidate_white) - np.asarray(white)))),
        "camera_white_error_max": float(
            np.max(np.abs(candidate_state.camera_white - original_state.camera_white))
        ),
        "camera_to_pcs_error_max": float(
            np.max(np.abs(candidate_state.camera_to_pcs - original_state.camera_to_pcs))
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-dcp",
        type=Path,
        default=ROOT / "local_assets" / "Sony ILCE-7RM5 Adobe Standard.dcp",
    )
    parser.add_argument(
        "--candidate-dcp",
        type=Path,
        default=ROOT / "build" / "sony-ilce-7rm5" / "HNCS Base - ILCE-7RM5.dcp",
    )
    parser.add_argument(
        "--candidate-report",
        type=Path,
        default=ROOT / "build" / "sony-ilce-7rm5" / "build_report.json",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "full_audit")
    parser.add_argument("--start", type=int, default=DEFAULT_START)
    parser.add_argument("--end", type=int, default=DEFAULT_END)
    parser.add_argument("--step", type=int, default=DEFAULT_STEP)
    parser.add_argument("--test-seed", type=int, default=DEFAULT_TEST_SEED)
    parser.add_argument(
        "--base-samples",
        type=int,
        default=40000,
        help="Adobe-base RGB samples per Kelvin; default reproduces the 40k definition.",
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip the independent Adobe-base 40k audit; HNCS audit remains exact.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N Kelvin points.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.step <= 0 or args.end < args.start:
        raise SystemExit("invalid temperature range")

    temperatures = np.arange(args.start, args.end + 1, args.step, dtype=np.int32)
    exposures = tuple(int(x) for x in EXPOSURES)
    if exposures != (-3, -2, -1, 0, 1, 2, 3):
        raise RuntimeError(f"frozen exposure definition changed unexpectedly: {exposures}")

    args.output.mkdir(parents=True, exist_ok=True)
    candidate_spec, bases, candidate_info = _read_candidate_dcp(args.candidate_dcp)
    coefficients = _read_coefficients(args.candidate_report)

    exp = Experiment(args.original_dcp, sample_count=max(40000, int(args.base_samples)))

    # Build the unchanged downstream once. FullProbe reconstructs the persisted 72x32x32
    # HDR/multi-EV definition. Replace only its carrier spec with the exact DCP round-trip spec.
    report = json.loads(args.candidate_report.read_text(encoding="utf-8"))
    requested = report["carrier"]["descriptors_stored_roundtrip"]
    values = np.asarray(
        [x["temperature_K"] for x in requested]
        + [x["tint"] for x in requested]
        + coefficients.tolist(),
        dtype=np.float64,
    )
    probe = FullProbe(exp, values)
    probe.spec = candidate_spec

    samples = test_grid(seed=args.test_seed)
    n_t = len(temperatures)
    n_e = len(exposures)
    p95 = np.empty((n_t, n_e), dtype=np.float64)
    median = np.empty((n_t, n_e), dtype=np.float64)
    maximum = np.empty((n_t, n_e), dtype=np.float64)
    de00_median = np.empty((n_t, n_e), dtype=np.float64)
    de00_p95 = np.empty((n_t, n_e), dtype=np.float64)
    de00_max = np.empty((n_t, n_e), dtype=np.float64)
    hsm_weights = np.empty((n_t, 3), dtype=np.float64)
    candidate_white_xy = np.empty((n_t, 2), dtype=np.float64)
    base_p95 = np.full(n_t, np.nan, dtype=np.float64)
    base_median = np.full(n_t, np.nan, dtype=np.float64)
    base_max = np.full(n_t, np.nan, dtype=np.float64)

    csv_path = args.output / "per_state.csv"
    started = time.time()
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["temperature_K", "EV", "oklab_median", "oklab_p95", "oklab_max", "de00_median", "de00_p95", "de00_max"],
        )
        writer.writeheader()

        for ti, temperature in enumerate(temperatures):
            rows, temp_p95, diag = _evaluate_hncs_temperature(
                probe, bases, samples, int(temperature), exposures
            )
            p95[ti] = temp_p95
            median[ti] = [float(row["median"]) for row in rows]
            maximum[ti] = [float(row["max"]) for row in rows]
            de00_median[ti] = [float(row["de00_median"]) for row in rows]
            de00_p95[ti] = [float(row["de00_p95"]) for row in rows]
            de00_max[ti] = [float(row["de00_max"]) for row in rows]
            hsm_weights[ti] = np.asarray(diag["hsm_weights"], dtype=np.float64)
            candidate_white_xy[ti] = np.asarray(diag["candidate_white_xy"], dtype=np.float64)

            for row in rows:
                writer.writerow(
                    {
                        "temperature_K": row["temperature_K"],
                        "EV": row["EV"],
                        "oklab_median": f"{float(row['median']):.12g}",
                        "oklab_p95": f"{float(row['p95']):.12g}",
                        "oklab_max": f"{float(row['max']):.12g}",
                        "de00_median": f"{float(row['de00_median']):.12g}",
                        "de00_p95": f"{float(row['de00_p95']):.12g}",
                        "de00_max": f"{float(row['de00_max']):.12g}",
                    }
                )

            if not args.skip_base:
                base = _evaluate_adobe_base_temperature(
                    probe, coefficients, int(temperature), int(args.base_samples)
                )
                base_median[ti] = float(base["median"])
                base_p95[ti] = float(base["p95"])
                base_max[ti] = float(base["max"])

            if (ti + 1) % max(1, args.progress_every) == 0 or ti == 0 or ti + 1 == n_t:
                elapsed = time.time() - started
                rate = (ti + 1) / max(elapsed, 1.0e-12)
                remaining = (n_t - ti - 1) / max(rate, 1.0e-12)
                print(
                    f"[{ti+1:5d}/{n_t}] T={int(temperature)}K "
                    f"HNCS-worst={float(np.max(p95[ti])):.6f} "
                    + (
                        f"AdobeBase-P95={base_p95[ti]:.6f} "
                        if not args.skip_base
                        else ""
                    )
                    + f"elapsed={elapsed:.1f}s ETA={remaining:.1f}s",
                    flush=True,
                )
                handle.flush()

    worst_flat = int(np.nanargmax(p95))
    worst_ti, worst_ei = np.unravel_index(worst_flat, p95.shape)
    temp_worst = np.max(p95, axis=1)
    temp_mean = np.mean(p95, axis=1)

    summary: dict[str, object] = {
        "classification": "HNCS COLOR EXACT 1-K OFFLINE AUDIT — Phocus WB-dependent HNCS ground truth",
        "original_adobe_dcp": {
            "path": str(args.original_dcp.resolve()),        },
        "candidate": candidate_info,
        "scientific_definition": {
            "ground_truth": "Recovered Phocus 4.0.1 WB-dependent HNCS at each integer Kelvin; 01/5550 is anchor only, not full-range ground truth",
            "temperature_start_K": int(args.start),
            "temperature_end_K": int(args.end),
            "temperature_step_K": int(args.step),
            "temperature_count": int(n_t),
            "EV_stops": list(exposures),
            "hsm_dims": list(EXPECTED_CANDIDATE_HSM_DIMS),
            "p": EXPECTED_P,
            "test_grid_seed": int(args.test_seed),
            "test_grid_samples": int(len(samples)),
            "adobe_base_samples_per_K": None if args.skip_base else int(args.base_samples),
        },
        "carrier": {
            "adobe_line_coefficients": coefficients.tolist(),
        },
        "hncs": {
            "mean_of_state_p95": float(np.mean(p95)),
            "median_of_state_p95": float(np.median(p95)),
            "max_state_p95": float(p95[worst_ti, worst_ei]),
            "worst_temperature_K": int(temperatures[worst_ti]),
            "worst_EV": int(exposures[worst_ei]),
            "mean_of_per_temperature_worst_p95": float(np.mean(temp_worst)),
            "max_per_temperature_mean_p95": float(np.max(temp_mean)),
        },
        "ciede2000_reporting_only": {
            "mean_of_state_p95": float(np.mean(de00_p95)),
            "median_of_state_p95": float(np.median(de00_p95)),
            "max_state_p95": float(np.max(de00_p95)),
            "worst_temperature_K": int(temperatures[np.unravel_index(int(np.argmax(de00_p95)), de00_p95.shape)[0]]),
            "worst_EV": int(exposures[np.unravel_index(int(np.argmax(de00_p95)), de00_p95.shape)[1]]),
            "note": "Auxiliary familiar display-style metric; not used as the optimization objective."
        },
        "adobe_base": None,
        "runtime_seconds": float(time.time() - started),
        "artifacts": {
            "per_state_csv": str(csv_path.resolve()),
            "arrays_npz": str((args.output / "audit_arrays.npz").resolve()),
        },
    }

    if not args.skip_base:
        bi = int(np.nanargmax(base_p95))
        summary["adobe_base"] = {
            "mean_p95_over_K": float(np.nanmean(base_p95)),
            "worst_p95": float(base_p95[bi]),
            "worst_temperature_K": int(temperatures[bi]),
            "constraint_0p004_pass": bool(float(base_p95[bi]) <= 0.004),
        }

    np.savez_compressed(
        args.output / "audit_arrays.npz",
        temperatures_K=temperatures,
        EV_stops=np.asarray(exposures, dtype=np.int32),
        oklab_median=median,
        oklab_p95=p95,
        oklab_max=maximum,
        de00_median=de00_median,
        de00_p95=de00_p95,
        de00_max=de00_max,
        per_temperature_mean_p95=temp_mean,
        per_temperature_worst_p95=temp_worst,
        hsm_weights=hsm_weights,
        candidate_white_xy=candidate_white_xy,
        adobe_base_median=base_median,
        adobe_base_p95=base_p95,
        adobe_base_max=base_max,
        adobe_line_coefficients=coefficients,
    )
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
