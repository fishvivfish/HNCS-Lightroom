#!/usr/bin/env python3
"""
One sequential variable-projection refinement step for the HNCS Adobe-line carrier.

This is intentionally a LOCAL / SEQUENTIAL step:

1. Exact 72x32x32 ideal HSMs are loaded from the current reference carrier active-set cache.
2. For each trial carrier, the three HSM bases are re-solved analytically
   (variable projection) under the exact 5550-K hard anchor.
3. Adobe Base is handled as an epsilon constraint, NOT as a weighted-sum term.
4. The outer objective is the worst active-set HNCS OKLab P95 across -3..+3 EV.
5. Because the exact ideal-HSM preimage is weakly carrier-dependent, any material
   carrier move is PROVISIONAL.  Its exact ideal cache must be rebuilt and the
   step repeated before it can be accepted as a new scientific checkpoint.

No frozen HNCS parameter is changed here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hncs_core.adobe_triple_illuminant import temperature_tint_to_xy
from hncs_core.colour import prophoto_to_xyz_d65, xyz_d65_to_oklab
from tools.carrier_model import (
    SEARCH_BOUNDS,
    Experiment,
)
from tools.hncs_full_probe import (
    H,
    S,
    V,
    HSM_SPEC,
    MAX_USED_V,
    STOPS,
    apply_map_hdr,
    met,
    FullProbe,
    test_grid,
)


DEFAULT_CACHE = ROOT / "work/ideal_cache" / "active_ideal_payloads.npz"
DEFAULT_REPORT = ROOT / "config" / "sony_ilce_7rm5.json"
DEFAULT_OUTPUT = ROOT / "work/hsm_varpro"
ORIGINAL_DCP = ROOT / "local_assets" / "Sony ILCE-7RM5 Adobe Standard.dcp"

HARD_ANCHOR_K = 5550
ADOBE_EPSILON = 0.004
TEST_GRID_SEED = 260237


def _candidate_values(report_path: Path) -> np.ndarray:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    descriptors = report["carrier"]["descriptors_stored_roundtrip"]
    coefficients = report["carrier"]["coefficients"]
    return np.asarray(
        [item["temperature_K"] for item in descriptors]
        + [item["tint"] for item in descriptors]
        + coefficients,
        dtype=np.float64,
    )


def _bounds() -> list[tuple[float, float]]:
    return [
        tuple(x)
        for group in (
            SEARCH_BOUNDS["temperatures_K"],
            SEARCH_BOUNDS["tints"],
            SEARCH_BOUNDS["adobe_line_coefficients"],
        )
        for x in group
    ]


def _oklab_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = xyz_d65_to_oklab(prophoto_to_xyz_d65(np.asarray(a, dtype=np.float64)))
    bb = xyz_d65_to_oklab(prophoto_to_xyz_d65(np.asarray(b, dtype=np.float64)))
    return np.linalg.norm(aa - bb, axis=1)


def _candidate_weights(exp: Experiment, spec, temperatures: np.ndarray) -> np.ndarray:
    rows = []
    for temperature in temperatures:
        white = temperature_tint_to_xy(float(temperature), 0.0)
        original_state = exp.original.set_white_xy(white)
        candidate_white = spec.neutral_to_xy(original_state.camera_white)
        candidate_state = spec.set_white_xy(candidate_white)
        rows.append(candidate_state.weights)
    return np.asarray(rows, dtype=np.float64)


def solve_bases_hard_anchor_active(
    weights: np.ndarray,
    ideals: np.ndarray,
    temperatures: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Equal-temperature least squares in the exact nullspace of the 5550-K row.

    Equal active-set weighting is an ALGORITHM choice for this sequential
    variable-projection step, not a recovered historical scientific parameter.
    The outer minimax objective is still the exact worst active-state P95.
    """
    temperatures = np.asarray(temperatures, dtype=np.int32)
    matches = np.flatnonzero(temperatures == HARD_ANCHOR_K)
    if len(matches) != 1:
        raise RuntimeError("active set must contain exactly one 5550-K anchor")
    ai = int(matches[0])

    Y = np.asarray(ideals, dtype=np.float64).reshape(len(temperatures), -1, 3).copy()
    raw_anchor_hue = Y[ai, :, 0].copy()

    # Choose a continuous hue branch along the ordered temperature axis.
    Y[:, :, 0] = np.rad2deg(np.unwrap(np.deg2rad(Y[:, :, 0]), axis=0))
    branch = np.rint((Y[ai, :, 0] - raw_anchor_hue) / 360.0) * 360.0
    Y[:, :, 0] -= branch[None, :]

    wa = weights[ai : ai + 1]
    B0 = np.repeat(Y[ai][None, :, :], 3, axis=0)

    # Two-dimensional nullspace = the only residual freedom left after fixing
    # the 5550-K affine anchor.
    _, _, vh = np.linalg.svd(wa, full_matrices=True)
    N = vh[1:].T  # 3 x 2

    free = [i for i in range(len(temperatures)) if i != ai]
    X = weights[free] @ N
    R = np.stack([Y[i] - Y[ai] for i in free], axis=0)

    C = np.linalg.pinv(X, rcond=1.0e-12) @ R.reshape(len(free), -1)
    delta = (N @ C).reshape(3, *Y.shape[1:])

    # Preserve legal saturation/value scale range without post-fit clipping,
    # which would break the hard anchor.
    for comp in (1, 2):
        base = B0[:, :, comp]
        d = delta[:, :, comp]
        alpha = np.ones(base.shape[1], dtype=np.float64)
        for k in range(3):
            neg = d[k] < 0.0
            if np.any(neg):
                alpha[neg] = np.minimum(
                    alpha[neg],
                    np.divide(
                        base[k, neg],
                        -d[k, neg],
                        out=np.ones(np.sum(neg), dtype=np.float64),
                        where=(-d[k, neg]) > 0.0,
                    ),
                )
            pos = d[k] > 0.0
            if np.any(pos):
                alpha[pos] = np.minimum(
                    alpha[pos],
                    np.divide(
                        256.0 - base[k, pos],
                        d[k, pos],
                        out=np.ones(np.sum(pos), dtype=np.float64),
                        where=d[k, pos] > 0.0,
                    ),
                )
        alpha = np.clip(alpha, 0.0, 1.0)
        delta[:, :, comp] *= alpha[None, :]

    B = (B0 + delta).reshape(3, V, H, S, 3).astype(np.float32)

    # Fixed legal cells.  All three bases receive the same fixed value so the
    # anchor is preserved.
    for k in range(3):
        B[k, 0, :, :] = (0.0, 1.0, 1.0)
        B[k, : MAX_USED_V + 1, :, 0, 0] = 0.0
        B[k, : MAX_USED_V + 1, :, 0, 1] = 1.0
        for vv in range(MAX_USED_V + 1, V):
            B[k, vv] = B[k, MAX_USED_V]

    target = ideals[ai].astype(np.float32)
    anchor = np.tensordot(weights[ai].astype(np.float32), B, axes=1)
    anchor_error = float(np.max(np.abs(anchor - target)))
    return B, anchor_error


class SequentialVarPro:
    def __init__(
        self,
        exp: Experiment,
        initial_values: np.ndarray,
        temperatures: np.ndarray,
        ideals: np.ndarray,
        base_samples: int,
    ) -> None:
        self.exp = exp
        self.initial_values = np.asarray(initial_values, dtype=np.float64)
        self.temperatures = np.asarray(temperatures, dtype=np.int32)
        self.ideals = np.asarray(ideals, dtype=np.float32)
        self.base_samples = int(base_samples)
        self.test = test_grid(TEST_GRID_SEED)

        # Build expensive downstream tables only once.  We update probe.spec
        # per candidate; the downstream tables themselves do not depend on the
        # triple carrier descriptors/c_i.
        self.probe = FullProbe(exp, self.initial_values)

        self.eval_count = 0
        self.best_feasible: dict | None = None
        self.best_infeasible: dict | None = None

    def _generic_adobe_base(self, spec, coefficients: np.ndarray, count: int) -> tuple[float, dict[int, float]]:
        hsm_slots = np.asarray(
            [
                c * self.exp.original_hsm[0] + (1.0 - c) * self.exp.original_hsm[1]
                for c in coefficients
            ],
            dtype=np.float32,
        )
        per: dict[int, float] = {}
        n = min(int(count), len(self.exp.unit_samples))

        for temperature in self.temperatures:
            white = temperature_tint_to_xy(float(temperature), 0.0)
            original_state = self.exp.original.set_white_xy(white)
            original_weights = self.exp.original.weights(white)
            original_hsm = np.tensordot(
                original_weights.astype(np.float32),
                self.exp.original_hsm,
                axes=1,
            ).astype(np.float32)

            camera_rgb = self.exp.unit_samples[:n] * original_state.camera_white[None, :]
            reference = self.exp._render(original_state, original_hsm, camera_rgb)

            candidate_white = spec.neutral_to_xy(original_state.camera_white)
            candidate_state = spec.set_white_xy(candidate_white)
            candidate_hsm = np.tensordot(
                candidate_state.weights.astype(np.float32),
                hsm_slots,
                axes=1,
            ).astype(np.float32)
            output = self.exp._render(candidate_state, candidate_hsm, camera_rgb)
            delta = _oklab_error(output, reference)
            per[int(temperature)] = float(np.percentile(delta, 95))

        return max(per.values()), per

    def _hncs_active(self, spec, bases: np.ndarray, weights: np.ndarray) -> dict:
        # The exact target path depends on the current carrier through the
        # camera-matrix coordinate mapping.  Reuse the expensive downstream
        # tables but update the carrier spec for every trial candidate.
        self.probe.spec = spec

        rows = []
        values = []
        for i, temperature in enumerate(self.temperatures):
            payload = np.tensordot(
                weights[i].astype(np.float32), bases, axes=1
            ).astype(np.float32)
            for ev in STOPS:
                m = met(
                    self.probe.candidate_output(self.test, int(temperature), payload, int(ev)),
                    self.probe.original_target(self.test, int(temperature), int(ev)),
                )
                p95 = float(m["oklab"]["p95"])
                rows.append(
                    {
                        "temperature_K": int(temperature),
                        "EV": int(ev),
                        "p95": p95,
                    }
                )
                values.append(p95)

        worst = int(np.argmax(values))
        return {
            "mean_p95": float(np.mean(values)),
            "max_p95": float(np.max(values)),
            "worst_temperature_K": rows[worst]["temperature_K"],
            "worst_EV": rows[worst]["EV"],
            "per_state": rows,
        }

    def evaluate(self, values: np.ndarray, full_base: bool = False) -> dict:
        values = np.asarray(values, dtype=np.float64)
        spec, _, _, _ = self.exp.candidate(values)
        _, coefficients = self.exp.unpack(values)

        base_worst, base_per = self._generic_adobe_base(
            spec,
            coefficients,
            count=(len(self.exp.unit_samples) if full_base else self.base_samples),
        )

        result = {
            "values": values.tolist(),
            "adobe_base_worst_p95": float(base_worst),
            "adobe_base_p95": {str(k): float(v) for k, v in base_per.items()},
            "feasible": bool(base_worst <= ADOBE_EPSILON),
        }

        # Epsilon constraint: an infeasible candidate is not allowed to trade
        # Adobe Base damage against lower HNCS error.
        if not result["feasible"]:
            return result

        weights = _candidate_weights(self.exp, spec, self.temperatures)
        bases, anchor_error = solve_bases_hard_anchor_active(
            weights, self.ideals, self.temperatures
        )
        hncs = self._hncs_active(spec, bases, weights)
        result["weights"] = weights.tolist()
        result["5550_anchor_max_abs_payload_error"] = anchor_error
        result["hncs"] = hncs
        result["_bases"] = bases
        return result

    def objective(self, values: np.ndarray) -> float:
        self.eval_count += 1
        try:
            result = self.evaluate(values, full_base=False)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return 2.0

        if not result["feasible"]:
            # Strict feasibility-first barrier.  This is epsilon-constraint,
            # not a weighted sum.  All infeasible objectives are >1.
            score = 1.0 + float(result["adobe_base_worst_p95"])
            if (
                self.best_infeasible is None
                or result["adobe_base_worst_p95"]
                < self.best_infeasible["adobe_base_worst_p95"]
            ):
                self.best_infeasible = result
        else:
            score = float(result["hncs"]["max_p95"])
            if (
                self.best_feasible is None
                or result["hncs"]["max_p95"] < self.best_feasible["hncs"]["max_p95"]
            ):
                self.best_feasible = result

        if self.eval_count % 10 == 0:
            print(
                json.dumps(
                    {
                        "eval": self.eval_count,
                        "score": score,
                        "base": result["adobe_base_worst_p95"],
                        "feasible": result["feasible"],
                        "hncs_max": (
                            result.get("hncs", {}).get("max_p95")
                            if result["feasible"]
                            else None
                        ),
                    }
                ),
                flush=True,
            )
        return score


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--candidate-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--base-samples",
        type=int,
        default=4096,
        help=(
            "Optimization-only Adobe Base sketch size. Final selected candidate is "
            "revalidated on all 40k samples. This is an algorithmic speed setting."
        ),
    )
    parser.add_argument("--method", choices=("powell", "de"), default="powell")
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    cache = np.load(args.cache)
    temperatures = np.asarray(cache["temperatures_K"], dtype=np.int32)
    ideals = np.asarray(cache["ideals"], dtype=np.float32)
    cached_values = np.asarray(cache["candidate_values"], dtype=np.float64)
    initial_values = _candidate_values(args.candidate_report)

    if not np.array_equal(cached_values, initial_values):
        raise RuntimeError(
            "ideal cache candidate-values do not match reference carrier report; "
            "refresh the cache rather than silently mixing carriers"
        )

    args.output.mkdir(parents=True, exist_ok=True)

    exp = Experiment(ORIGINAL_DCP, sample_count=40000)
    problem = SequentialVarPro(
        exp,
        initial_values=initial_values,
        temperatures=temperatures,
        ideals=ideals,
        base_samples=args.base_samples,
    )

    bounds = _bounds()
    started = time.time()

    initial_eval = problem.evaluate(initial_values, full_base=True)
    print(
        json.dumps(
            {
                "initial": {
                    "adobe_base_worst_p95": initial_eval["adobe_base_worst_p95"],
                    "feasible": initial_eval["feasible"],
                    "hncs": initial_eval.get("hncs"),
                },
                "epsilon": ADOBE_EPSILON,
                "active_temperatures_K": temperatures.tolist(),
            },
            indent=2,
        ),
        flush=True,
    )

    if args.method == "powell":
        opt = minimize(
            problem.objective,
            initial_values,
            method="Powell",
            bounds=bounds,
            options={
                "maxiter": int(args.maxiter),
                "xtol": 2.0e-5,
                "ftol": 2.0e-6,
                "disp": True,
            },
        )
    else:
        opt = differential_evolution(
            problem.objective,
            bounds=bounds,
            seed=int(args.seed),
            popsize=6,
            maxiter=int(args.maxiter),
            polish=False,
            workers=1,
            updating="immediate",
        )

    selected_values = np.asarray(opt.x, dtype=np.float64)
    selected = problem.evaluate(selected_values, full_base=True)

    descriptors, coefficients = exp.unpack(selected_values)
    move = selected_values - initial_values

    result = {
        "classification": (
            "PROVISIONAL SEQUENTIAL VARPRO STEP — frozen reference carrier exact ideal-HSM "
            "preimages; MUST refresh ideals after this carrier move before acceptance"
        ),
        "scientific_constraints": {
            "adobe_base_epsilon": ADOBE_EPSILON,
            "5550_hard_anchor_K": HARD_ANCHOR_K,
            "hsm_dims": [H, S, V],
            "EV_stops": [int(x) for x in STOPS],
            "active_temperatures_K": temperatures.tolist(),
        },
        "algorithm_settings": {
            "classification": "DEBUG-ONLY / numerical optimizer settings",
            "method": args.method,
            "maxiter": int(args.maxiter),
            "seed": int(args.seed),
            "base_sketch_samples": int(args.base_samples),
            "bounds": SEARCH_BOUNDS,
            "inner_basis_fit": "equal active-temperature least squares in 5550-row nullspace",
            "outer_objective": "min active-set worst HNCS OKLab P95 subject to AdobeBase worst P95 <= 0.004",
        },
        "initial_values": initial_values.tolist(),
        "initial_full_evaluation": {
            k: v for k, v in initial_eval.items() if k != "_bases"
        },
        "optimizer": {
            "success": bool(opt.success),
            "message": str(opt.message),
            "fun": float(opt.fun),
            "nfev": int(opt.nfev),
            "nit": int(getattr(opt, "nit", -1)),
        },
        "selected": {
            "values": selected_values.tolist(),
            "descriptors": [
                {
                    "temperature_K": float(d.temperature),
                    "tint": float(d.tint),
                    "white_xy": [float(x) for x in d.white_xy],
                }
                for d in descriptors
            ],
            "coefficients": coefficients.tolist(),
            "full_evaluation": {
                k: v for k, v in selected.items() if k not in ("_bases",)
            },
        },
        "move_from_reference_carrier": {
            "delta_values": move.tolist(),
            "l2": float(np.linalg.norm(move)),
            "max_abs": float(np.max(np.abs(move))),
        },
        "refresh_required": True,
        "acceptance_rule": (
            "Do not call this a new scientific checkpoint. Rebuild exact active ideal HSMs "
            "at the selected carrier, re-run this step, then perform a full 2400..10000 K "
            "1-K exact audit before accepting the final profile."
        ),
        "runtime_seconds": float(time.time() - started),
    }

    (args.output / "sequential_varpro_step.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if "_bases" in selected:
        np.savez_compressed(
            args.output / "provisional_bases.npz",
            bases=selected["_bases"],
            temperatures_K=temperatures,
            values=selected_values,
        )

    print(
        json.dumps(
            {
                "selected_base_worst_p95": selected["adobe_base_worst_p95"],
                "selected_feasible": selected["feasible"],
                "selected_hncs": selected.get("hncs"),
                "refresh_required": True,
                "runtime_seconds": result["runtime_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
