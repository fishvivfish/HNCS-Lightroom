#!/usr/bin/env python3
"""
Descriptor-only sequential VarPro refinement for final HNCS.

This version removes the three Adobe-line coefficients from the outer nonlinear
search.  For every trial descriptor triplet, c1/c2/c3 are re-solved by an
iterated bounded linear projection that matches the original Adobe dual
A<->D65 interpolation coefficient over the current exchange active set.

Then:
- Adobe Base is checked by the SDK-exact rendering path.
- If AdobeBase_worst_P95 > epsilon, the candidate is infeasible.
- If feasible, the three 72x32x32 HSM bases are re-solved analytically under the
  exact 5550-K hard anchor using the current reference carrier exact ideal-HSM cache.
- The outer objective is the worst active-set HNCS OKLab P95 across -3..+3 EV.

IMPORTANT:
The ideal-HSM cache is current-carrier dependent.  Therefore the selected
carrier is PROVISIONAL and must be followed by an exact ideal-cache refresh
before it can become a scientific checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear, minimize


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hncs_core.adobe_sdk_color import TripleColorSpec, adobe_line_slots
from hncs_core.adobe_triple_illuminant import IlluminantDescriptor, temperature_tint_to_xy
from tools.carrier_model import SEARCH_BOUNDS, Experiment
from tools.optimize_hsm_varpro import (
    ADOBE_EPSILON,
    DEFAULT_CACHE,
    DEFAULT_REPORT,
    ORIGINAL_DCP,
    SequentialVarPro,
    _candidate_values,
)


DEFAULT_OUTPUT = ROOT / "work/carrier_optimization"


def descriptor_bounds() -> list[tuple[float, float]]:
    return [
        tuple(x)
        for group in (SEARCH_BOUNDS["temperatures_K"], SEARCH_BOUNDS["tints"])
        for x in group
    ]


class DescriptorProblem:
    def __init__(
        self,
        experiment: Experiment,
        initial_values: np.ndarray,
        temperatures: np.ndarray,
        ideals: np.ndarray,
        base_samples: int,
        epsilon: float,
    ) -> None:
        self.exp = experiment
        self.initial_values = np.asarray(initial_values, dtype=np.float64)
        self.initial_descriptors = self.initial_values[:6].copy()
        self.initial_c = self.initial_values[6:9].copy()
        self.temperatures = np.asarray(temperatures, dtype=np.int32)
        self.ideals = np.asarray(ideals, dtype=np.float32)
        self.base_samples = int(base_samples)
        self.epsilon = float(epsilon)

        self.engine = SequentialVarPro(
            experiment,
            initial_values=self.initial_values,
            temperatures=self.temperatures,
            ideals=self.ideals,
            base_samples=self.base_samples,
        )

        # Original Adobe dual coefficient g(T) on the exact nominal white path.
        self.target_g = np.asarray(
            [
                self.exp.original.set_white_xy(
                    temperature_tint_to_xy(float(t), 0.0)
                ).weights[0]
                for t in self.temperatures
            ],
            dtype=np.float64,
        )

        self.eval_count = 0
        self.best_feasible: dict | None = None
        self.best_infeasible: dict | None = None

    @staticmethod
    def _descriptors(x: np.ndarray) -> tuple[IlluminantDescriptor, ...]:
        x = np.asarray(x, dtype=np.float64)
        return tuple(
            IlluminantDescriptor(float(x[i]), float(x[i + 3]))
            for i in range(3)
        )

    def _weights_for_c(
        self,
        descriptors: tuple[IlluminantDescriptor, ...],
        c: np.ndarray,
    ) -> np.ndarray:
        colors, forwards = adobe_line_slots(
            self.exp.cm1, self.exp.cm2, self.exp.fm1, self.exp.fm2, c
        )
        spec = TripleColorSpec(descriptors, colors, forwards)
        rows = []
        for temperature in self.temperatures:
            white = temperature_tint_to_xy(float(temperature), 0.0)
            original_state = self.exp.original.set_white_xy(white)
            candidate_white = spec.neutral_to_xy(original_state.camera_white)
            candidate_state = spec.set_white_xy(candidate_white)
            rows.append(candidate_state.weights)
        return np.asarray(rows, dtype=np.float64)

    def project_coefficients(
        self,
        descriptor_values: np.ndarray,
        c_start: np.ndarray | None = None,
        max_iter: int = 12,
    ) -> tuple[np.ndarray, dict]:
        """
        Iterated coefficient projection.

        With candidate triple weights W held fixed, the effective Adobe-line
        coordinate is W @ c.  Solve bounded LS to the original dual coefficient
        g(T), rebuild the SDK-exact triple spec, recompute W, and repeat.
        """
        descriptors = self._descriptors(descriptor_values)
        c = self.initial_c.copy() if c_start is None else np.asarray(c_start, dtype=np.float64).copy()

        history = []
        for iteration in range(max_iter):
            W = self._weights_for_c(descriptors, c)
            solution = lsq_linear(
                W,
                self.target_g,
                bounds=(0.0, 1.0),
                lsq_solver="exact",
                tol=1.0e-12,
                max_iter=100,
            )
            new_c = np.asarray(solution.x, dtype=np.float64)
            delta = float(np.max(np.abs(new_c - c)))
            residual = W @ new_c - self.target_g
            history.append(
                {
                    "iteration": iteration,
                    "c": new_c.tolist(),
                    "max_abs_weight_residual": float(np.max(np.abs(residual))),
                    "rms_weight_residual": float(np.sqrt(np.mean(residual * residual))),
                    "delta_c_max": delta,
                }
            )
            c = new_c
            if delta < 1.0e-9:
                break

        return c, {
            "iterations": len(history),
            "history": history,
            "final_max_abs_weight_residual": history[-1]["max_abs_weight_residual"],
            "final_rms_weight_residual": history[-1]["rms_weight_residual"],
        }

    def evaluate(self, descriptor_values: np.ndarray, full_base: bool = False) -> dict:
        c, projection = self.project_coefficients(descriptor_values)
        values = np.concatenate(
            [np.asarray(descriptor_values, dtype=np.float64), c]
        )
        result = self.engine.evaluate(values, full_base=full_base)
        result["coefficient_projection"] = projection
        return result

    def objective(self, descriptor_values: np.ndarray) -> float:
        self.eval_count += 1
        try:
            result = self.evaluate(descriptor_values, full_base=False)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return 2.0

        base = float(result["adobe_base_worst_p95"])
        feasible = base <= self.epsilon

        if not feasible:
            score = 1.0 + base
            if (
                self.best_infeasible is None
                or base < self.best_infeasible["adobe_base_worst_p95"]
            ):
                self.best_infeasible = result
        else:
            score = float(result["hncs"]["max_p95"])
            if (
                self.best_feasible is None
                or score < self.best_feasible["hncs"]["max_p95"]
            ):
                self.best_feasible = result

        if self.eval_count % 5 == 0:
            print(
                json.dumps(
                    {
                        "eval": self.eval_count,
                        "score": score,
                        "base": base,
                        "feasible": feasible,
                        "hncs_max": (
                            result.get("hncs", {}).get("max_p95") if feasible else None
                        ),
                        "descriptor_T": np.asarray(descriptor_values)[:3].tolist(),
                        "c": result["values"][6:9],
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
    parser.add_argument("--epsilon", type=float, default=ADOBE_EPSILON)
    parser.add_argument("--base-samples", type=int, default=4096)
    parser.add_argument("--maxiter", type=int, default=30)
    parser.add_argument(
        "--seed-first-temperature",
        type=float,
        default=2500.0,
        help=(
            "DEBUG-ONLY local initializer. 2500 K is not a scientific constant: "
            "a direct coefficient-projection probe around the reference carrier found this "
            "region feasible for AdobeBase<=0.004."
        ),
    )
    args = parser.parse_args()

    cache = np.load(args.cache)
    temperatures = np.asarray(cache["temperatures_K"], dtype=np.int32)
    ideals = np.asarray(cache["ideals"], dtype=np.float32)
    cached_values = np.asarray(cache["candidate_values"], dtype=np.float64)
    initial_values = _candidate_values(args.candidate_report)
    if not np.array_equal(cached_values, initial_values):
        raise RuntimeError("active ideal cache does not belong to the configured reference carrier")

    exp = Experiment(ORIGINAL_DCP, sample_count=40000)
    problem = DescriptorProblem(
        exp,
        initial_values=initial_values,
        temperatures=temperatures,
        ideals=ideals,
        base_samples=args.base_samples,
        epsilon=args.epsilon,
    )

    x0 = initial_values[:6].copy()
    x0[0] = float(args.seed_first_temperature)

    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    seed_eval = problem.evaluate(x0, full_base=True)
    print(
        json.dumps(
            {
                "seed": {
                    "descriptors": x0.tolist(),
                    "c": seed_eval["values"][6:9],
                    "adobe_base_worst_p95": seed_eval["adobe_base_worst_p95"],
                    "feasible": seed_eval["adobe_base_worst_p95"] <= args.epsilon,
                    "hncs": seed_eval.get("hncs"),
                },
                "epsilon": args.epsilon,
            },
            indent=2,
        ),
        flush=True,
    )

    opt = minimize(
        problem.objective,
        x0,
        method="Powell",
        bounds=descriptor_bounds(),
        options={
            "maxiter": int(args.maxiter),
            "xtol": 2.0e-5,
            "ftol": 2.0e-6,
            "disp": True,
        },
    )

    selected = problem.evaluate(np.asarray(opt.x, dtype=np.float64), full_base=True)
    values = np.asarray(selected["values"], dtype=np.float64)
    descriptors, coefficients = exp.unpack(values)

    result = {
        "classification": (
            "PROVISIONAL DESCRIPTOR-VARPRO STEP — coefficients projected, "
            "reference carrier exact ideal preimages frozen; refresh required"
        ),
        "epsilon": float(args.epsilon),
        "active_temperatures_K": temperatures.tolist(),
        "algorithm_settings": {
            "classification": "DEBUG-ONLY / numerical settings",
            "outer_method": "Powell on 6 descriptor variables",
            "maxiter": int(args.maxiter),
            "base_sketch_samples": int(args.base_samples),
            "descriptor_bounds": descriptor_bounds(),
            "coefficient_projection": (
                "iterated bounded LS of W(D,c)@c to original Adobe dual coefficient g(T)"
            ),
            "inner_hsm_projection": (
                "equal-active-temperature LS in exact 5550-K nullspace"
            ),
            "outer_objective": (
                "min active-set worst HNCS OKLab P95 subject to AdobeBase worst P95 <= epsilon"
            ),
        },
        "seed_first_temperature_K": float(args.seed_first_temperature),
        "seed_full_evaluation": {
            k: v for k, v in seed_eval.items() if k != "_bases"
        },
        "optimizer": {
            "success": bool(opt.success),
            "message": str(opt.message),
            "fun": float(opt.fun),
            "nfev": int(opt.nfev),
            "nit": int(getattr(opt, "nit", -1)),
        },
        "selected": {
            "values": values.tolist(),
            "descriptors": [
                {
                    "temperature_K": float(d.temperature),
                    "tint": float(d.tint),
                    "white_xy": [float(q) for q in d.white_xy],
                }
                for d in descriptors
            ],
            "coefficients": coefficients.tolist(),
            "full_evaluation": {
                k: v for k, v in selected.items() if k != "_bases"
            },
        },
        "refresh_required": True,
        "runtime_seconds": float(time.time() - started),
    }

    (args.output / "descriptor_varpro_step.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if "_bases" in selected:
        np.savez_compressed(
            args.output / "provisional_bases.npz",
            bases=selected["_bases"],
            temperatures_K=temperatures,
            values=values,
        )

    print(
        json.dumps(
            {
                "selected_descriptors": result["selected"]["descriptors"],
                "selected_coefficients": result["selected"]["coefficients"],
                "selected_base_worst_p95": selected["adobe_base_worst_p95"],
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
