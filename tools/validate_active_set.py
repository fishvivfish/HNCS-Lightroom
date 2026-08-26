#!/usr/bin/env python3
"""Fast exact validation on the frozen final exchange workset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_profile import (
    _read_candidate_dcp, _read_coefficients,
    _evaluate_hncs_temperature, _evaluate_adobe_base_temperature,
)
from tools.carrier_model import Experiment, EXPOSURES
from tools.hncs_full_probe import FullProbe, test_grid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-dcp', type=Path, default=ROOT/'local_assets'/'Sony ILCE-7RM5 Adobe Standard.dcp')
    parser.add_argument('--candidate-dcp', type=Path, default=ROOT/'build'/'sony-ilce-7rm5'/'HNCS Base - ILCE-7RM5.dcp')
    parser.add_argument('--candidate-report', type=Path, default=ROOT/'build'/'sony-ilce-7rm5'/'build_report.json')
    parser.add_argument('--active-set', type=Path, default=ROOT/'config'/'final_active_set.json')
    parser.add_argument('--output', type=Path, default=ROOT/'work'/'active_roundtrip_validation.json')
    parser.add_argument('--base-samples', type=int, default=40000)
    args = parser.parse_args()

    temps = np.asarray(json.loads(args.active_set.read_text(encoding='utf-8'))['active_temperatures_K'], dtype=np.int32)
    spec, bases, info = _read_candidate_dcp(args.candidate_dcp)
    coefficients = _read_coefficients(args.candidate_report)
    exp = Experiment(args.original_dcp, sample_count=max(40000, int(args.base_samples)))

    report = json.loads(args.candidate_report.read_text(encoding='utf-8'))
    requested = report['carrier']['descriptors_stored_roundtrip']
    values = np.asarray(
        [x['temperature_K'] for x in requested] + [x['tint'] for x in requested] + coefficients.tolist(),
        dtype=np.float64,
    )
    probe = FullProbe(exp, values)
    probe.spec = spec
    samples = test_grid(seed=260237)
    exposures = tuple(map(int, EXPOSURES))

    rows = []
    for i, temperature in enumerate(temps, start=1):
        _, p95, _ = _evaluate_hncs_temperature(probe, bases, samples, int(temperature), exposures)
        base = _evaluate_adobe_base_temperature(probe, coefficients, int(temperature), int(args.base_samples))
        row = {
            'temperature_K': int(temperature),
            'hncs_mean_state_p95': float(np.mean(p95)),
            'hncs_worst_p95': float(np.max(p95)),
            'worst_EV': int(exposures[int(np.argmax(p95))]),
            'adobe_base_p95': float(base['p95']),
        }
        rows.append(row)
        print(f"{i}/{len(temps)} {int(temperature)}K {row}", flush=True)

    worst = max(rows, key=lambda r: r['hncs_worst_p95'])
    base_worst = max(rows, key=lambda r: r['adobe_base_p95'])
    result = {
        'classification': 'HNCS Color DCP round-trip active-set exact validation',
        'candidate': info,
        'rows': rows,
        'hncs_mean_of_temperature_mean_state_p95': float(np.mean([r['hncs_mean_state_p95'] for r in rows])),
        'hncs_max_p95': worst['hncs_worst_p95'],
        'hncs_worst_temperature_K': worst['temperature_K'],
        'hncs_worst_EV': worst['worst_EV'],
        'adobe_base_worst_p95': base_worst['adobe_base_p95'],
        'adobe_base_worst_temperature_K': base_worst['temperature_K'],
        'adobe_0p004_pass': bool(base_worst['adobe_base_p95'] <= 0.004),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
