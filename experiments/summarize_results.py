#!/usr/bin/env python3
"""Validate saved PF outputs and write one portable CSV row per completed run."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def check(condition, message):
    if not condition:
        raise ValueError(message)


def summarize(expected_runs=None, output=None):
    receipts = sorted((ROOT / "results" / "pf").glob("*/execution.json"))
    if expected_runs is not None:
        check(len(receipts) == expected_runs,
              f"Expected {expected_runs} execution receipts, found {len(receipts)}")
    manifests = ["particle_sweep.json", "resampling_comparison.json",
                 "additional_counts.json", "robustness_counts.json"]
    requested = set()
    for name in manifests:
        path = ROOT / "experiments" / name
        if path.exists():
            requested.update(item["run_id"] for item in json.loads(path.read_text()))
    completed = {path.parent.name for path in receipts}
    check(requested <= completed, f"Missing requested runs: {sorted(requested - completed)}")
    sources = [ROOT / "code" / "pf" / name for name in ("PF.py", "Laser.py", "Gridmap.py", "RunPF.py")]
    source_hash = hashlib.sha256(b"".join(path.read_bytes() for path in sources)).hexdigest()
    rows = []

    for receipt_path in receipts:
        receipt = json.loads(receipt_path.read_text())
        run_id = receipt["spec"]["run_id"]
        check(receipt["returncode"] == 0, f"{run_id}: process failed")
        check(receipt["source_sha256"] == source_hash, f"{run_id}: inference source hash differs")
        metric_paths = [p for p in receipt_path.parent.glob("*.json") if p.name != "execution.json"]
        check(len(metric_paths) == 1, f"{run_id}: expected exactly one metrics JSON")
        metric_path = metric_paths[0]
        png_path, npz_path = metric_path.with_suffix(".png"), metric_path.with_suffix(".npz")
        check(png_path.is_file() and png_path.stat().st_size > 0, f"{run_id}: missing/empty PNG")
        check(npz_path.is_file(), f"{run_id}: missing NPZ")
        metrics = json.loads(metric_path.read_text())
        N, T = metrics["num_particles"], metrics["num_steps"]
        with np.load(npz_path, allow_pickle=False) as arrays:
            check(json.loads(arrays["metadata_json"].item()) == metrics,
                  f"{run_id}: NPZ and JSON metadata disagree")
            for name in ("final_particles", "posterior_particles"):
                check(arrays[name].shape == (3, N), f"{run_id}: bad {name} shape")
                check(np.all(np.isfinite(arrays[name])), f"{run_id}: nonfinite {name}")
            for name in ("final_weights", "posterior_weights"):
                values = arrays[name]
                check(values.shape == (N,), f"{run_id}: bad {name} shape")
                check(np.all(np.isfinite(values)) and np.all(values >= 0) and np.isclose(values.sum(), 1),
                      f"{run_id}: invalid {name}")
            means, weights = arrays["means"], arrays["posterior_weights"]
            check(means.shape == (3, T + 1) and np.all(np.isfinite(means)), f"{run_id}: bad mean history")
            check(np.array_equal(means, arrays["estimated_poses"]), f"{run_id}: mean aliases disagree")
            check(np.allclose(arrays["posterior_particles"][:2] @ weights, means[:2, -1]),
                  f"{run_id}: final mean is not the weighted posterior mean")
            neff, resampled = arrays["neff_before_resampling"], arrays["resampled"]
            check(neff.shape == (T,) and resampled.shape == (T,), f"{run_id}: bad ESS/resampling history")
            check(np.all((neff >= 1 - 1e-8) & (neff <= N + 1e-6)), f"{run_id}: ESS outside [1,N]")
            check(np.isclose(neff[-1], 1 / np.dot(weights, weights)), f"{run_id}: final ESS mismatch")
            strategy = metrics["resampling"]
            expected_resampling = (np.ones(T, dtype=bool) if strategy == "always" else
                                   np.zeros(T, dtype=bool) if strategy == "never" else
                                   neff < metrics["neff_threshold_fraction"] * N)
            check(np.array_equal(resampled, expected_resampling), f"{run_id}: resampling strategy mismatch")
            check(int(resampled.sum()) == metrics["resampling_count"], f"{run_id}: resampling count mismatch")
            if resampled[-1]:
                check(np.allclose(arrays["final_weights"], 1 / N), f"{run_id}: resampled weights not uniform")
            else:
                check(np.array_equal(arrays["final_weights"], weights), f"{run_id}: unexpected final weight change")
                check(np.array_equal(arrays["final_particles"], arrays["posterior_particles"]),
                      f"{run_id}: unexpected final particle change")
            if metrics["ground_truth_available"]:
                truth = arrays["XGT"]
                check(truth.shape == (3, T + 1), f"{run_id}: wrong ground-truth shape")
                errors = np.linalg.norm(means[:2, 1:] - truth[:2, 1:], axis=0)
                check(np.allclose(errors, arrays["position_errors"]), f"{run_id}: time alignment/error mismatch")
                check(np.isclose(np.sqrt(np.mean(errors ** 2)), metrics["position_rmse"]),
                      f"{run_id}: position RMSE mismatch")
                check(np.isclose(errors[-1], metrics["final_position_error"]), f"{run_id}: final error mismatch")
                check(np.isclose(np.sqrt(np.mean(errors[int(np.floor(.8 * T)):] ** 2)),
                                 metrics["tail20_position_rmse"]), f"{run_id}: tail RMSE mismatch")
            else:
                check(arrays["XGT"].shape == (3, 0), f"{run_id}: unexpected ground truth")
                check("position_rmse" not in metrics, f"{run_id}: accuracy claimed without ground truth")

        row = {"run_id": run_id, "dataset": receipt["spec"]["dataset"],
               "purpose": receipt["spec"].get("purpose", "")}
        for key in ("num_particles", "resampling", "seed", "sparsity", "initialization",
                    "num_steps", "ground_truth_available", "position_rmse", "final_position_error",
                    "tail20_position_rmse", "heading_rmse_rad", "final_heading_error_rad",
                    "final_position_spread_rms", "mean_neff", "final_neff_before_resampling",
                    "resampling_count", "impossible_scans_skipped", "collision_fallbacks",
                    "inference_seconds", "total_seconds"):
            row[key] = metrics.get(key, "")
        row.update(wall_seconds=receipt["wall_seconds"],
                   metrics_path=str(metric_path.relative_to(ROOT)),
                   npz_path=str(npz_path.relative_to(ROOT)), png_path=str(png_path.relative_to(ROOT)),
                   source_sha256=receipt["source_sha256"])
        rows.append(row)

    rows.sort(key=lambda row: (row["dataset"], row["resampling"], row["num_particles"], row["seed"]))
    output = ROOT / "results" / "pf_summary.csv" if output is None else Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    check(bool(rows), "No completed PF runs found")
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"validated_pf_runs": len(rows), "requested_manifest_runs": len(requested),
               "datasets": len({row["dataset"] for row in rows}), "output": str(output),
               "source_sha256": source_hash, "validation": "passed"}
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-runs", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    summarize(args.expected_runs, args.output)


if __name__ == "__main__":
    main()
