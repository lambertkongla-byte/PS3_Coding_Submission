#!/usr/bin/env python3
"""Reproduce PF experiments listed in a JSON manifest.

Run from any working directory. Every experiment is an independent process,
with one seed and one output directory. No ground truth is passed to the
filter other than the original driver uses for evaluation/plotting.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def run_one(spec: dict, overwrite: bool = False) -> dict:
    out = ROOT / "results" / "pf" / spec["run_id"]
    out.mkdir(parents=True, exist_ok=True)
    receipt_path = out / "execution.json"
    source_paths = [ROOT / "code" / "pf" / name for name in ("PF.py", "Laser.py", "Gridmap.py", "RunPF.py")]
    source_hash = hashlib.sha256(b"".join(p.read_bytes() for p in source_paths)).hexdigest()
    if receipt_path.exists() and not overwrite:
        old = json.loads(receipt_path.read_text())
        if old.get("returncode") == 0 and old.get("spec") == spec and old.get("source_sha256") == source_hash:
            return {**old, "cached": True}

    command = [
        sys.executable, str(ROOT / "code" / "pf" / "RunPF.py"),
        str(ROOT / "data" / "pf" / (spec["dataset"] + ".pickle")),
        "--numParticles", str(spec["particles"]),
        "--sparsity", str(spec.get("sparsity", 5)),
        "--seed", str(spec.get("seed", 7)),
        "--resampling", spec.get("resampling", "always"),
        "--neff-threshold", str(spec.get("neff_threshold", 2 / 3)),
        "--init", spec.get("initialization", "auto"),
        "--output-dir", str(out),
    ]
    env = dict(os.environ)
    env.update({"MPLBACKEND": "Agg", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    start = time.perf_counter()
    with (out / "stdout.txt").open("w") as stdout, (out / "stderr.txt").open("w") as stderr:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=False)
    receipt = {
        "spec": spec,
        "command": command,
        "returncode": result.returncode,
        "wall_seconds": time.perf_counter() - start,
        "source_sha256": source_hash,
        "output_dir": str(out.relative_to(ROOT)),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON array of experiment specifications")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true", help="Re-run completed matching experiments")
    args = parser.parse_args()
    specs = json.loads(args.manifest.read_text())
    if not isinstance(specs, list) or args.jobs < 1:
        parser.error("Manifest must be a list and --jobs must be positive")
    run_ids = [s["run_id"] for s in specs]
    if len(run_ids) != len(set(run_ids)):
        parser.error("Duplicate run IDs in manifest")
    failures = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_one, spec, args.overwrite): spec for spec in specs}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            spec = result["spec"]
            label = "cached" if result.get("cached") else ("ok" if result["returncode"] == 0 else "FAILED")
            print(f"[{i:02d}/{len(specs):02d}] {spec['run_id']}: {label}, {result['wall_seconds']:.2f} s", flush=True)
            if result["returncode"]:
                failures.append(spec["run_id"])
    print(f"Completed in {time.perf_counter()-start:.1f} s. Failures: {failures}", flush=True)
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
