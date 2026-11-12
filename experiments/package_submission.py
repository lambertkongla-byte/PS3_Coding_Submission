#!/usr/bin/env python3
"""Build a clean submission archive, then test a fresh extracted copy."""

from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT.parent / "output" / "PS3_Coding_Submission.zip"
REQUIRED = (
    "README.md", "AI_DISCLOSURE.txt", "VALIDATION.md", "requirements.txt",
    "report/PS3_Coding_Report.pdf", "reports/build_report.py",
    "code/pf/PF.py", "code/pf/Laser.py", "code/pf/Gridmap.py",
    "code/pf/Visualization.py", "code/pf/RunPF.py",
    "code/ekf-slam/EKFSLAM.py", "code/ekf-slam/Renderer.py",
    "code/ekf-slam/RunEKFSLAM.py", "results/pf_summary.csv",
)


def main():
    missing = [p for p in REQUIRED if not (ROOT / p).is_file()]
    if missing:
        raise RuntimeError(f"Required deliverables missing: {missing}")
    files = sorted(p for p in ROOT.rglob("*") if p.is_file()
                   and "__pycache__" not in p.parts
                   and not any(part.startswith(".") for part in p.relative_to(ROOT).parts)
                   and p.suffix not in (".pyc", ".pyo")
                   and "smoke" not in str(p.relative_to(ROOT)))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, Path(ROOT.name) / p.relative_to(ROOT))
    env = dict(os.environ)
    env.update(MPLBACKEND="Agg", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    with tempfile.TemporaryDirectory(prefix="ps3_archive_check_") as tmp:
        with zipfile.ZipFile(OUTPUT) as z:
            if z.testzip() is not None:
                raise RuntimeError("Archive integrity check failed")
            z.extractall(tmp)
        base = Path(tmp) / ROOT.name
        commands = [
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            [sys.executable, "code/pf/RunPF.py", "--numParticles", "100", "--seed", "7",
             "--max-steps", "3", "--output-dir", str(Path(tmp) / "pf_smoke"),
             "data/pf/simple_world_path1.pickle"],
            [sys.executable, "code/ekf-slam/RunEKFSLAM.py",
             "data/ekf-slam/ekf-slam-small-noise.pickle",
             "--output-dir", str(Path(tmp) / "ekf_smoke")],
        ]
        for command in commands:
            run = subprocess.run(command, cwd=base, env=env, capture_output=True, text=True)
            if run.returncode:
                raise RuntimeError(f"Extracted-package command failed: {command}\n{run.stdout}\n{run.stderr}")
    print(json.dumps({"archive": str(OUTPUT), "files": len(files),
                      "bytes": OUTPUT.stat().st_size,
                      "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
                      "fresh_extraction_tests": "34 analytic tests + PF CLI + full EKF CLI passed"}, indent=2))


if __name__ == "__main__":
    main()
