"""Command-line runner; the assignment's original invocation remains valid."""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Particle Filter-based Localization")
    parser.add_argument("pickle_file", help="Path to the supplied pickle file")
    parser.add_argument("--numParticles", type=int, default=1000, help="Number of particles")
    parser.add_argument("--sparsity", type=int, default=5, help="Keep every Mth LIDAR beam")
    parser.add_argument("-a", "--animate", action="store_true", help="Animate while filtering")
    parser.add_argument("--seed", type=int, default=0, help="Reproducible random seed (default 0)")
    parser.add_argument("--resampling", choices=("always", "never", "adaptive"), default="always")
    parser.add_argument("--neff-threshold", type=float, default=2.0 / 3.0,
                        help="Adaptive ESS threshold as a fraction of N (default 2/3)")
    parser.add_argument("--init", choices=("auto", "gaussian", "uniform"), default="auto",
                        help="auto: Gaussian xy if X0 exists, otherwise uniform free space")
    parser.add_argument("--output-dir", default="results", help="Directory for PNG, NPZ and JSON outputs")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Optional prefix length for a smoke test; omitted uses all data")
    args = parser.parse_args()
    if args.numParticles <= 0 or args.sparsity <= 0:
        parser.error("--numParticles and --sparsity must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if not 0 < args.neff_threshold <= 1:
        parser.error("--neff-threshold must be in (0,1]")
    if not args.animate:
        matplotlib.use("Agg")

    from PF import PF
    from Laser import Laser
    from Gridmap import Gridmap

    # Pickle is used only for the course-provided, trusted data files.
    with open(args.pickle_file, "rb") as stream:
        data = pickle.load(stream, encoding="latin1")
    all_ranges = np.asarray(data["Ranges"])
    U = np.asarray(data["U"])
    steps = U.shape[1] if args.max_steps is None else min(args.max_steps, U.shape[1])
    U = U[:, :steps]
    ranges = all_ranges[::args.sparsity, :steps]
    X0, XGT = data.get("X0"), data.get("XGT")
    if XGT is not None:
        XGT = np.asarray(XGT)[:, :steps + 1]
    if args.init == "gaussian" and X0 is None:
        parser.error("--init gaussian requires X0, absent from this scenario")
    gridmap = Gridmap(data["occupancy"])
    laser = Laser(all_ranges.shape[0], args.sparsity)
    pf = PF(args.numParticles, data["Alpha"], laser, gridmap, args.animate,
            seed=args.seed, resampling=args.resampling,
            neff_threshold=args.neff_threshold, init=args.init)
    initialization = ("gaussian" if X0 is not None else "uniform") if args.init == "auto" else args.init
    stem = (f"{Path(args.pickle_file).stem}_N{args.numParticles}_{args.resampling}"
            f"_s{args.sparsity}_seed{args.seed}_{initialization}")
    if steps != np.asarray(data["U"]).shape[1]:
        stem += f"_steps{steps}"
    prefix = Path(args.output_dir) / stem
    metrics = pf.run(U, ranges, float(data["deltat"]), X0, XGT, str(prefix))
    print(json.dumps({"outputs": pf.output_paths, "metrics": metrics}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
