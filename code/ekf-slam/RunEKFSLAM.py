"""Run PS3 EKF SLAM: python RunEKFSLAM.py file.pickle [--output-dir DIR]."""

import argparse
import json
from pathlib import Path
import pickle

import matplotlib
import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="One of the supplied EKF-SLAM pickle files.")
    parser.add_argument("--output-dir", type=Path,
                        help="Save figures, metrics.json, and results.npz here.")
    parser.add_argument("--animate", action="store_true", help="Show live rendering (default: headless).")
    args = parser.parse_args(argv)
    if not args.animate:
        matplotlib.use("Agg")
    from EKFSLAM import EKFSLAM
    with args.data.open("rb") as stream:
        data = pickle.load(stream)
    # The problem explicitly supplies X0. XGT is reserved for evaluation/plots.
    mu0 = np.asarray(data["X0"], dtype=float).reshape(3)
    Sigma0 = 0.01 * np.eye(3)
    Sigma0[2, 2] = np.deg2rad(0.5) ** 2
    slam = EKFSLAM(mu0, Sigma0, data["R"], data["Q"],
                   data.get("XGT"), data.get("MGT"), animate=args.animate)
    slam.run(data["U"], data["Z"])
    destination = args.output_dir or Path("results") / args.data.stem
    metrics = slam.save_results(destination, args.data.stem)
    print(json.dumps(metrics, indent=2))
    if args.animate:
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
