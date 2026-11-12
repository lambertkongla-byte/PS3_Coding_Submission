# PS3 coding submission: particle-filter localization and EKF SLAM

This package covers **Question 2(a–f)** and **Question 3(b)** of TTIC 31170 Problem Set 3. It includes the completed Python modules, the supplied data, reproducible experiment specifications, numerical outputs, figures, and analytic tests.

Read **[report/PS3_Coding_Report.pdf](report/PS3_Coding_Report.pdf)** for the experiment results and discussion. The separate non-coding solutions cover Question 1 and Question 3(a); those derivations are outside this coding package.

## Setup

Run the commands below from this package's root directory. Python 3.12 is recommended.

```bash
python3 -m pip install -r requirements.txt
```

The implementation was tested with **Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, and Matplotlib 3.10.8**. `requirements.txt` gives compatible dependency ranges. The normal filter runs use a headless Matplotlib backend and do not require a display. Interactive animation is optional.

## Package contents

| Location | Contents |
|---|---|
| `code/pf/` | Particle filter, beam sensor model, map utilities, plotting, and `RunPF.py`. |
| `code/ekf-slam/` | EKF prediction, update, landmark augmentation, plotting, and `RunEKFSLAM.py`. |
| `data/pf/` | All eight supplied particle-filter datasets and their original README. |
| `data/ekf-slam/` | Both supplied EKF-SLAM datasets and their original README. |
| `experiments/` | Particle-count selections, experiment manifests, and the batch runner. |
| `results/pf/` | Per-experiment particle distributions, arrays, metrics, and execution records. |
| `results/ekf/` | Final maps, trajectory comparisons, error plots, arrays, and metrics for both EKF datasets. |
| `tests/` | Independent analytic and synthetic checks using `unittest`. |
| `report/PS3_Coding_Report.pdf` | Written experimental discussion and figures. |
| `reports/build_report.py` | Source for generating the report. |

## Run one particle-filter scenario

The original assignment interface is supported. For example:

```bash
python3 code/pf/RunPF.py --numParticles 1000 --sparsity 5 --seed 7 --resampling always --output-dir results/pf/manual_simple_world_path1 data/pf/simple_world_path1.pickle
```

The required options remain `--numParticles N`, `--sparsity M`, and `-a` for animation. Additional options make the experiments reproducible:

| Option | Meaning |
|---|---|
| `--seed 7` | Set the random seed; the command-line default is `0`. The main report experiments use seed `7` unless their manifest says otherwise. |
| `--resampling always` | Resample after every observation update; this is the default. |
| `--resampling never` | Preserve recursive importance weights without resampling. |
| `--resampling adaptive` | Resample only when effective sample size is below the selected fraction of the particle count. |
| `--neff-threshold 0.6666666666666666` | Adaptive resampling threshold as a fraction of `N`; default is `2/3`. |
| `--init auto` | Use the known-start initializer when `X0` exists; otherwise use uniform initialization over free map space. |
| `--init gaussian` / `--init uniform` | Explicitly choose the initializer. Gaussian initialization requires `X0`. |
| `--output-dir DIR` | Save the final-distribution PNG, arrays NPZ, and metrics JSON in `DIR`. |
| `--max-steps K` | Run a prefix for a smoke test. Omit this option for the full assignment experiments. |
| `-a` or `--animate` | Display live filtering using an interactive Matplotlib backend. |

`--sparsity 5` retains every fifth LIDAR beam. The known-start initializer uses a Gaussian position distribution centered at `X0` with standard deviation `0.5` m in each coordinate, truncated to free space, and a uniform heading distribution. This retains the supplied initialization convention; the initial heading is not fixed to ground truth. Uniform initialization samples both free-space position and heading. Ground truth is used for evaluation and plotting, not to update particles.

The output filename records the dataset, particle count, resampling strategy, beam sparsity, seed, and initialization. For example, `simple_world_path1_N1000_always_s5_seed7_gaussian.json` is a metric file; its neighboring PNG and NPZ contain the final distribution and saved numerical arrays.

## Reproduce the particle-filter experiments

The JSON manifests specify independent full-dataset runs. Execute them with:

```bash
python3 experiments/run_experiments.py experiments/particle_sweep.json --jobs 3
python3 experiments/run_experiments.py experiments/additional_counts.json --jobs 3
python3 experiments/run_experiments.py experiments/resampling_comparison.json --jobs 3
python3 experiments/run_experiments.py experiments/robustness_counts.json --jobs 3
```

The batch runner uses the Python interpreter that launches it. It records the command, experiment specification, exit status, elapsed time, and a source-code hash in each run's `execution.json`, alongside `stdout.txt` and `stderr.txt`. Completed runs are reused only when the specification and source hash match. The cache does not hash input data or revalidate saved output files: **use `--overwrite` if data have changed or outputs have been changed, removed, or regenerated**. Add `--overwrite` whenever a fresh run is desired. Use `--jobs 1` to run sequentially or reduce memory use.

| File | Purpose |
|---|---|
| `particle_sweep.json` | Particle-count comparisons for the five known-start cases and the three global-localization cases, using resampling at every step. |
| `additional_counts.json` | Additional particle-count experiments needed to document behavior not resolved by the initial sweep. |
| `resampling_comparison.json` | Never-resample and adaptive-resampling runs at the selected baseline counts. The corresponding always-resample runs are in the sweep. |
| `selected_counts.json` | A dataset-to-count lookup for the baseline comparisons; this is **not** an executable experiment manifest. |
| `robustness_counts.json` | Seven increased-particle-count tests of the never-resample strategy, all with seed `7`: `N=10000` for four known-start cases and `N=20000` for the three global-localization cases. These are particle-count checks, not additional-seed trials. |

The selected baseline counts are:

| Dataset | Initialization | Baseline particles |
|---|---|---:|
| `simple_world_path1` | Known-start position prior | 1,000 |
| `simple_world_path3` | Known-start position prior | 1,000 |
| `loops_world_path0` | Known-start position prior | 1,000 |
| `loops_world_path1` | Known-start position prior | 1,000 |
| `rooms_world_path1` | Known-start position prior | 1,000 |
| `simple_world_path2` | Uniform global initialization | 5,000 |
| `loops_world_path2` | Uniform global initialization | 2,000 |
| `rooms_world_path2` | Uniform global initialization | 2,000 |

These are experimental comparison settings, not universal minimum particle counts or guarantees of convergence. Random seed, initial coverage, measurement ambiguity, and resampling policy all affect the outcome. The report distinguishes baseline runs from any additional counts used to investigate failures.

**Ground-truth limitation:** the three `path2` datasets do not contain `X0` or `XGT`. Their final distributions can be assessed for concentration, map compatibility, and consistency across runs, but a narrow particle cloud alone does not prove correct localization. No ground-truth position RMSE is claimed for those scenarios.

## Reproduce both EKF-SLAM experiments

```bash
python3 code/ekf-slam/RunEKFSLAM.py data/ekf-slam/ekf-slam-small-noise.pickle --output-dir results/ekf/ekf-slam-small-noise
python3 code/ekf-slam/RunEKFSLAM.py data/ekf-slam/ekf-slam-large-noise.pickle --output-dir results/ekf/ekf-slam-large-noise
```

The original `python3 RunEKFSLAM.py file.pickle` invocation also works from its code directory. `--output-dir DIR` controls where results are written. Add `--animate` for live rendering; it is off by default.

Each EKF output directory contains:

| File | Contents |
|---|---|
| `final_map.png` / `.pdf` | Final estimated robot pose and landmark map, 3-sigma position covariance contours, and ground-truth robot/map positions. |
| `trajectories.png` / `.pdf` | Complete EKF-SLAM, dead-reckoning, and ground-truth trajectories. |
| `pose_errors.png` / `.pdf` | Position and wrapped-heading errors over time. |
| `metrics.json` | Position/heading/map errors, actual noise matrices, counts, and covariance checks. |
| `results.npz` | Pose/dead-reckoning histories, pose covariance histories, final joint mean/covariance, landmark indices, aligned ground truth, and innovations. |

The filter initializes from the supplied **`X0`**, not `XGT`. It uses the provided initial covariance and dataset-specific `R` and `Q`; it does not tune inference using ground truth. All 847 controls and all observations are processed. Histories include the initial pose, giving 848 poses at times `0` through `847`, and truth is aligned by timestamp. Map errors match landmark IDs without post-hoc rigid alignment.

A new landmark is initialized with full robot/map cross covariance; its first observation is not immediately reused for a second update. The implementation retains the full joint covariance, uses a linear solve for the gain, and uses the Joseph covariance update. Dead reckoning follows the same motion model and initial pose while omitting measurement corrections.

The dataset names do not imply that every noise term grows together: the `large-noise` file has larger **process** covariance but smaller **measurement** covariance than `small-noise`. Each metric file records the exact matrices used.

## Run the analytic tests

```bash
MPLBACKEND=Agg python3 -m unittest discover -s tests -v
```

The tests use `unittest`; pytest is not required. They cover particle motion and limiting cases, weighting and resampling, circular means, ray geometry and sensor probabilities, plus independently finite-differenced EKF Jacobians and full covariance propagation. Synthetic sequences check timestamp handling, first-observation use, and independence of inference from ground-truth inputs. See [tests/README.md](tests/README.md) for details.

The analytic tests complement the full-data experiments. A passing test suite does not establish convergence of every stochastic localization run.

## Rebuild the report

Report generation needs the additional packages in `requirements-report.txt`:

```bash
python3 -m pip install -r requirements-report.txt
python3 reports/build_report.py
```

Run the required experiments first if their outputs are absent. The report generator reads saved experiment outputs; the filter programs themselves do not require ReportLab or Pillow.

## Attribution

The course-provided starter code and datasets are included with the completed implementation. ChatGPT assisted with the implementation, analytic tests, experiment execution, and report preparation.
