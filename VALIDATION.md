# Validation and reproducibility

This record describes checks actually executed for the packaged implementations and data. The unit-test transcript is in `results/validation.log`; the per-run PF metrics and artifact locations are in `results/pf_summary.csv`. Individual runs retain their JSON metrics, numerical NPZ output, final PNG, execution receipt, and process logs.

## Tested environment

| Component | Version used |
| --- | --- |
| Python | 3.12.14 |
| NumPy | 2.3.5 |
| SciPy | 1.17.0 |
| Matplotlib | 3.10.8 |

`requirements.txt` declares compatible API ranges rather than exact experimental pins. The recorded environment above is the tested combination; every combination at the declared lower bounds was not separately tested. Each PF run uses one deterministic NumPy Generator with the seed recorded in its metadata. The supplied experiment manifests use seed 7. Numerical trajectories are reproducible in the same environment; timing and identical image bytes are not guaranteed across machines or library versions.

## Automated unit tests

The command below completed successfully with **34 tests passed**:

```bash
python -m unittest discover -s tests -v
```

The tests comprise 12 particle-filter tests, 11 beam-model/ray-casting tests, and 11 EKF-SLAM tests. They check independent calculations or invariants:

- Straight, near-zero-turn, and quarter-circle velocity motion; angle wrapping; sampling variances; bounded collision handling.
- Recursive importance weights, log-space behavior after ordinary likelihood underflow, circular heading means, effective sample size, and resampling support/weight reset.
- Truncated Gaussian and exponential beam components, maximum-range and random components, mixture calculation, multi-beam likelihood multiplication, and exact ray intersections including unequal cell resolutions.
- Motion and observation Jacobians against numerical derivatives; process covariance including correlated input noise; inverse/forward observation consistency; and full landmark cross-covariance augmentation.
- EKF processing order at initial and subsequent times, avoiding a second update using a just-initialized landmark's measurement, and covariance reduction under zero innovation.
- An explicit EKF test that changing ground-truth inputs does not change filter inference.

These synthetic checks complement, rather than replace, evaluation on the provided data.

## Complete particle-filter experiment coverage

The four run manifests jointly specify **48 distinct PF runs**, covering all eight provided scenarios:

| Manifest | Runs | Purpose |
| --- | ---: | --- |
| `experiments/particle_sweep.json` | 24 | Three particle counts for each of eight scenarios |
| `experiments/additional_counts.json` | 1 | Additional low-particle failure example |
| `experiments/resampling_comparison.json` | 16 | Never and adaptive resampling on all eight scenarios |
| `experiments/robustness_counts.json` | 7 | Larger particle counts for selected never-resample cases |

All 48 execution receipts reported exit code 0, and all 48 saved artifact sets passed the checks below after the final run completed. No requested run or output was missing.

The original particle sweep uses N = 10, 100, and 1,000 on the five scenarios with X0, and N = 250, 2,000, and 5,000 on the three scenarios without X0. The selected counts and all exact follow-up settings are recorded in the manifests. The baseline sensor parameters are unchanged across the comparisons: mixture weights (0.97, 0.01, 0.01, 0.01), sigmaHit = 0.5 m, lambdaShort = 1/m, zMax = 20 m, zMaxEps = 0.1 m, and beam sparsity 5. No likelihood tempering is used.

Revalidate saved outputs and regenerate the CSV:

```bash
python experiments/summarize_results.py --expected-runs 48
```

The summarizer checks that every requested manifest run has a successful execution receipt; the current inference source hash matches; PNG/NPZ/JSON outputs exist; arrays have the correct dimensions and finite values; weights normalize; ESS is in [1,N]; and the actual resampling decisions match the selected strategy. It verifies the before/after-resampling conventions, agreement between NPZ and JSON metadata, and the time-aligned ground-truth position errors, RMSE, final error, and final-20% RMSE. It writes portable relative artifact paths to the CSV.

The inference source hash covers `PF.py`, `Laser.py`, `Gridmap.py`, and `RunPF.py`, in that order, and is retained in each execution receipt and CSV row. README edits do not alter that hash.

### Five known-initial-position baseline runs

All five N = 1,000, always-resample, seed-7 baseline runs tracked the provided ground truth accurately in these recorded trials:

| Scenario | Position RMSE [m] | Final position error [m] |
| --- | ---: | ---: |
| simple_world_path1 | 0.142103 | 0.057184 |
| simple_world_path3 | 0.099035 | 0.121145 |
| loops_world_path0 | 0.056633 | 0.084803 |
| loops_world_path1 | 0.044997 | 0.022892 |
| rooms_world_path1 | 0.062140 | 0.061035 |

Thus all five recorded runs have position RMSE below 0.15 m and final error below 0.13 m. This is an observation for these seeds and settings, not a guarantee of convergence for every random initialization or a claim that 1,000 is a minimum particle count.

### Timing and ground-truth alignment

The PF data have T control columns, T scan columns, and T+1 ground-truth columns. Ground-truth column 0 is the initial pose. Control and scan column k produce the estimate in column k+1; that estimate is compared with `XGT[:, k+1]`. Accuracy metrics exclude the initial sampled estimate. Headings are compared using wrapped angular errors. The final-20% window begins at `floor(0.8*T)`.

The inference algorithm uses X0 only for permitted initialization. Its Gaussian initializer uses sigma = 0.5 m in xy and uniform heading, as in the starter interface. XGT is used only for evaluation and plotting. The scalar `theta` estimate is a circular mean.

`inference_seconds` includes initialization, filtering, and optional animation. `total_seconds` extends through final plotting but precedes NPZ/JSON serialization. Execution-receipt `wall_seconds` includes subprocess startup and all writing. The experiment runner limits numerical-library thread counts; concurrent runs and machine load still affect timings.

### No-ground-truth cases and plot semantics

The three path2 datasets do not supply X0 or XGT. Their metrics deliberately omit position accuracy fields. A narrow particle cloud, high scan likelihood, or agreement between runs does not establish correct localization: map symmetries can produce confident incorrect modes. Their spread and likelihood metrics are concentration and sensor-consistency diagnostics only.

The final PNG plots the weighted posterior before optional final resampling, matching `posterior_particles` and `posterior_weights`. Marker area is `3 + 65*w_i/max(w)`, so tiny or zero weights remain visible; area is not exactly proportional to probability. The saved `final_particles` and `final_weights` instead describe the actual state after final resampling, if it occurred. The orange trajectory joins posterior means, which may pass through walls or jump between modes when the distribution is ambiguous. Representative known-pose and global-localization PNGs were visually reviewed for legibility and labelling.

## EKF-SLAM data runs

Both provided EKF datasets were run for all 847 controls and 848 timestamp-matched poses. Each run estimates 24 landmarks, producing a 51-dimensional final state. The output metrics report finite states, zero recorded covariance symmetry error, and positive minimum eigenvalues across the recorded covariance sequence.

| Dataset | SLAM position RMSE [m] | Dead-reckoning position RMSE [m] | Map position RMSE [m] |
| --- | ---: | ---: | ---: |
| ekf-slam-small-noise | 0.066696 | 0.645763 | 0.059539 |
| ekf-slam-large-noise | 0.189853 | 2.883096 | 0.186414 |

These EKF metrics include the initial pose, as explicitly stated in their per-run JSON files; this convention differs from the PF metrics. Map errors use matching landmark IDs in the anchored world frame, without a post-hoc alignment. The innovation statistics and covariance checks are diagnostics; no claim of exact nonlinear uncertainty calibration follows from passing them.

## Re-running and packaging notes

Run commands from the submission root, or invoke the experiment scripts by absolute path. The manifest runner resolves the package location from its own file. Historical absolute paths in execution receipts are provenance, not commands that must be copied unchanged onto another machine.

Use separate output directories when comparing adaptive thresholds because the PF filename does not include the threshold. The manifest runner caches matching successful specs/source hashes; if input data change or output files are removed, pass `--overwrite`. The output summarizer independently detects missing files. No full input-data checksum is asserted by this record.
