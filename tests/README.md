# Analytic filter checks

From the submission directory, run:

```bash
MPLBACKEND=Agg python3 -m unittest discover -s tests -v
```

The suite uses the Python standard library's `unittest`, NumPy, and the filter
dependencies. It does not require pytest or the supplied experiment data.

The tests check deterministic and limiting PF motion, collision fallback,
recursive importance weights, numerical underflow, effective sample size,
resampling, circular means, exact ray geometry, and beam-mixture normalization.
EKF checks use independently finite-differenced process, measurement, and inverse
measurement functions as oracles for full joint covariance calculations. Small
synthetic sequences also check time alignment and that a first observation is
used once when its landmark is initialized.

These checks validate mathematical behavior; they do not substitute for the
assignment's eight particle-filter scenarios or its two EKF-SLAM experiments.
