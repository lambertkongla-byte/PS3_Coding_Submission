"""Bootstrap particle-filter localization with reproducible experiment outputs."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from Visualization import Visualization


def wrap_angle(angle):
    """Map angles to [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


class PF(object):
    """Particle poses are (3,N); normalized public weights are (1,N).

    The original first five constructor arguments and public method signatures
    remain valid. All random choices use one seeded NumPy Generator. No ground
    truth enters prediction, initialization, weighting, or resampling.
    """

    def __init__(self, numParticles, Alpha, laser, gridmap, visualize=True,
                 seed=0, resampling="always", neff_threshold=2.0 / 3.0,
                 init="auto", gaussian_sigma=0.5):
        self.numParticles = int(numParticles)
        self.Alpha = np.asarray(Alpha, dtype=float).reshape(-1)
        if self.numParticles < 1 or self.Alpha.size != 6 or np.any(self.Alpha < 0):
            raise ValueError("Use positive N and six nonnegative motion coefficients")
        if resampling not in ("always", "never", "adaptive"):
            raise ValueError("Unknown resampling strategy")
        if init not in ("auto", "gaussian", "uniform"):
            raise ValueError("Unknown initialization strategy")
        if not 0.0 < neff_threshold <= 1.0 or gaussian_sigma <= 0:
            raise ValueError("ESS fraction must be in (0,1]; sigma must be positive")
        self.laser, self.gridmap = laser, gridmap
        self.visualize = bool(visualize)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.resampling = resampling
        self.neff_threshold = float(neff_threshold)
        self.init = init
        self.gaussian_sigma = float(gaussian_sigma)
        self.max_motion_attempts = 10
        self.particles = None
        self.weights = None
        self.log_weights = None
        self._weights_snapshot = None
        self.impossible_scans_skipped = 0
        self.last_collision_rejections = 0
        self.last_collision_fallbacks = 0
        self.vis = Visualization() if self.visualize else None
        if self.vis is not None:
            self.vis.drawGridmap(gridmap)

    def _set_log_weights(self, values):
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size != self.numParticles:
            raise ValueError("One log weight is required per particle")
        normalizer = logsumexp(values)
        if not np.isfinite(normalizer):
            raise ValueError("Particle weights contain no finite positive mass")
        self.log_weights = values - normalizer
        self.weights = np.exp(self.log_weights)[None, :]
        self._weights_snapshot = self.weights.copy()

    def _uniform_weights(self):
        self._set_log_weights(np.zeros(self.numParticles))

    def _synchronize_log_weights(self):
        # Preserve exact log weights even if exp(log_weight) underflows. Also
        # allow callers of the original public interface to assign weights.
        if (self.log_weights is None or self._weights_snapshot is None
                or not np.array_equal(self.weights, self._weights_snapshot)):
            w = self.getNormalizedWeights().reshape(-1)
            logw = np.full(w.shape, -np.inf)
            np.log(w, out=logw, where=w > 0.0)
            self._set_log_weights(logw)

    def sampleParticlesUniform(self):
        """Sample uniformly in free map area and uniformly in heading."""
        free_rows, free_cols = np.nonzero(~self.gridmap.occupancy)
        if free_rows.size == 0:
            raise ValueError("Cannot initialize particles in an entirely occupied map")
        cells = self.rng.integers(free_rows.size, size=self.numParticles)
        self.particles = np.empty((3, self.numParticles))
        # All cells have equal area, so uniform free-cell selection followed by
        # uniform within-cell offsets is uniform over continuous free space.
        self.particles[0] = (free_cols[cells] + self.rng.random(self.numParticles)) * self.gridmap.xres
        self.particles[1] = (free_rows[cells] + self.rng.random(self.numParticles)) * self.gridmap.yres
        self.particles[2] = self.rng.uniform(-np.pi, np.pi, self.numParticles)
        self._uniform_weights()

    def sampleParticlesGaussian(self, x0, y0, sigma):
        """Gaussian xy truncated to free space; heading is uniform as provided."""
        if sigma <= 0:
            raise ValueError("The Gaussian xy standard deviation must be positive")
        self.particles = np.empty((3, self.numParticles))
        pending = np.arange(self.numParticles)
        for _ in range(100):
            candidate = self.rng.normal(np.array([[x0], [y0]]), sigma, (2, pending.size))
            good = ~self.gridmap.inCollision(candidate[0], candidate[1])
            self.particles[:2, pending[good]] = candidate[:, good]
            pending = pending[~good]
            if pending.size == 0:
                break
        if pending.size:
            raise ValueError("Gaussian initialization could not reach free space after 100 tries")
        self.particles[2] = self.rng.uniform(-np.pi, np.pi, self.numParticles)
        self._uniform_weights()

    def getParticle(self, k):
        if 0 <= k < self.particles.shape[1]:
            return self.particles[:, k], self.weights[:, k]
        return None, None

    def getNormalizedWeights(self):
        weights = np.asarray(self.weights, dtype=float).reshape(1, -1)
        if (weights.size != self.numParticles or np.any(~np.isfinite(weights))
                or np.any(weights < 0.0) or weights.sum() <= 0.0):
            raise ValueError("Particle weights must contain finite nonnegative positive mass")
        return weights / weights.sum()

    def effectiveSampleSize(self):
        weights = self.getNormalizedWeights().reshape(-1)
        return float(1.0 / np.dot(weights, weights))

    def getMean(self):
        """Weighted xy mean and circular (not arithmetic) heading mean."""
        weights = self.getNormalizedWeights().reshape(-1)
        xy = self.particles[:2] @ weights
        sine = np.sin(self.particles[2]) @ weights
        cosine = np.cos(self.particles[2]) @ weights
        if np.hypot(sine, cosine) < 1e-14:
            heading = self.particles[2, np.argmax(weights)]
        else:
            heading = np.arctan2(sine, cosine)
        return np.array([xy[0], xy[1], float(wrap_angle(heading))])

    def sampleMotion(self, u, deltat, particles=None):
        """Draw one Table 5.3 velocity-model proposal per input particle.

        This helper does not reject collisions; prediction does. Noise formulas
        specify variances, so normal draws are multiplied by their square roots.
        """
        old = self.particles if particles is None else np.asarray(particles, dtype=float)
        velocity, angular_velocity = np.asarray(u, dtype=float).reshape(2)
        if deltat < 0:
            raise ValueError("deltat must be nonnegative")
        variances = self.Alpha.reshape(3, 2) @ np.array([velocity ** 2, angular_velocity ** 2])
        noise = self.rng.normal(size=(3, old.shape[1])) * np.sqrt(variances)[:, None]
        noisy_v, noisy_w, gamma = velocity + noise[0], angular_velocity + noise[1], noise[2]
        half_turn = 0.5 * noisy_w * deltat
        # sin(a+b)-sin(a) = 2 sin(b/2) cos(a+b/2). This sinc form
        # has the exact straight-motion limit when angular velocity is zero.
        travel = noisy_v * deltat * np.sinc(half_turn / np.pi)
        proposal = old.copy()
        proposal[0] += travel * np.cos(old[2] + half_turn)
        proposal[1] += travel * np.sin(old[2] + half_turn)
        proposal[2] = wrap_angle(old[2] + noisy_w * deltat + gamma * deltat)
        return proposal

    def prediction(self, u, deltat):
        """Propose motion; retry occupied endpoints at most ten times."""
        old = self.particles.copy()
        predicted = old.copy()
        pending = np.arange(self.numParticles)
        rejected = 0
        for _ in range(self.max_motion_attempts):
            candidate = self.sampleMotion(u, deltat, old[:, pending])
            good = ~self.gridmap.inCollision(candidate[0], candidate[1])
            predicted[:, pending[good]] = candidate[:, good]
            rejected += int((~good).sum())
            pending = pending[~good]
            if pending.size == 0:
                break
        # As suggested in the starter, keep the prior pose if every bounded
        # proposal collided; no unbounded loop or map-dependent teleport occurs.
        self.particles = predicted
        self.particles[2] = wrap_angle(self.particles[2])
        self.last_collision_rejections = rejected
        self.last_collision_fallbacks = int(pending.size)
        return self.particles

    def update(self, ranges):
        """Recursively weight particles by the full independent-beam product.

        For a bootstrap proposal, w_t is proportional to w_(t-1) p(z_t|x_t).
        Keeping the previous log weights is essential when resampling is not
        performed on every step. Resampling alone resets them to equal values.
        """
        self._synchronize_log_weights()
        if hasattr(self.laser, "scanLogProbability"):
            likelihood = self.laser.scanLogProbability(ranges, self.particles, self.gridmap)
        else:
            probability = np.asarray(self.laser.scanProbability(ranges, self.particles, self.gridmap))
            likelihood = np.full(probability.shape, -np.inf, dtype=float)
            np.log(probability, out=likelihood, where=probability > 0)
        posterior = self.log_weights + np.asarray(likelihood).reshape(-1)
        if np.isfinite(logsumexp(posterior)):
            self._set_log_weights(posterior)
        else:
            # An impossible scan cannot define a posterior. Preserve the prior
            # rather than inventing a location; record the event in the output.
            self.impossible_scans_skipped += 1
        return self.weights

    def resample(self):
        """Systematic resampling with replacement, then reset weights to 1/N."""
        weights = self.getNormalizedWeights().reshape(-1)
        cdf = np.cumsum(weights)
        cdf[-1] = 1.0
        positions = (self.rng.random() + np.arange(self.numParticles)) / self.numParticles
        indices = np.searchsorted(cdf, positions, side="right")
        self.particles = self.particles[:, indices].copy()
        self._uniform_weights()
        return indices

    def render(self, ranges, deltat, XGT):
        if self.vis is None:
            return
        self.vis.drawParticles(self.particles)
        if XGT is not None:
            self.vis.drawLidar(ranges, self.laser.Angles, *XGT)
            self.vis.drawGroundTruthPose(*XGT)
        mean = self.getMean()
        self.vis.drawMeanPose(mean[0], mean[1])
        plt.pause(max(float(deltat), 0.001))

    def _save_final_plot(self, path, title, means, XGT, posterior_particles, posterior_weights):
        fig, ax = plt.subplots(figsize=(7.2, 7.5))
        ax.imshow(self.gridmap.occupancy, origin="lower", interpolation="nearest",
                  extent=(0, self.gridmap.n * self.gridmap.xres,
                          0, self.gridmap.m * self.gridmap.yres),
                  cmap=ListedColormap(["#ffffff", "#335d7e"]), vmin=0, vmax=1)
        ax.plot(means[0], means[1], color="#d18a00", lw=1, alpha=0.65, label="Posterior mean path")
        sizes = 3.0 + 65.0 * posterior_weights / max(float(posterior_weights.max()), 1e-300)
        ax.scatter(posterior_particles[0], posterior_particles[1], s=sizes,
                   color="#171b28", alpha=0.45, linewidths=0, label="Final weighted particles")
        ax.plot(means[0, -1], means[1, -1], marker="*", ms=14, color="#ef9d00",
                markeredgecolor="white", label="Final posterior mean")
        heading = means[2, -1]
        ax.arrow(means[0, -1], means[1, -1], 0.55 * np.cos(heading), 0.55 * np.sin(heading),
                 width=0.045, color="#b66c00", length_includes_head=True)
        if XGT is not None:
            ax.plot(XGT[0], XGT[1], "--", color="#cf3344", lw=1.2, label="Ground truth (evaluation)")
            ax.plot(XGT[0, -1], XGT[1, -1], "x", color="#cf3344", ms=10, mew=2)
        ax.set_aspect("equal")
        ax.set(xlabel="x [m]", ylabel="y [m]", title=title)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def run(self, U, Ranges, deltat, X0, XGT, filename):
        """Run the filter and save PNG, NPZ histories, and JSON metrics.

        U[:,k] and Ranges[:,k] produce estimate k+1. XGT[:,0] is the
        initial ground truth, so evaluation pairs the result with XGT[:,k+1].
        Estimates and ESS histories are recorded before optional resampling.
        """
        started = perf_counter()
        U, Ranges = np.asarray(U, dtype=float), np.asarray(Ranges, dtype=float)
        if U.ndim != 2 or U.shape[0] != 2 or Ranges.shape != (self.laser.Angles.size, U.shape[1]):
            raise ValueError("Expected U (2,T) and Ranges (retained beams,T)")
        steps = U.shape[1]
        if steps == 0:
            raise ValueError("At least one timestep is required")
        initialization = ("gaussian" if X0 is not None else "uniform") if self.init == "auto" else self.init
        if initialization == "gaussian":
            if X0 is None:
                raise ValueError("Gaussian initialization requires the supplied X0")
            initial = np.asarray(X0, dtype=float).reshape(3)
            self.sampleParticlesGaussian(initial[0], initial[1], self.gaussian_sigma)
        else:
            self.sampleParticlesUniform()
        means = np.empty((3, steps + 1))
        means[:, 0] = self.getMean()
        neff = np.empty(steps)
        resampled = np.zeros(steps, dtype=bool)
        collision_rejections = np.zeros(steps, dtype=int)
        collision_fallbacks = np.zeros(steps, dtype=int)

        for k in range(steps):
            self.prediction(U[:, k], deltat)
            collision_rejections[k] = self.last_collision_rejections
            collision_fallbacks[k] = self.last_collision_fallbacks
            self.update(Ranges[:, k])
            means[:, k + 1] = self.getMean()
            neff[k] = self.effectiveSampleSize()
            if k == steps - 1:
                posterior_particles = self.particles.copy()
                posterior_weights = self.getNormalizedWeights().reshape(-1).copy()
            resampled[k] = (self.resampling == "always" or
                            (self.resampling == "adaptive" and neff[k] < self.neff_threshold * self.numParticles))
            if resampled[k]:
                self.resample()
            if self.visualize:
                self.render(Ranges[:, k], deltat, None if XGT is None else np.asarray(XGT)[:, k + 1])

        inference_seconds = perf_counter() - started
        # Everything involving XGT below is evaluation, after inference ends.
        truth = None if XGT is None else np.asarray(XGT, dtype=float)
        if truth is not None and truth.shape != means.shape:
            raise ValueError("XGT must be (3,T+1), including its initial column")
        centered = posterior_particles[:2] - means[:2, -1, None]
        covariance_xy = (centered * posterior_weights) @ centered.T
        loglik = np.asarray(self.laser.scanLogProbability(Ranges[:, -1], posterior_particles, self.gridmap))
        mean_loglik = float(self.laser.scanLogProbability(Ranges[:, -1], means[:, -1], self.gridmap))
        params = {name: float(getattr(self.laser, name)) for name in
                  ("pHit", "pShort", "pMax", "pRand", "sigmaHit", "lambdaShort", "zMax", "zMaxEps")}
        metrics = {
            "num_particles": self.numParticles, "num_steps": steps,
            "num_beams": int(self.laser.Angles.size), "seed": self.seed,
            "resampling": self.resampling, "neff_threshold_fraction": self.neff_threshold,
            "initialization": initialization, "gaussian_xy_sigma": self.gaussian_sigma,
            "initial_heading_distribution": "uniform[-pi,pi)",
            "sparsity": int(getattr(self.laser, "sparsity", 1)),
            "laser_parameters": params, "motion_alpha": self.Alpha.tolist(),
            "inference_seconds": inference_seconds, "resampling_count": int(resampled.sum()),
            "mean_neff": float(neff.mean()), "minimum_neff": float(neff.min()),
            "final_neff_before_resampling": float(neff[-1]),
            "impossible_scans_skipped": self.impossible_scans_skipped,
            "collision_rejections": int(collision_rejections.sum()),
            "collision_fallbacks": int(collision_fallbacks.sum()),
            "final_position_spread_rms": float(np.sqrt(np.trace(covariance_xy))),
            "final_weighted_xy_covariance": covariance_xy.tolist(),
            "final_mean": means[:, -1].tolist(),
            "final_scan_mean_pose_log_likelihood": mean_loglik if np.isfinite(mean_loglik) else None,
            "final_scan_best_particle_log_likelihood": float(np.max(loglik)),
            "ground_truth_available": truth is not None,
            "ground_truth_alignment": "estimate[:,k+1] versus XGT[:,k+1] for input column k",
            "means_recorded": "weighted posterior before resampling",
            "final_particles_recorded": "after final resampling if performed",
        }
        if truth is not None:
            position_errors = np.linalg.norm(means[:2, 1:] - truth[:2, 1:], axis=0)
            heading_errors = wrap_angle(means[2, 1:] - truth[2, 1:])
            tail_start = int(np.floor(0.8 * steps))
            metrics.update({
                "position_rmse": float(np.sqrt(np.mean(position_errors ** 2))),
                "final_position_error": float(position_errors[-1]),
                "tail20_position_rmse": float(np.sqrt(np.mean(position_errors[tail_start:] ** 2))),
                "heading_rmse_rad": float(np.sqrt(np.mean(heading_errors ** 2))),
                "final_heading_error_rad": float(abs(heading_errors[-1])),
                "tail20_heading_rmse_rad": float(np.sqrt(np.mean(heading_errors[tail_start:] ** 2))),
            })
        else:
            position_errors = np.empty(0)
            heading_errors = np.empty(0)

        prefix = Path(filename)
        if prefix.suffix.lower() == ".png":
            prefix = prefix.with_suffix("")
        prefix.parent.mkdir(parents=True, exist_ok=True)
        plot_path = Path(str(prefix) + ".png")
        npz_path = Path(str(prefix) + ".npz")
        json_path = Path(str(prefix) + ".json")
        title = (f"{prefix.name.split('_N')[0].replace('_', ' ')}\n"
                 f"N={self.numParticles}, {self.resampling}, seed={self.seed}, init={initialization}")
        self._save_final_plot(plot_path, title, means, truth, posterior_particles, posterior_weights)
        metrics["total_seconds"] = perf_counter() - started
        metadata_json = json.dumps(metrics, sort_keys=True, allow_nan=False)
        np.savez_compressed(
            npz_path, final_particles=self.particles,
            final_weights=self.getNormalizedWeights().reshape(-1),
            posterior_particles=posterior_particles, posterior_weights=posterior_weights,
            means=means, estimated_poses=means, neff_before_resampling=neff,
            resampled=resampled, collision_rejections=collision_rejections,
            collision_fallbacks=collision_fallbacks,
            XGT=np.empty((3, 0)) if truth is None else truth,
            position_errors=position_errors, heading_errors_rad=heading_errors,
            time=np.arange(steps + 1) * deltat, seed=self.seed, strategy=self.resampling,
            init=initialization, sigma=self.gaussian_sigma,
            sparsity=metrics["sparsity"], metadata_json=metadata_json)
        json_path.write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n")
        self.means = means
        self.neff_history = neff
        self.resampled_history = resampled
        self.metrics = metrics
        self.output_paths = {"png": str(plot_path), "npz": str(npz_path), "json": str(json_path)}
        return metrics
