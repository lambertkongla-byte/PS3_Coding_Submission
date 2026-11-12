"""Extended Kalman filter SLAM for the Cartesian feature model in PS3.

The state is [x, y, heading, landmark_1_x, landmark_1_y, ...]. Ground-truth
arrays are retained only for plotting and evaluation, never for inference.
"""

import json
from pathlib import Path
import time

import numpy as np


class EKFSLAM:
    """EKF SLAM with full robot-map correlations and first-sighting augmentation."""

    def __init__(self, mu, Sigma, R, Q, XGT=None, MGT=None,
                 animate=False, initial_time=0.0):
        self.mu = np.asarray(mu, dtype=float).reshape(-1).copy()
        self.Sigma = np.asarray(Sigma, dtype=float).copy()
        self.R = np.asarray(R, dtype=float).copy()
        self.Q = np.asarray(Q, dtype=float).copy()
        if self.mu.size < 3 or (self.mu.size - 3) % 2:
            raise ValueError("State must contain a robot pose and 2-D landmarks.")
        if self.Sigma.shape != (self.mu.size, self.mu.size):
            raise ValueError("State covariance has the wrong shape.")
        if self.R.shape != (2, 2) or self.Q.shape != (2, 2):
            raise ValueError("R and Q must both be 2 x 2 covariance matrices.")
        for name, covariance in (("Sigma", self.Sigma), ("R", self.R), ("Q", self.Q)):
            if not np.all(np.isfinite(covariance)):
                raise ValueError(f"{name} contains non-finite values.")
            if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
                raise ValueError(f"{name} must be symmetric.")
            if np.linalg.eigvalsh(covariance).min() < -1e-10:
                raise ValueError(f"{name} must be positive semidefinite.")
        if not np.all(np.isfinite(self.mu)):
            raise ValueError("Initial mean contains non-finite values.")
        self.mu[2] = self.angleWrap(self.mu[2])
        self.initial_mu = self.mu.copy()
        self.initial_covariance = self.Sigma.copy()
        self.initial_time = float(initial_time)
        self.XGT = None if XGT is None else np.asarray(XGT, dtype=float).copy()
        self.MGT = None if MGT is None else np.asarray(MGT, dtype=float).copy()
        # Preserve the string-key convention used in the supplied Renderer.py.
        self.mapLUT = {}
        self.animate = bool(animate)
        self.renderer = None
        self.prediction_count = 0
        self.update_count = 0
        self.augmentation_count = 0
        self.innovations = []
        self.current_time = self.initial_time
        self._has_run = False
        if self.animate:
            from Renderer import Renderer
            points = []
            if self.XGT is not None:
                points.append(self.XGT[1:3])
            if self.MGT is not None:
                points.append(self.MGT[1:3])
            if points:
                xy = np.concatenate(points, axis=1)
                xlim = (float(xy[0].min()) - 2, float(xy[0].max()) + 2)
                ylim = (float(xy[1].min()) - 2, float(xy[1].max()) + 2)
            else:
                xlim, ylim = (-8, 8), (-8, 8)
            self.renderer = Renderer(xlim, ylim, 3, "#c34d26", "#2f7199")
            if self.MGT is not None:
                self.renderer.drawMap(self.MGT)

    @staticmethod
    def angleWrap(theta):
        """Wrap scalar or array angles into [-pi, pi)."""
        return (np.asarray(theta) + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _key(feature_id):
        value = float(feature_id)
        if not np.isfinite(value) or not value.is_integer():
            raise ValueError("Landmark IDs must be finite integers.")
        return str(int(value))

    @staticmethod
    def _motion(pose, control):
        """Equation (5), evaluated at zero process noise."""
        distance, dtheta = np.asarray(control, dtype=float).reshape(2)
        heading = pose[2]
        result = np.asarray(pose, dtype=float).copy()
        result[0] += distance * np.cos(heading)
        result[1] += distance * np.sin(heading)
        result[2] = EKFSLAM.angleWrap(heading + dtheta)
        return result

    def prediction(self, u):
        """Predict with distance/heading noise and preserve map correlations.

        F differentiates the motion with respect to the old robot pose.
        V differentiates it with respect to (forward-distance, heading) noise.
        R is a 2 x 2 noise covariance; it is not added directly to pose covariance.
        """
        control = np.asarray(u, dtype=float).reshape(2)
        if not np.all(np.isfinite(control)):
            raise ValueError("Control contains non-finite values.")
        distance = control[0]
        c, s = np.cos(self.mu[2]), np.sin(self.mu[2])
        F = np.array([[1, 0, -distance * s],
                      [0, 1, distance * c],
                      [0, 0, 1]], dtype=float)
        V = np.array([[c, 0], [s, 0], [0, 1]], dtype=float)
        previous = self.Sigma
        predicted = previous.copy()
        predicted[:3, :3] = F @ previous[:3, :3] @ F.T + V @ self.R @ V.T
        predicted[:3, 3:] = F @ previous[:3, 3:]
        predicted[3:, :3] = predicted[:3, 3:].T
        # Landmarks are static, so the map-map covariance block is unchanged.
        self.mu[:3] = self._motion(self.mu[:3], control)
        self.Sigma = (predicted + predicted.T) * 0.5
        self.prediction_count += 1

    def update(self, z, id):
        """Update a known landmark using a solve and Joseph covariance form."""
        key = self._key(id)
        if key not in self.mapLUT:
            raise KeyError("Call augmentState for a landmark's first observation.")
        reading = np.asarray(z, dtype=float).reshape(2)
        if not np.all(np.isfinite(reading)):
            raise ValueError("Measurement contains non-finite values.")
        index = self.mapLUT[key]
        c, s = np.cos(self.mu[2]), np.sin(self.mu[2])
        world_to_robot = np.array([[c, s], [-s, c]])
        predicted = world_to_robot @ (self.mu[index:index + 2] - self.mu[:2])
        H = np.zeros((2, self.mu.size))
        H[:, :3] = np.array([[-c, -s, predicted[1]],
                              [s, -c, -predicted[0]]])
        H[:, index:index + 2] = world_to_robot
        innovation = reading - predicted
        PHt = self.Sigma @ H.T
        S = H @ PHt + self.Q
        S = (S + S.T) * 0.5
        # Solve S K^T = (P H^T)^T; avoid forming an explicit inverse.
        K = np.linalg.solve(S, PHt.T).T
        self.mu += K @ innovation
        self.mu[2] = self.angleWrap(self.mu[2])
        A = np.eye(self.mu.size) - K @ H
        covariance = A @ self.Sigma @ A.T + K @ self.Q @ K.T
        self.Sigma = (covariance + covariance.T) * 0.5
        nis = float(innovation @ np.linalg.solve(S, innovation))
        self.innovations.append([self.current_time, int(key),
                                 float(innovation[0]), float(innovation[1]), nis])
        self.update_count += 1

    def augmentState(self, z, id):
        """Initialize m = p + C(theta) z, including all cross-covariances."""
        key = self._key(id)
        if key in self.mapLUT:
            raise ValueError("Landmark is already present; use update instead.")
        reading = np.asarray(z, dtype=float).reshape(2)
        if not np.all(np.isfinite(reading)):
            raise ValueError("Measurement contains non-finite values.")
        c, s = np.cos(self.mu[2]), np.sin(self.mu[2])
        C = np.array([[c, -s], [s, c]])
        landmark = self.mu[:2] + C @ reading
        G = np.zeros((2, self.mu.size))
        G[:, :3] = np.array([[1, 0, -s * reading[0] - c * reading[1]],
                              [0, 1, c * reading[0] - s * reading[1]]])
        cross = self.Sigma @ G.T
        landmark_covariance = G @ cross + C @ self.Q @ C.T
        augmented = np.block([[self.Sigma, cross],
                              [cross.T, landmark_covariance]])
        self.mapLUT[key] = self.mu.size
        self.mu = np.concatenate((self.mu, landmark))
        self.Sigma = (augmented + augmented.T) * 0.5
        self.augmentation_count += 1

    def _ground_truth_at(self, timestamp):
        """Read a truth pose only for animation/evaluation, never for filtering."""
        if self.XGT is None:
            return None
        matches = np.flatnonzero(np.isclose(self.XGT[0], timestamp, rtol=0, atol=1e-9))
        if len(matches) != 1:
            raise ValueError(f"Expected one ground-truth pose at time {timestamp}.")
        return self.XGT[1:4, matches[0]]

    def run(self, U, Z):
        """Process every control and observation, matched by actual timestamps.

        U has columns [time, distance, heading_change]; Z has columns
        [time, landmark_id, relative_x, relative_y]. The initial state is at
        initial_time (zero in the released data). Histories include that pose.
        A first observation augments once and is not also used in an update.
        """
        if self._has_run:
            raise RuntimeError("Create a fresh EKFSLAM instance for each run.")
        U = np.asarray(U, dtype=float)
        Z = np.asarray(Z, dtype=float)
        if U.ndim != 2 or U.shape[0] != 3 or Z.ndim != 2 or Z.shape[0] != 4:
            raise ValueError("Expected U with shape (3,T) and Z with shape (4,N).")
        if not np.all(np.isfinite(U)) or not np.all(np.isfinite(Z)):
            raise ValueError("Controls and observations must be finite.")
        U = U[:, np.argsort(U[0], kind="stable")]
        if U.shape[1] and (U[0, 0] <= self.initial_time or np.any(np.diff(U[0]) <= 0)):
            raise ValueError("Control times must be unique and after initial_time.")
        self.times = np.concatenate(([self.initial_time], U[0]))
        observations = {float(t): [] for t in self.times}
        for column in Z.T:
            timestamp = float(column[0])
            # Released timestamps are exact integers; use no index-to-time assumption.
            if timestamp not in observations:
                raise ValueError(f"Observation time {timestamp} has no matching control/state.")
            observations[timestamp].append(column)

        self._has_run = True
        start = time.perf_counter()
        pose_history, dead_history, covariance_history = [], [], []
        min_eigenvalues, symmetry_errors = [], []
        dead_pose = self.initial_mu[:3].copy()
        for step, timestamp in enumerate(self.times):
            self.current_time = float(timestamp)
            if step:
                self.prediction(U[1:3, step - 1])
                dead_pose = self._motion(dead_pose, U[1:3, step - 1])
            columns = observations[float(timestamp)]
            for column in columns:
                if self._key(column[1]) in self.mapLUT:
                    self.update(column[2:4], column[1])
                else:
                    self.augmentState(column[2:4], column[1])
            if not np.all(np.isfinite(self.mu)) or not np.all(np.isfinite(self.Sigma)):
                raise FloatingPointError(f"Non-finite filter state at time {timestamp}.")
            pose_history.append(self.mu[:3].copy())
            dead_history.append(dead_pose.copy())
            covariance_history.append(self.Sigma[:3, :3].copy())
            min_eigenvalues.append(float(np.linalg.eigvalsh(self.Sigma)[0]))
            symmetry_errors.append(float(np.max(np.abs(self.Sigma - self.Sigma.T))))
            if self.renderer is not None:
                Zt = np.array(columns).T if columns else np.empty((4, 0))
                self.renderer.render(self.mu, self.Sigma, self._ground_truth_at(timestamp),
                                     Zt, self.mapLUT)
        self.pose_history = np.asarray(pose_history).T
        self.dead_reckoning_history = np.asarray(dead_history).T
        self.pose_covariance_history = np.asarray(covariance_history)
        self.covariance_min_eigenvalue_history = np.asarray(min_eigenvalues)
        self.covariance_symmetry_error_history = np.asarray(symmetry_errors)
        self.runtime_seconds = time.perf_counter() - start
        self.observation_count = Z.shape[1]
        return self

    def metrics(self):
        """Compute metrics after inference, aligning truth by timestamp and ID."""
        if not self._has_run:
            raise RuntimeError("Run the filter before evaluating it.")
        values = {
            "control_count": self.prediction_count,
            "pose_count_including_initial": int(len(self.times)),
            "observation_count": int(self.observation_count),
            "landmark_initializations": self.augmentation_count,
            "known_landmark_updates": self.update_count,
            "landmarks_estimated": len(self.mapLUT),
            "first_time": float(self.times[0]), "last_time": float(self.times[-1]),
            "initialization_source": "X0 (ground truth is never used for inference)",
            "initial_pose": self.initial_mu[:3].tolist(),
            "initial_pose_covariance": self.initial_covariance[:3, :3].tolist(),
            "process_covariance_R": self.R.tolist(),
            "measurement_covariance_Q": self.Q.tolist(),
            "runtime_seconds": self.runtime_seconds,
            "final_state_dimension": int(self.mu.size),
            "covariance_min_eigenvalue_final": float(np.linalg.eigvalsh(self.Sigma)[0]),
            "covariance_min_eigenvalue_over_recorded_posteriors":
                float(self.covariance_min_eigenvalue_history.min()),
            "covariance_max_symmetry_error": float(self.covariance_symmetry_error_history.max()),
            "all_states_finite": bool(np.all(np.isfinite(self.pose_history))
                                      and np.all(np.isfinite(self.Sigma))),
            "metric_convention": "Euclidean position RMSE over all timestamp-matched poses, including initial pose; heading errors wrapped to [-pi,pi). Map errors use matching IDs in the anchored world frame, with no post-hoc alignment.",
        }
        if self.XGT is not None:
            truth = np.column_stack([self._ground_truth_at(t) for t in self.times])
            for name, history in (("slam", self.pose_history),
                                  ("dead_reckoning", self.dead_reckoning_history)):
                error = history - truth
                error[2] = self.angleWrap(error[2])
                position = np.linalg.norm(error[:2], axis=0)
                values[name] = {
                    "position_rmse_m": float(np.sqrt(np.mean(position ** 2))),
                    "position_mean_error_m": float(np.mean(position)),
                    "position_max_error_m": float(np.max(position)),
                    "final_position_error_m": float(position[-1]),
                    "heading_rmse_rad": float(np.sqrt(np.mean(error[2] ** 2))),
                    "heading_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(error[2] ** 2)))),
                    "final_heading_error_rad": float(abs(error[2, -1])),
                    "final_heading_error_deg": float(np.rad2deg(abs(error[2, -1]))),
                    "final_pose": history[:, -1].tolist(),
                }
        if self.MGT is not None:
            map_truth = {self._key(col[0]): col[1:3] for col in self.MGT.T}
            matched = [key for key in self.mapLUT if key in map_truth]
            map_errors = np.array([self.mu[self.mapLUT[key]:self.mapLUT[key] + 2]
                                   - map_truth[key] for key in matched])
            if len(matched):
                norms = np.linalg.norm(map_errors, axis=1)
                values["map"] = {"matched_landmarks": len(matched),
                                  "ground_truth_landmarks": int(self.MGT.shape[1]),
                                  "position_rmse_m": float(np.sqrt(np.mean(norms ** 2))),
                                  "position_max_error_m": float(norms.max())}
        if self.innovations:
            values["mean_normalized_innovation_squared"] = float(np.mean(np.asarray(self.innovations)[:, 4]))
        return values

    @staticmethod
    def _ellipse(ax, mean, covariance, color, linewidth=1.0, label=None):
        """Plot a 3-sigma covariance contour using a symmetric eigensolver."""
        from matplotlib.patches import Ellipse
        covariance = (covariance + covariance.T) * 0.5
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if eigenvalues.min() < -1e-9:
            raise ValueError("Cannot plot an indefinite position covariance.")
        eigenvalues = np.maximum(eigenvalues, 0)
        orientation = np.rad2deg(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
        ellipse = Ellipse(mean, width=6 * np.sqrt(eigenvalues[1]),
                          height=6 * np.sqrt(eigenvalues[0]), angle=orientation,
                          edgecolor=color, facecolor="none", linewidth=linewidth,
                          label=label, zorder=2)
        ax.add_patch(ellipse)

    def save_results(self, output_dir, dataset_name="ekf-slam"):
        """Save reproducible arrays, metrics, and the two required figure types."""
        import matplotlib.pyplot as plt
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics = self.metrics()
        metrics["dataset"] = dataset_name
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        truth = (np.column_stack([self._ground_truth_at(t) for t in self.times])
                 if self.XGT is not None else np.empty((3, 0)))
        sorted_ids = sorted(self.mapLUT, key=int)
        np.savez_compressed(
            output_dir / "results.npz", times=self.times, pose_history=self.pose_history,
            dead_reckoning_history=self.dead_reckoning_history,
            pose_covariance_history=self.pose_covariance_history,
            final_mean=self.mu, final_covariance=self.Sigma,
            landmark_ids=np.array([int(key) for key in sorted_ids]),
            landmark_state_indices=np.array([self.mapLUT[key] for key in sorted_ids]),
            ground_truth_pose_aligned=truth,
            ground_truth_map=self.MGT if self.MGT is not None else np.empty((3, 0)),
            innovations=np.array(self.innovations).reshape(-1, 5),
            covariance_min_eigenvalue_history=self.covariance_min_eigenvalue_history,
            covariance_symmetry_error_history=self.covariance_symmetry_error_history,
            process_covariance=self.R, measurement_covariance=self.Q,
        )
        colors = {"slam": "#c34d26", "truth": "#246b8e", "dead": "#6e7782"}
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                             "axes.spines.top": False, "axes.spines.right": False,
                             "axes.grid": True, "grid.alpha": 0.18})

        def save(fig, name):
            fig.savefig(output_dir / f"{name}.png", dpi=180, bbox_inches="tight")
            fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.4, 7.0), constrained_layout=True)
        if self.MGT is not None:
            ax.scatter(self.MGT[1], self.MGT[2], marker="x", s=45,
                       c=colors["truth"], label="Ground-truth landmarks", zorder=4)
            for column in self.MGT.T:
                ax.annotate(str(int(column[0])), column[1:3], xytext=(5, 5),
                            textcoords="offset points", fontsize=7, color=colors["truth"])
        for number, key in enumerate(sorted_ids):
            index = self.mapLUT[key]
            ax.scatter(*self.mu[index:index + 2], s=22, c=colors["slam"],
                       label="Estimated landmarks" if number == 0 else None, zorder=3)
            self._ellipse(ax, self.mu[index:index + 2], self.Sigma[index:index + 2, index:index + 2],
                          colors["slam"], label="3-sigma position contours" if number == 0 else None)
        ax.scatter(*self.mu[:2], marker="o", s=80, edgecolor="white", c=colors["slam"],
                   label="Final SLAM robot pose", zorder=5)
        self._ellipse(ax, self.mu[:2], self.Sigma[:2, :2], colors["slam"], linewidth=1.8)
        ax.quiver(self.mu[0], self.mu[1], np.cos(self.mu[2]), np.sin(self.mu[2]),
                  angles="xy", scale_units="xy", scale=1.8, color=colors["slam"], zorder=5)
        if truth.size:
            ax.scatter(*truth[:2, -1], marker="*", s=135, c=colors["truth"],
                       label="Final ground-truth robot pose", zorder=4)
            ax.quiver(truth[0, -1], truth[1, -1], np.cos(truth[2, -1]), np.sin(truth[2, -1]),
                      angles="xy", scale_units="xy", scale=1.8, color=colors["truth"], zorder=4)
        ax.set(xlabel="World x (m)", ylabel="World y (m)",
               title=f"{dataset_name}\nFinal robot pose and landmark map")
        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(0.12)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=8)
        save(fig, "final_map")

        fig, ax = plt.subplots(figsize=(8.8, 6.8), constrained_layout=True)
        ax.plot(*self.dead_reckoning_history[:2], color=colors["dead"], alpha=0.85,
                linewidth=1.15, label="Dead reckoning")
        if truth.size:
            ax.plot(*truth[:2], color=colors["truth"], linewidth=2.0,
                    label="Ground truth", alpha=0.8)
        ax.plot(*self.pose_history[:2], color=colors["slam"], linewidth=1.2,
                linestyle="--", label="EKF SLAM")
        ax.scatter(*self.initial_mu[:2], marker="s", s=35, c="#28303b", label="Start", zorder=5)
        ax.scatter(*self.pose_history[:2, -1], marker="o", s=40, c=colors["slam"], zorder=5)
        ax.scatter(*self.dead_reckoning_history[:2, -1], marker="o", s=40, c=colors["dead"], zorder=5)
        ax.set(xlabel="World x (m)", ylabel="World y (m)",
               title=f"{dataset_name}\nFull trajectory comparison (all {len(self.times)} poses)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="best")
        save(fig, "trajectories")

        if truth.size:
            fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True, constrained_layout=True)
            for name, history, label in (("slam", self.pose_history, "EKF SLAM"),
                                          ("dead", self.dead_reckoning_history, "Dead reckoning")):
                error = history - truth
                axes[0].plot(self.times, np.linalg.norm(error[:2], axis=0),
                             color=colors[name], label=label, linewidth=1.1)
                axes[1].plot(self.times, np.rad2deg(np.abs(self.angleWrap(error[2]))),
                             color=colors[name], label=label, linewidth=1.1)
            axes[0].set(ylabel="Position error (m)", title=f"{dataset_name}: timestamp-aligned errors")
            axes[1].set(xlabel="Time step", ylabel="Absolute heading error (deg)")
            axes[0].legend()
            save(fig, "pose_errors")
        return metrics
