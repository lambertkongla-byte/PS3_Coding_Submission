"""Analytic PF checks; no assignment ground truth is used as filter input."""

import unittest
import numpy as np

from tests.helpers import ROOT  # Adds the original module directories to sys.path.
from PF import PF
from Gridmap import Gridmap


def room_map(size=30, blocked=None):
    world_grid = np.zeros((size, size), dtype=np.uint8)
    world_grid[[0, -1], :] = 1
    world_grid[:, [0, -1]] = 1
    for row, column in blocked or []:
        world_grid[row, column] = 1
    return Gridmap(world_grid[::-1, :])


class FixedLikelihood:
    def __init__(self, values=None, log_values=None):
        self.log_values = (np.asarray(log_values, dtype=float) if log_values is not None
                           else np.log(np.asarray(values, dtype=float)))

    def scanProbability(self, ranges, particles, gridmap):
        return np.exp(self.log_values)

    def scanLogProbability(self, ranges, particles, gridmap):
        return self.log_values.copy()


class ParticleFilterTests(unittest.TestCase):
    def make_filter(self, particles, likelihood=None, gridmap=None, alpha=None):
        particles = np.asarray(particles, dtype=float)
        if particles.ndim == 1:
            particles = particles.reshape(3, 1)
        count = particles.shape[1]
        pf = PF(count, np.zeros(6) if alpha is None else alpha,
                likelihood or FixedLikelihood(np.ones(count)),
                room_map() if gridmap is None else gridmap, visualize=False)
        pf.particles = particles.copy()
        pf.weights = np.full((1, count), 1.0 / count)
        return pf

    def test_straight_motion_zero_angular_velocity(self):
        headings = np.array([0.0, np.pi / 2, -np.pi / 2, 0.4])
        particles = np.vstack((np.full(4, 8.0), np.full(4, 8.0), headings))
        pf = self.make_filter(particles)
        pf.prediction(np.array([2.0, 0.0]), 0.5)
        expected = particles.copy()
        expected[0] += np.cos(headings)
        expected[1] += np.sin(headings)
        np.testing.assert_allclose(pf.particles, expected, atol=1e-12)
        self.assertTrue(np.isfinite(pf.particles).all())

    def test_quarter_circle_motion(self):
        pf = self.make_filter([5.0, 5.0, 0.0])
        pf.prediction(np.array([2.0, np.pi / 2]), 1.0)
        expected = [5 + 4 / np.pi, 5 + 4 / np.pi, np.pi / 2]
        np.testing.assert_allclose(pf.particles[:, 0], expected, atol=1e-12)

    def test_near_zero_turn_has_continuous_straight_limit(self):
        pf = self.make_filter([5.0, 5.0, 0.7])
        pf.prediction(np.array([1.2, 1e-12]), 2.0)
        expected = [5 + 2.4 * np.cos(0.7), 5 + 2.4 * np.sin(0.7), 0.7 + 2e-12]
        np.testing.assert_allclose(pf.particles[:, 0], expected, atol=1e-10)

    def test_heading_wraps_without_changing_direction(self):
        pf = self.make_filter([5.0, 5.0, 3.1])
        pf.prediction(np.array([0.0, 1.0]), 1.0)
        angle = pf.particles[2, 0]
        self.assertGreaterEqual(angle, -np.pi)
        self.assertLessEqual(angle, np.pi)
        self.assertAlmostEqual(np.sin(angle), np.sin(4.1), places=12)
        self.assertAlmostEqual(np.cos(angle), np.cos(4.1), places=12)

    def test_motion_noise_uses_variances_not_standard_deviations(self):
        # With no angular-velocity noise, x noise is exactly v1 * dt and
        # heading noise is exactly gamma * dt, making their moments analytic.
        np.random.seed(191)
        count = 20000
        particles = np.vstack((np.full(count, 50.0), np.full(count, 50.0),
                               np.zeros(count)))
        alpha = np.array([0.04, 0.0, 0.0, 0.0, 0.09, 0.0])
        pf = self.make_filter(particles, gridmap=room_map(100), alpha=alpha)
        pf.prediction(np.array([1.0, 0.0]), 0.5)
        forward_errors = pf.particles[0] - 50.5
        heading_errors = pf.particles[2]
        self.assertLess(abs(np.mean(forward_errors)), 0.005)
        self.assertLess(abs(np.mean(heading_errors)), 0.006)
        self.assertAlmostEqual(np.var(forward_errors), 0.04 * 0.5 ** 2, delta=0.0007)
        self.assertAlmostEqual(np.var(heading_errors), 0.09 * 0.5 ** 2, delta=0.0015)
        np.testing.assert_allclose(pf.particles[1], 50.0, atol=1e-12)

    def test_deterministic_collision_falls_back_to_previous_pose(self):
        original = np.array([5.5, 5.5, 0.0])
        pf = self.make_filter(original, gridmap=room_map(blocked=[(5, 7)]))
        pf.prediction(np.array([2.0, 0.0]), 1.0)
        np.testing.assert_array_equal(pf.particles[:, 0], original)
        self.assertFalse(bool(pf.gridmap.inCollision(*pf.particles[:2, 0])))

    def test_prediction_preserves_importance_weights(self):
        pf = self.make_filter([[5, 7], [5, 7], [0, 0]])
        pf.weights = np.array([[0.8, 0.2]])
        pf.prediction(np.array([0.1, 0.0]), 1.0)
        np.testing.assert_allclose(pf.getNormalizedWeights().ravel(), [0.8, 0.2])

    def test_updates_multiply_previous_weights_when_not_resampling(self):
        pf = self.make_filter([[5, 7], [5, 7], [0, 0]], FixedLikelihood([1.0, 3.0]))
        pf.weights = np.array([[0.8, 0.2]])
        pf.update(np.array([1.0]))
        np.testing.assert_allclose(pf.getNormalizedWeights().ravel(), [4 / 7, 3 / 7])
        pf.update(np.array([1.0]))
        np.testing.assert_allclose(pf.getNormalizedWeights().ravel(), [4 / 13, 9 / 13])

    def test_log_update_retains_relative_weights_after_probability_underflow(self):
        pf = self.make_filter([[5, 7], [5, 7], [0, 0]],
                              FixedLikelihood(log_values=[-1000.0, -1001.0]))
        pf.update(np.array([1.0]))
        expected = np.array([1.0, np.exp(-1.0)]) / (1.0 + np.exp(-1.0))
        np.testing.assert_allclose(pf.getNormalizedWeights().ravel(), expected, atol=1e-13)

    def test_effective_sample_size_matches_definition(self):
        pf = self.make_filter([[5, 6, 7, 8], [5, 6, 7, 8], [0, 0, 0, 0]])
        self.assertAlmostEqual(pf.effectiveSampleSize(), 4.0)
        pf.weights = np.array([[0.7, 0.1, 0.1, 0.1]])
        self.assertAlmostEqual(pf.effectiveSampleSize(), 1.0 / 0.52)

    def test_resampling_copies_only_positive_weight_support_and_resets_weights(self):
        pf = self.make_filter([[5, 6, 7, 8], [4, 5, 6, 7], [0.1, 0.2, 0.3, 0.4]])
        survivor = pf.particles[:, 2].copy()
        pf.weights = np.array([[0.0, 0.0, 1.0, 0.0]])
        pf.resample()
        np.testing.assert_allclose(pf.particles, np.repeat(survivor[:, None], 4, axis=1))
        np.testing.assert_allclose(pf.weights.ravel(), np.full(4, 0.25))

    def test_orientation_mean_is_circular(self):
        pf = self.make_filter([[2, 8], [3, 7], [np.pi - 0.1, -np.pi + 0.1]])
        mean = np.asarray(pf.getMean()).reshape(-1)
        np.testing.assert_allclose(mean[:2], [5, 5])
        self.assertAlmostEqual(abs(mean[2]), np.pi, places=12)


if __name__ == "__main__":
    unittest.main()
