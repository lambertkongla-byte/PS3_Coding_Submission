"""Independent finite-difference and analytic checks for joint EKF-SLAM."""

import unittest
import numpy as np

from tests.helpers import (ROOT, augmented_covariance, correlated_covariance,
                           finite_jacobian, inverse_model, measurement_model,
                           motion_model)
from EKFSLAM import EKFSLAM


class EKFSLAMTests(unittest.TestCase):
    def make_filter(self, state, covariance=None, process=None, sensor=None, **kwargs):
        state = np.asarray(state, dtype=float)
        covariance = correlated_covariance(state.size) if covariance is None else covariance
        process = np.array([[0.04, 0.012], [0.012, 0.01]]) if process is None else process
        sensor = np.array([[0.02, 0.003], [0.003, 0.04]]) if sensor is None else sensor
        return EKFSLAM(state.copy(), np.asarray(covariance).copy(),
                       np.asarray(process).copy(), np.asarray(sensor).copy(), **kwargs)

    def assert_covariance(self, covariance):
        covariance = np.asarray(covariance)
        self.assertTrue(np.isfinite(covariance).all())
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-11)
        self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), -1e-10)

    def test_prediction_matches_independent_model_and_numerical_jacobians(self):
        state = np.array([1.0, 2.0, 0.7, 4.0, 3.0, -2.0, 1.0])
        covariance = correlated_covariance(state.size)
        control = np.array([1.2, -0.4])
        ekf = self.make_filter(state, covariance)
        ekf.mapLUT = {"10": 3, "20": 5}
        state_jacobian = finite_jacobian(lambda value: motion_model(value, control), state)
        noise_jacobian = finite_jacobian(lambda value: motion_model(state, control, value), np.zeros(2))
        expected_covariance = (state_jacobian @ covariance @ state_jacobian.T
                               + noise_jacobian @ ekf.R @ noise_jacobian.T)
        ekf.prediction(control)
        np.testing.assert_allclose(np.asarray(ekf.mu).reshape(-1), motion_model(state, control), atol=1e-12)
        np.testing.assert_allclose(ekf.Sigma, expected_covariance, rtol=2e-8, atol=1e-10)
        np.testing.assert_allclose(ekf.Sigma[3:, 3:], covariance[3:, 3:], atol=1e-12)
        self.assert_covariance(ekf.Sigma)

    def test_process_covariance_includes_correlated_distance_and_heading_noise(self):
        state = np.array([1.0, 2.0, 0.4])
        ekf = self.make_filter(state, covariance=np.zeros((3, 3)))
        control = np.array([1.0, 0.2])
        noise_jacobian = finite_jacobian(lambda noise: motion_model(state, control, noise), np.zeros(2))
        expected = noise_jacobian @ ekf.R @ noise_jacobian.T
        ekf.prediction(control)
        np.testing.assert_allclose(ekf.Sigma, expected, rtol=2e-8, atol=1e-10)
        self.assertGreater(abs(ekf.Sigma[0, 2]), 0.005)

    def test_prediction_uses_old_heading_for_translation_and_wraps_new_heading(self):
        state = np.array([0.0, 0.0, np.pi - 0.1])
        ekf = self.make_filter(state, process=np.zeros((2, 2)))
        ekf.prediction(np.array([2.0, 0.5]))
        mean = np.asarray(ekf.mu).reshape(-1)
        np.testing.assert_allclose(mean[:2], [2 * np.cos(state[2]), 2 * np.sin(state[2])])
        self.assertGreaterEqual(mean[2], -np.pi)
        self.assertLessEqual(mean[2], np.pi)
        self.assertAlmostEqual(np.sin(mean[2]), np.sin(state[2] + 0.5), places=12)

    def test_known_landmark_update_matches_numerical_measurement_jacobian(self):
        state = np.array([1.0, 2.0, 0.4, 4.0, 3.0, -2.0, 1.0])
        covariance = correlated_covariance(state.size)
        ekf = self.make_filter(state, covariance)
        ekf.mapLUT = {"17": 3, "29": 5}
        prediction = measurement_model(state, 3)
        measurement = prediction + np.array([0.2, -0.15])
        jacobian = finite_jacobian(lambda value: measurement_model(value, 3), state)
        innovation_covariance = jacobian @ covariance @ jacobian.T + ekf.Q
        gain = np.linalg.solve(innovation_covariance, jacobian @ covariance).T
        expected_mean = state + gain @ (measurement - prediction)
        remaining = np.eye(state.size) - gain @ jacobian
        expected_covariance = remaining @ covariance @ remaining.T + gain @ ekf.Q @ gain.T
        ekf.update(measurement, 17)
        np.testing.assert_allclose(np.asarray(ekf.mu).reshape(-1), expected_mean, rtol=2e-8, atol=1e-10)
        np.testing.assert_allclose(ekf.Sigma, expected_covariance, rtol=2e-8, atol=1e-10)
        self.assertGreater(np.linalg.norm(expected_mean[5:] - state[5:]), 1e-5)
        self.assert_covariance(ekf.Sigma)

    def test_zero_innovation_changes_covariance_but_not_mean(self):
        state = np.array([1.0, 2.0, 0.4, 4.0, 3.0])
        ekf = self.make_filter(state)
        ekf.mapLUT = {"17": 3}
        prior_covariance = ekf.Sigma.copy()
        ekf.update(measurement_model(state, 3), 17)
        np.testing.assert_allclose(np.asarray(ekf.mu).reshape(-1), state, atol=1e-12)
        self.assertLess(np.trace(ekf.Sigma), np.trace(prior_covariance))
        self.assertGreaterEqual(np.linalg.eigvalsh(prior_covariance - ekf.Sigma).min(), -1e-10)
        self.assert_covariance(ekf.Sigma)

    def test_landmark_augmentation_preserves_full_existing_cross_covariances(self):
        state = np.array([2.0, 1.0, 0.7, 5.0, 4.0])
        covariance = correlated_covariance(state.size)
        measurement = np.array([1.2, -0.8])
        ekf = self.make_filter(state, covariance)
        ekf.mapLUT = {"17": 3}
        expected_covariance = augmented_covariance(state, covariance, measurement, ekf.Q)
        expected_mean = np.concatenate((state, inverse_model(state, measurement)))
        ekf.augmentState(measurement, 42)
        np.testing.assert_allclose(np.asarray(ekf.mu).reshape(-1), expected_mean, atol=1e-12)
        np.testing.assert_allclose(ekf.Sigma, expected_covariance, rtol=2e-8, atol=1e-10)
        np.testing.assert_allclose(ekf.Sigma[:5, :5], covariance, atol=1e-12)
        self.assertEqual(ekf.mapLUT["17"], 3)
        self.assertEqual(ekf.mapLUT["42"], 5)
        self.assertGreater(np.linalg.norm(ekf.Sigma[3:5, 5:]), 0.001)
        self.assert_covariance(ekf.Sigma)

    def test_inverse_initialization_and_forward_measurement_round_trip(self):
        state = np.array([2.0, 1.0, np.pi / 2])
        measurement = np.array([3.0, -1.0])
        ekf = self.make_filter(state)
        ekf.augmentState(measurement, 8)
        np.testing.assert_allclose(np.asarray(ekf.mu).reshape(-1)[3:], [3.0, 4.0], atol=1e-12)
        np.testing.assert_allclose(measurement_model(np.asarray(ekf.mu).reshape(-1), 3), measurement,
                                   atol=1e-12)

    def test_run_does_not_reuse_the_first_measurement_after_augmentation(self):
        state = np.array([0.0, 0.0, 0.2])
        covariance = np.diag([0.04, 0.04, 0.01])
        measurement = np.array([2.0, 1.0])
        ekf = self.make_filter(state, covariance, process=np.zeros((2, 2)))
        expected_covariance = augmented_covariance(state, covariance, measurement, ekf.Q)
        controls = np.array([[1.0], [0.0], [0.0]])
        observations = np.array([[1.0], [42.0], [measurement[0]], [measurement[1]]])
        ekf.run(controls, observations)
        np.testing.assert_allclose(ekf.Sigma, expected_covariance, rtol=2e-8, atol=1e-10)
        np.testing.assert_allclose(np.asarray(ekf.mu).reshape(-1)[:3], state, atol=1e-12)
        self.assertEqual(np.asarray(ekf.mu).size, 5)

    def test_run_predicts_before_observations_at_control_time(self):
        state = np.array([0.0, 0.0, 0.0])
        ekf = self.make_filter(state, process=np.zeros((2, 2)))
        controls = np.array([[1.0], [1.0], [np.pi / 2]])
        observations = np.array([[1.0], [42.0], [2.0], [0.0]])
        ekf.run(controls, observations)
        np.testing.assert_allclose(np.asarray(ekf.mu).reshape(-1), [1, 0, np.pi / 2, 1, 2], atol=1e-12)
        np.testing.assert_allclose(ekf.times, [0, 1])
        self.assertEqual(np.asarray(ekf.pose_history).shape, (3, 2))
        np.testing.assert_allclose(ekf.pose_history[:, 0], state, atol=1e-12)
        np.testing.assert_allclose(ekf.pose_history[:, 1], [1, 0, np.pi / 2], atol=1e-12)

    def test_run_processes_observations_at_initial_time(self):
        state = np.array([1.0, 0.0, 0.0])
        ekf = self.make_filter(state, process=np.zeros((2, 2)))
        controls = np.array([[1.0], [0.0], [0.0]])
        observations = np.array([[0.0], [42.0], [2.0], [1.0]])
        ekf.run(controls, observations)
        self.assertEqual(np.asarray(ekf.mu).size, 5)
        np.testing.assert_allclose(np.asarray(ekf.mu).reshape(-1)[3:], [3, 1], atol=1e-12)

    def test_ground_truth_does_not_change_filter_inference(self):
        state = np.array([0.0, 0.0, 0.0])
        controls = np.array([[1.0, 2.0], [1.0, 0.5], [0.1, -0.05]])
        observations = np.array([[1.0, 2.0], [42.0, 42.0], [2.0, 1.5], [1.0, 1.0]])
        baseline = self.make_filter(state)
        fake_trajectory = np.array([[0, 1, 2], [500, 600, 700], [800, 900, 1000], [0, 0, 0]], dtype=float)
        fake_map = np.array([[42], [900], [800]], dtype=float)
        decorated = self.make_filter(state, XGT=fake_trajectory, MGT=fake_map)
        baseline.run(controls, observations)
        decorated.run(controls, observations)
        np.testing.assert_allclose(decorated.mu, baseline.mu, atol=1e-12)
        np.testing.assert_allclose(decorated.Sigma, baseline.Sigma, atol=1e-12)
        np.testing.assert_allclose(decorated.pose_history, baseline.pose_history, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
