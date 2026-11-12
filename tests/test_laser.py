"""Exact geometry and normalized beam-mixture tests on tiny synthetic maps."""

import math
import unittest
import numpy as np

from tests.helpers import ROOT
from Gridmap import Gridmap
from Laser import Laser


def enclosed_map(xres=1.0, yres=1.0):
    occupancy = np.zeros((7, 7), dtype=np.uint8)
    occupancy[[0, -1], :] = 1
    occupancy[:, [0, -1]] = 1
    return Gridmap(occupancy, xres=xres, yres=yres)


def normal_cdf(value):
    return 0.5 * (1 + math.erf(value / math.sqrt(2)))


class LaserTests(unittest.TestCase):
    def setUp(self):
        self.grid = enclosed_map()
        self.pose = np.array([3.5, 3.5, 0.0])
        self.laser = Laser(numBeams=1)
        self.laser.Angles = np.array([0.0])

    def set_component(self, name):
        for component in ("pHit", "pShort", "pMax", "pRand"):
            setattr(self.laser, component, float(component == name))

    def probability(self, value, pose=None):
        result = self.laser.scanProbability(np.array([value]), self.pose if pose is None else pose,
                                            self.grid)
        return float(np.asarray(result).reshape(-1)[0])

    def test_cardinal_rays_return_exact_cell_boundaries(self):
        bearings = np.array([0, np.pi / 2, np.pi, -np.pi / 2])
        distances, points = self.laser.rayTracing(3.5, 3.5, 0.0, bearings, self.grid)
        np.testing.assert_allclose(distances, [[2.5, 2.5, 2.5, 2.5]], atol=1e-10)
        expected = [[[6, 3.5], [3.5, 6], [1, 3.5], [3.5, 1]]]
        np.testing.assert_allclose(points, expected, atol=1e-10)

    def test_rays_respect_robot_heading(self):
        distances, points = self.laser.rayTracing(2.5, 3.5, np.pi / 2,
                                                  np.array([0.0]), self.grid)
        np.testing.assert_allclose(distances, [[2.5]], atol=1e-10)
        np.testing.assert_allclose(points, [[[2.5, 6.0]]], atol=1e-10)

    def test_anisotropic_map_resolution_uses_physical_distance(self):
        bearings = np.array([0, np.pi / 2, np.pi, -np.pi / 2])
        grid = enclosed_map(xres=2.0, yres=0.5)
        distances, points = self.laser.rayTracing(7.0, 1.75, 0.0, bearings, grid)
        np.testing.assert_allclose(distances, [[5.0, 1.25, 5.0, 1.25]], atol=1e-10)
        expected = [[[12, 1.75], [7, 3], [2, 1.75], [7, 0.5]]]
        np.testing.assert_allclose(points, expected, atol=1e-10)

    def test_default_mixture_matches_independent_scalar_calculation(self):
        measured, expected = 2.0, 2.5
        sigma = self.laser.sigmaHit
        hit = math.exp(-0.5 * ((measured - expected) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
        hit /= normal_cdf((self.laser.zMax - expected) / sigma) - normal_cdf(-expected / sigma)
        rate = self.laser.lambdaShort
        short = rate * math.exp(-rate * measured) / (-math.expm1(-rate * expected))
        truth = self.laser.pHit * hit + self.laser.pShort * short + self.laser.pRand / self.laser.zMax
        self.assertAlmostEqual(self.probability(measured), truth, places=11)

    def test_hit_is_normalized_when_expected_range_is_close_to_zero(self):
        self.set_component("pHit")
        pose = np.array([5.95, 3.5, 0.0])
        measured, expected, sigma = 0.2, 0.05, self.laser.sigmaHit
        normalization = normal_cdf((self.laser.zMax - expected) / sigma) - normal_cdf(-expected / sigma)
        truth = math.exp(-0.5 * ((measured - expected) / sigma) ** 2)
        truth /= sigma * math.sqrt(2 * math.pi) * normalization
        self.assertAlmostEqual(self.probability(measured, pose), truth, places=10)

    def test_short_component_has_truncated_exponential_density(self):
        self.set_component("pShort")
        truth = math.exp(-1) / (1 - math.exp(-2.5))
        self.assertAlmostEqual(self.probability(1.0), truth, places=11)
        self.assertAlmostEqual(self.probability(3.0), 0.0, places=14)

    def test_random_component_support_and_density(self):
        self.set_component("pRand")
        self.assertAlmostEqual(self.probability(0.0), 1 / 20, places=12)
        self.assertAlmostEqual(self.probability(19.5), 1 / 20, places=12)
        self.assertAlmostEqual(self.probability(20.0), 0.0, places=14)

    def test_max_component_is_saturation_mass(self):
        self.set_component("pMax")
        self.assertAlmostEqual(self.probability(20.0), 1.0, places=12)
        self.assertAlmostEqual(self.probability(20.0 - self.laser.zMaxEps / 2), 1.0, places=12)
        self.assertAlmostEqual(self.probability(19.0), 0.0, places=14)

    def test_finite_out_of_support_ranges_have_zero_likelihood(self):
        self.assertAlmostEqual(self.probability(-0.1), 0.0, places=14)
        self.assertAlmostEqual(self.probability(self.laser.zMax + 1), 0.0, places=14)

    def test_multiple_beams_multiply_and_vectorized_particles_match(self):
        self.set_component("pRand")
        self.laser.Angles = np.array([0.0, np.pi / 2])
        particles = np.column_stack((self.pose, np.array([2.5, 2.5, 0.2])))
        likelihood = self.laser.scanProbability(np.array([1.0, 2.0]), particles, self.grid)
        np.testing.assert_allclose(np.asarray(likelihood).reshape(-1), [1 / 400, 1 / 400], atol=1e-15)
        logs = self.laser.scanLogProbability(np.array([1.0, 2.0]), particles, self.grid)
        np.testing.assert_allclose(np.asarray(logs).reshape(-1), [-2 * np.log(20)] * 2, atol=1e-12)

    def test_log_likelihood_survives_long_scan_underflow(self):
        self.set_component("pRand")
        count = 500
        self.laser.Angles = np.linspace(-np.pi, np.pi, count)
        log_probability = self.laser.scanLogProbability(np.full(count, 2.0), self.pose, self.grid)
        self.assertTrue(np.isfinite(log_probability))
        self.assertAlmostEqual(float(log_probability), -count * np.log(20), places=9)


if __name__ == "__main__":
    unittest.main()
