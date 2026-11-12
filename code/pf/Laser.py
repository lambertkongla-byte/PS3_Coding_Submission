"""Cartesian-grid ray casting and the four-component LIDAR beam model."""

import numpy as np
from scipy.special import ndtr


class Laser(object):
    """Beam model of Probabilistic Robotics, Table 6.1.

    ``pHit``, ``pShort``, ``pMax`` and ``pRand`` are mixture weights.
    Public poses have shape (3, N), and a scan has one range per retained beam.
    The starter's deliberately tolerant default sensor parameters are retained.
    """

    def __init__(self, numBeams=41, sparsity=1):
        if int(numBeams) < 1 or int(sparsity) < 1:
            raise ValueError("numBeams and sparsity must be positive integers")
        self.pHit = 0.97
        self.pShort = 0.01
        self.pMax = 0.01
        self.pRand = 0.01
        self.sigmaHit = 0.5
        self.lambdaShort = 1.0
        self.zMax = 20.0
        self.zMaxEps = 0.1
        self.sparsity = int(sparsity)
        self.Angles = np.linspace(-np.pi, np.pi, int(numBeams))[::self.sparsity]

    def scanProbability(self, z, x, gridmap):
        """Return the scan likelihood, keeping the original public interface.

        This product can underflow for long scans. The filter therefore calls
        ``scanLogProbability`` instead; densities may legitimately exceed one.
        A single pose (3,) returns a scalar; poses (3, N) return an array (N,).
        """
        return np.exp(self.scanLogProbability(z, x, gridmap))

    def scanLogProbability(self, z, x, gridmap):
        """Sum log mixture likelihoods of conditionally independent beams.

        The hit Gaussian is normalized on [0,zMax]; the short exponential is
        normalized on [0,z_expected]. The maximum-range term is the Table 6.1
        unit point mass, with zMaxEps used as its numerical detection tolerance.
        Missing (NaN or infinite) readings are ignored. Finite readings outside
        [0,zMax] have zero likelihood under this model.
        """
        poses = np.asarray(x, dtype=float)
        scalar_pose = poses.ndim == 1
        if scalar_pose:
            poses = poses.reshape(3, 1)
        if poses.ndim != 2 or poses.shape[0] != 3:
            raise ValueError("x must have shape (3,) or (3,N)")
        ranges = np.asarray(z, dtype=float).reshape(-1)
        if ranges.size != self.Angles.size:
            raise ValueError("A range is required for every retained LIDAR angle")
        mixture = np.array([self.pHit, self.pShort, self.pMax, self.pRand])
        if np.any(mixture < 0) or not np.isclose(mixture.sum(), 1.0):
            raise ValueError("The four nonnegative beam mixture weights must sum to 1")
        if self.zMax <= 0 or self.sigmaHit < 0 or self.lambdaShort <= 0:
            raise ValueError("Invalid LIDAR range, hit standard deviation, or short rate")

        valid = np.isfinite(ranges)
        if not np.any(valid):
            result = np.zeros(poses.shape[1])
            return float(result[0]) if scalar_pose else result
        observed = ranges[valid][None, :]
        expected, _ = self.rayTracing(
            poses[0], poses[1], poses[2], self.Angles[valid], gridmap)
        inside = (observed >= 0.0) & (observed <= self.zMax)
        probability = np.zeros_like(expected)

        if self.pHit > 0:
            if self.sigmaHit > 0:
                sigma = self.sigmaHit
                normalization = (ndtr((self.zMax - expected) / sigma)
                                 - ndtr(-expected / sigma))
                hit = (np.exp(-0.5 * ((observed - expected) / sigma) ** 2)
                       / (np.sqrt(2.0 * np.pi) * sigma * normalization))
            else:
                # A zero-noise ideal is a delta distribution. For the optional
                # deterministic model, represent it by a normalized resolution
                # bin rather than dividing by a zero standard deviation.
                eps = max(float(self.zMaxEps), 1e-9)
                low = np.maximum(0.0, expected - eps)
                high = np.minimum(self.zMax, expected + eps)
                hit = ((observed >= low) & (observed <= high)) / (high - low)
            probability += self.pHit * np.where(inside, hit, 0.0)

        if self.pShort > 0:
            normalization = -np.expm1(-self.lambdaShort * expected)
            short = np.zeros_like(expected)
            supported = inside & (observed <= expected) & (expected > 0.0)
            np.divide(self.lambdaShort * np.exp(-self.lambdaShort * observed),
                      normalization, out=short, where=supported)
            probability += self.pShort * short

        if self.pMax > 0:
            saturated = inside & (observed >= self.zMax - self.zMaxEps)
            probability += self.pMax * saturated
        if self.pRand > 0:
            probability += self.pRand * (inside & (observed < self.zMax)) / self.zMax

        log_beams = np.full_like(probability, -np.inf)
        np.log(probability, out=log_beams, where=probability > 0.0)
        result = log_beams.sum(axis=1)
        result[gridmap.inCollision(poses[0], poses[1])] = -np.inf
        return float(result[0]) if scalar_pose else result

    def getXY(self, range, bearing):
        """Convert sensor-frame polar coordinates to a (2,n) Cartesian array."""
        ranges = np.asarray(range).reshape(-1)
        bearings = np.asarray(bearing).reshape(-1)
        return np.vstack((np.cos(bearings), np.sin(bearings))) * ranges

    def rayTracing(self, xr, yr, thetar, lAngle, gridmap):
        """Exact cell-boundary ray traversal, vectorized over poses and beams.

        At each iteration, a ray crosses the nearest vertical or horizontal
        grid line (both at a corner). It stops on entering an occupied cell or
        leaving the map, or at zMax. No distance-step approximation is used.
        Ranges and returned (x,y) endpoints use physical map units, including
        when xres and yres differ. Results have shapes (N,B) and (N,B,2).
        """
        xr, yr, thetar = np.broadcast_arrays(
            np.atleast_1d(np.asarray(xr, dtype=float)),
            np.atleast_1d(np.asarray(yr, dtype=float)),
            np.atleast_1d(np.asarray(thetar, dtype=float)))
        xr, yr, thetar = xr.reshape(-1), yr.reshape(-1), thetar.reshape(-1)
        bearings = np.asarray(lAngle, dtype=float).reshape(-1)
        angle = thetar[:, None] + bearings[None, :]
        shape = angle.shape
        dx, dy = np.cos(angle).ravel(), np.sin(angle).ravel()
        dx[np.abs(dx) < 1e-14] = 0.0
        dy[np.abs(dy) < 1e-14] = 0.0
        x0 = np.broadcast_to(xr[:, None], shape).ravel()
        y0 = np.broadcast_to(yr[:, None], shape).ravel()
        if not (np.all(np.isfinite(x0)) and np.all(np.isfinite(y0))
                and np.all(np.isfinite(angle))):
            raise ValueError("Ray origins and directions must be finite")

        col = np.floor(x0 / gridmap.xres).astype(np.int64)
        row = np.floor(y0 / gridmap.yres).astype(np.int64)
        stepx, stepy = np.sign(dx).astype(np.int8), np.sign(dy).astype(np.int8)
        delta_x = np.full(dx.shape, np.inf)
        delta_y = np.full(dy.shape, np.inf)
        np.divide(gridmap.xres, np.abs(dx), out=delta_x, where=dx != 0.0)
        np.divide(gridmap.yres, np.abs(dy), out=delta_y, where=dy != 0.0)
        next_x = (col + (stepx > 0)) * gridmap.xres
        next_y = (row + (stepy > 0)) * gridmap.yres
        tmax_x = np.full(dx.shape, np.inf)
        tmax_y = np.full(dy.shape, np.inf)
        np.divide(next_x - x0, dx, out=tmax_x, where=dx != 0.0)
        np.divide(next_y - y0, dy, out=tmax_y, where=dy != 0.0)
        distance = np.full(dx.shape, self.zMax, dtype=float)
        blocked = gridmap.inCollision(col, row, True)
        distance[blocked] = 0.0
        active = np.flatnonzero(~blocked)

        # A monotone ray can cross at most m+n grid lines before leaving a map.
        for _ in range(gridmap.m + gridmap.n + 2):
            if active.size == 0:
                break
            tx, ty = tmax_x[active], tmax_y[active]
            next_distance = np.minimum(tx, ty)
            before_limit = next_distance < self.zMax
            active, tx, ty = active[before_limit], tx[before_limit], ty[before_limit]
            next_distance = next_distance[before_limit]
            if active.size == 0:
                break
            cross_x, cross_y = tx <= ty, ty <= tx
            ix, iy = active[cross_x], active[cross_y]
            col[ix] += stepx[ix]
            row[iy] += stepy[iy]
            tmax_x[ix] += delta_x[ix]
            tmax_y[iy] += delta_y[iy]
            hit = gridmap.inCollision(col[active], row[active], True)
            distance[active[hit]] = np.maximum(next_distance[hit], 0.0)
            active = active[~hit]
        else:
            if active.size:
                raise RuntimeError("Ray traversal exceeded the finite-grid bound")

        endpoints = np.stack((x0 + distance * dx, y0 + distance * dy), axis=-1)
        return distance.reshape(shape), endpoints.reshape(shape + (2,))
