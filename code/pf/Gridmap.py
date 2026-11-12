"""Occupancy-grid helpers, with physical coordinates measured from lower left."""

import numpy as np


class Gridmap(object):
    def __init__(self, occupancy, xres=1, yres=1):
        array = np.asarray(occupancy)
        if array.ndim != 2 or array.size == 0 or xres <= 0 or yres <= 0:
            raise ValueError("A nonempty 2-D grid and positive resolutions are required")
        self.occupancy = array[::-1, :].astype(bool)
        self.xres, self.yres = float(xres), float(yres)
        self.m, self.n = self.occupancy.shape

    def inCollision(self, x, y, ij=False):
        """Check scalar/array (x,y), or (column,row) if ij=True.

        Out-of-map and nonfinite positions are occupied. Array inputs broadcast.
        """
        x, y = np.broadcast_arrays(np.asarray(x), np.asarray(y))
        finite = np.isfinite(x) & np.isfinite(y)
        if ij:
            j = np.where(finite, x, -1).astype(np.int64)
            i = np.where(finite, y, -1).astype(np.int64)
        else:
            j = np.floor(np.where(finite, x / self.xres, -1)).astype(np.int64)
            i = np.floor(np.where(finite, y / self.yres, -1)).astype(np.int64)
        inside = finite & (i >= 0) & (i < self.m) & (j >= 0) & (j < self.n)
        collision = np.ones(i.shape, dtype=bool)
        collision[inside] = self.occupancy[i[inside], j[inside]]
        return collision

    def getShape(self):
        return self.m, self.n

    def ij2xy(self, i, j):
        """Return the lower-left corner of a cell in physical coordinates."""
        return np.asarray(j, dtype=float) * self.xres, np.asarray(i, dtype=float) * self.yres

    def xy2ij(self, x, y):
        return int(np.floor(y / self.yres)), int(np.floor(x / self.xres))
