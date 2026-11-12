"""Small independent models used as numerical test oracles."""

import os
from pathlib import Path
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "code" / "pf", ROOT / "code" / "ekf-slam"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def finite_jacobian(function, argument, step=1e-6):
    """Central differences; independent of the filters' analytic Jacobians."""
    argument = np.asarray(argument, dtype=float).reshape(-1)
    columns = []
    for index in range(argument.size):
        offset = np.zeros_like(argument)
        offset[index] = step
        columns.append(
            (np.asarray(function(argument + offset)).reshape(-1)
             - np.asarray(function(argument - offset)).reshape(-1)) / (2 * step)
        )
    return np.column_stack(columns)


def motion_model(state, control, noise=None):
    """Assignment Eq. (5), including the two-dimensional process noise."""
    state = np.asarray(state, dtype=float)
    result = state.copy()
    distance, heading_change = np.asarray(control, dtype=float).reshape(2)
    noise = np.zeros(2) if noise is None else np.asarray(noise, dtype=float)
    result[0] += (distance + noise[0]) * np.cos(state[2])
    result[1] += (distance + noise[0]) * np.sin(state[2])
    result[2] += heading_change + noise[1]
    return result


def measurement_model(state, landmark_index):
    """Assignment Eq. (6) in scalar form, without using filter helpers."""
    state = np.asarray(state, dtype=float)
    dx, dy = state[landmark_index:landmark_index + 2] - state[:2]
    cosine, sine = np.cos(state[2]), np.sin(state[2])
    return np.array([cosine * dx + sine * dy, -sine * dx + cosine * dy])


def inverse_model(state, measurement):
    state = np.asarray(state, dtype=float)
    zx, zy = np.asarray(measurement, dtype=float).reshape(2)
    cosine, sine = np.cos(state[2]), np.sin(state[2])
    return np.array([state[0] + cosine * zx - sine * zy,
                     state[1] + sine * zx + cosine * zy])


def correlated_covariance(dimension):
    """A reproducible positive-definite covariance with cross correlations."""
    generator = np.random.default_rng(1234 + dimension)
    factor = generator.normal(size=(dimension, dimension))
    return 0.003 * (factor @ factor.T) + 0.01 * np.eye(dimension)


def augmented_covariance(state, covariance, measurement, sensor_covariance):
    state_jacobian = finite_jacobian(lambda value: inverse_model(value, measurement), state)
    sensor_jacobian = finite_jacobian(lambda value: inverse_model(state, value), measurement)
    cross = covariance @ state_jacobian.T
    landmark = (state_jacobian @ covariance @ state_jacobian.T
                + sensor_jacobian @ sensor_covariance @ sensor_jacobian.T)
    return np.block([[covariance, cross], [cross.T, landmark]])
