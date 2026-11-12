# Useful functions (* denotes files you should expect to modify)

COMPLETED IMPLEMENTATION - QUESTION 3(b)
=======================================

Requirements: Python 3, NumPy, and Matplotlib. SciPy is not required here.

From this directory, the original command remains supported:

    python RunEKFSLAM.py ../../data/ekf-slam/ekf-slam-small-noise.pickle

It runs without opening a window and saves results/<dataset-name>/ locally.
To reproduce the packaged outputs in the submission's results/ekf directory:

    python RunEKFSLAM.py ../../data/ekf-slam/ekf-slam-small-noise.pickle --output-dir ../../results/ekf/ekf-slam-small-noise
    python RunEKFSLAM.py ../../data/ekf-slam/ekf-slam-large-noise.pickle --output-dir ../../results/ekf/ekf-slam-large-noise

Add --animate to display live filter rendering. An interactive Matplotlib
backend is needed for animation; the default headless run uses Agg.

Each output directory contains:
    final_map.png / .pdf       Final robot and map, 3-sigma position covariance
                              contours, and ground-truth robot/map positions.
    trajectories.png / .pdf    Full EKF SLAM, dead-reckoning, and truth paths.
    pose_errors.png / .pdf     Timestamp-aligned position and heading errors.
    metrics.json              RMSE, final error, noise matrices, and checks.
    results.npz               Histories, final joint state/covariance, landmark
                              ID-to-state indices, aligned truth, innovations.

Algorithm and assumptions:
    - Initialize from the supplied X0, never from XGT. Ground truth is used
      only for evaluation and plotting.
    - Retain the provided initial covariance diag(0.01, 0.01, (0.5 deg)^2),
      with the last entry converted to radians squared.
    - Translate using the OLD heading, as specified by Eq. (5). Process
      covariance enters through V R V^T, including off-diagonal entries of R.
    - Predict the robot-map cross covariance as well as robot covariance.
    - Use the Cartesian body-frame observation model in Eq. (6), a linear
      solve for the Kalman gain, and the Joseph covariance update.
    - Augment a new landmark once with its pose-induced covariance and all
      existing-state cross covariances. Its initializing observation is not
      processed again as a known-landmark measurement.
    - Match each Z column to U using its actual time, and retain time 0 before
      all 847 controls. The 848 output poses are matched to XGT by timestamp,
      not by assuming that control index zero is ground-truth index zero.
    - Headings are wrapped to [-pi, pi). Landmark IDs map to state indices
      through mapLUT, retaining the starter renderer's string-key convention.
    - Position RMSE is sqrt(mean(dx^2 + dy^2)); map errors match landmark IDs.
      There is no post-hoc rigid alignment to the ground-truth trajectory/map.
    - No noise parameters or estimates are tuned using ground truth.

Important dataset detail: despite the filenames, the large-noise dataset
has larger PROCESS variance and smaller MEASUREMENT variance. The exact R
and Q supplied by each dataset are retained and recorded in metrics.json.

The full covariance is checked for finiteness, symmetry, and its minimum
eigenvalue after every recorded posterior. The original Renderer.py also has
small plotting fixes: symmetric covariance eigendecomposition, independent
observation rays, and support for omitted ground truth. Bounds in the new
constructor use the full minimum/maximum extents rather than clipping the map.

ORIGINAL FUNCTION GUIDE
=======================

EKFSLAM.py:

    *augmentState: Add new landmarks to state by updating the
                   mean vector and covariance matrix

    *prediction: Implements the EKF prediction step

    *update: Performs the EKF update step

    *run:  The EKF "main" function


Renderer.py:

    drawMap: Plots the ground-truth landmark positions

    render: Plots the current estimates of the robot pose and landmark
            positions along with ellipses that visualize uncertainty as
            level sets, the ground-truth pose, and lines from the robot
            to the landmarks observed at the current time step

    drawTrajectory: Plots the estimated and ground-truth robot trajectories
