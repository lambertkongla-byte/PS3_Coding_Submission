PS3 QUESTION 2: PARTICLE-FILTER LOCALIZATION
==========================================

This directory implements the supplied particle-filter starter interfaces.
The algorithm samples the velocity motion model (Probabilistic Robotics,
Table 5.3), weights particles using the independent-beam mixture model
(Table 6.1), and optionally performs systematic resampling.

1. INSTALLATION AND REPRODUCIBILITY
---------------------------------

From the PS3_Coding_Submission directory:

    python -m pip install -r requirements.txt

The numerical dependencies are NumPy, SciPy, and Matplotlib. No Numba, ROS,
OpenCV, or additional ray-casting package is required. Python 3.10 or later
is recommended. The recorded runs were tested with:

    Python       3.12.14
    NumPy        2.3.5
    SciPy        1.17.0
    Matplotlib   3.10.8

The lower bounds in requirements.txt describe supported API requirements;
every combination at those lower bounds has not been separately tested.
Use the tested versions, identical input data, and identical parameters for
the closest reproduction. Seeds make runs repeatable in the same software
environment; timings and byte-identical figures are not portable guarantees.

The CLI uses a deterministic seed of 0 by default. The recorded experiment
manifests explicitly use seed 7 unless a run specifies another seed.
Each PF instance uses its own NumPy Generator; it does not use global random
state. The optional experiment runner records command arguments and source
hashes in execution.json. Its absolute command paths are historical records;
rerun the manifest using the packaged script to resolve your local paths.

Only load the course-provided or otherwise trusted pickle files.

2. COMMAND-LINE USE
-------------------

Run these examples from the PS3_Coding_Submission directory:

    python code/pf/RunPF.py data/pf/simple_world_path1.pickle

    python code/pf/RunPF.py data/pf/loops_world_path1.pickle \
        --numParticles 1000 --sparsity 5 --seed 7 \
        --resampling adaptive --neff-threshold 0.6666666666666666 \
        --output-dir results/my_adaptive_run

    python code/pf/RunPF.py data/pf/rooms_world_path2.pickle \
        --numParticles 5000 --init uniform --seed 7 \
        --output-dir results/my_global_run

The assignment's original syntax also works when running from code/pf:

    python RunPF.py [--numParticles N] [--sparsity M] [-a] pickle_file

Options and defaults:

    --numParticles N       Number of particles; default 1000.
    --sparsity M           Retain every Mth LIDAR beam; default 5.
                           The supplied 41 beams become 9 beams at M=5.
    --seed S               Integer random seed; default 0.
    --resampling STRATEGY  always, never, or adaptive; default always.
    --neff-threshold F     Adaptive threshold as a fraction of N;
                           default 2/3, with 0 < F <= 1.
    --init METHOD          auto, gaussian, or uniform; default auto.
    --output-dir PATH      Output directory; default results.
    --max-steps K          Optional prefix length for a short smoke test.
                           Omit this option for the full assignment data.
    -a, --animate          Display the filter while running. This requires
                           an available interactive Matplotlib backend.

Without -a, the driver selects the Agg backend and still saves a final PNG.
The auto initialization is Gaussian xy when X0 is present and uniform free
space otherwise. Gaussian xy has standard deviation 0.5 m in each axis and
rejects occupied cells. Its heading is UNIFORM in [-pi,pi), as in the starter
sampleParticlesGaussian interface; it does not use X0's heading as a Gaussian
heading prior. Uniform initialization samples continuous free-cell area and
heading uniformly. Explicit --init gaussian requires X0.

Use a distinct output directory when comparing settings not included in the
filename, such as different adaptive thresholds. Repeating the same named
run in one directory overwrites its outputs.

3. NUMERICAL MODEL AND DEFAULTS
------------------------------

Particle poses are column vectors [x,y,theta]. Motion noise variances are

    var(v_noise)     = alpha1*v^2 + alpha2*omega^2
    var(omega_noise) = alpha3*v^2 + alpha4*omega^2
    var(gamma)       = alpha5*v^2 + alpha6*omega^2

Normal samples use the square roots of these variances. A sinc formulation
of the circular-arc displacement gives the exact straight-motion limit when
omega is zero. Headings are wrapped to [-pi,pi). A proposed occupied endpoint
is rejected, with at most ten attempts per particle. If all attempts collide,
that particle retains its previous pose, following the starter's suggestion.

The beam model defaults retained from Laser.py are:

    pHit = 0.97       pShort = 0.01     pMax = 0.01       pRand = 0.01
    sigmaHit = 0.5 m                  lambdaShort = 1 per meter
    zMax = 20 m                      zMaxEps = 0.1 m

These differ from the optional example parameter sets in the original
starter README. Every saved metrics JSON records the actual parameters.

The Gaussian hit density is normalized on [0,zMax]. The short-return
exponential is normalized on [0,z_expected]. The maximum-range component
is the Table 6.1 unit point mass; zMaxEps is its numerical saturation
tolerance. The random component is uniform on [0,zMax). For the optional
sigmaHit=0 case, a normalized resolution bin approximates a delta density.
The recorded baseline experiments use sigmaHit=0.5, not this approximation.

Exact grid-boundary traversal predicts beam ranges in physical units. It
stops at the first occupied cell, map boundary, or zMax. It supports distinct
x and y cell sizes. The supplied data use one-meter cells. The stored image
is flipped vertically once so world y increases upward.

The full product of beam likelihoods is evaluated as a sum of logarithms.
There is no likelihood tempering. NaN/infinite beams are treated as missing;
finite ranges outside [0,zMax] are impossible under the model. The supplied
datasets have finite ranges in [0,20]. If all posterior weights are impossible,
the scan is skipped while retaining the prior weights, and the event is counted
in impossible_scans_skipped. Such events should be inspected, not interpreted
as successful measurement updates.

The recursive update is log(w_t) = log(w_(t-1)) + log(p(z_t|x_t)), followed by
log-sum-exp normalization. Stored log weights remain available even if their
ordinary floating-point weights underflow. Resampling alone resets weights
to 1/N. Adaptive resampling occurs when ESS = 1/sum(w_i^2) < F*N.

4. PUBLIC INTERFACES
-------------------

PF.py:

    PF(numParticles, Alpha, laser, gridmap, visualize=True,
       seed=0, resampling='always', neff_threshold=2/3,
       init='auto', gaussian_sigma=0.5)

    particles                         Shape (3,N).
    weights                           Shape (1,N), normalized.
    log_weights                       Shape (N), normalized in log space.
    sampleParticlesUniform()          Initialize uniformly in free space.
    sampleParticlesGaussian(x0,y0,s)   Initialize Gaussian xy, uniform heading.
    getParticle(k)                    Return pose (3,) and weight (1,).
    getNormalizedWeights()            Return normalized weights (1,N).
    getMean()                         Return (x,y,circular heading), shape (3,).
    effectiveSampleSize()             Return 1/sum(w_i^2).
    sampleMotion(u,dt,particles=None)  Draw motion proposals; no collision test.
    prediction(u,dt)                  Propose with bounded collision rejection.
    update(ranges)                    Apply recursive log-likelihood weighting.
    resample()                        Systematic sampling; return ancestor indices.
    run(U,Ranges,dt,X0,XGT,filename)   Run and save outputs; return metrics dict.

Laser.py:

    Laser(numBeams=41, sparsity=1)
    scanProbability(z,x,map)          Ordinary scan likelihood. May underflow.
    scanLogProbability(z,x,map)       Stable scan log likelihood used by PF.
    rayTracing(xr,yr,theta,angles,map) Exact expected ranges and endpoints.
    getXY(ranges,bearings)            Sensor-frame Cartesian coordinates (2,B).

For scan methods, x=(3,) returns a scalar and x=(3,N) returns shape (N,).
rayTracing accepts scalar or vector poses and returns ranges (N,B) and
physical-coordinate endpoints (N,B,2). Ranges passed to a scan must already
be downsampled to match laser.Angles; RunPF.py performs both consistently.

Gridmap.py exposes the original inCollision, getShape, ij2xy, and xy2ij
interfaces. Out-of-bounds or nonfinite positions count as occupied. The
deprecated NumPy bool/float aliases were removed. Visualization.py retains
the starter's animation helpers; the final static figure is saved by PF.py.

5. OUTPUT FILES AND THEIR INTERPRETATION
--------------------------------------

Each run saves a PNG, NPZ, and JSON with this common prefix:

    <dataset>_N<N>_<strategy>_s<M>_seed<S>_<resolved-init>

Here s<M> is BEAM SPARSITY, while seed<S> is the random seed. A shortened
run appends _steps<K>. Experiment-run directory names may use a different
run_id convention; consult execution.json instead of inferring all parameters
from a directory name.

PNG:

    Shows the FINAL WEIGHTED POSTERIOR BEFORE the last optional resampling.
    It uses posterior_particles and posterior_weights, not final_particles.
    Dot area is 3 + 65*w_i/max(w), with fixed opacity. The positive base area
    keeps tiny or zero-weight particles visible. Thus dot area is not exactly
    proportional to probability, and the figure is not a probability-density
    heat map. Use the NPZ weights and numerical metrics for quantitative checks.
    The orange line joins posterior means; in a multimodal distribution these
    means can lie in a wall or jump between modes. It is an estimate history,
    not a claim that a physical particle followed that path. Ground truth is
    overlaid only when the dataset provides it.

NPZ (load with numpy.load(path, allow_pickle=False)):

    final_particles          (3,N), state AFTER any final resampling.
    final_weights            (N,), weights after any final resampling.
                             They equal 1/N if the last step resampled.
    posterior_particles      (3,N), weighted state BEFORE final resampling.
    posterior_weights        (N,), matching normalized posterior weights.
    means / estimated_poses  Both (3,T+1), identical arrays. Column 0 is the
                             initial sample mean; later columns are weighted
                             posterior means before resampling.
    neff_before_resampling   (T,), ESS after each measurement update.
    resampled                (T,), Boolean resampling decisions.
    collision_rejections     (T,), number of rejected proposals per timestep.
    collision_fallbacks      (T,), particles held still after ten failures.
    XGT                      (3,T+1) when supplied, otherwise empty (3,0).
    position_errors          (T,) Euclidean errors when truth exists; else empty.
    heading_errors_rad       (T,) wrapped heading errors; else empty.
    time                     (T+1,), timestamps starting at zero.
    seed, strategy, init,
    sigma, sparsity           Scalar metadata; init is the resolved method.
    metadata_json            Scalar JSON string, matching the JSON sidecar.

JSON:

    Contains all chosen parameters, sensor parameters, Alpha, ESS statistics,
    resampling and collision counts, final mean and weighted xy covariance,
    and timing. final_position_spread_rms is sqrt(trace(weighted xy covariance))
    of the final pre-resampling posterior. It measures concentration, not error.
    final_scan_mean_pose_log_likelihood and final_scan_best_particle_log_likelihood
    are sensor-consistency diagnostics, not ground-truth accuracy. The mean-pose
    value is null when that mean has zero likelihood, for example inside a wall.

    When XGT exists, position_rmse, final_position_error, and
    tail20_position_rmse use post-initialization estimates only. The tail starts
    at floor(0.8*T). Heading errors use wrapped angles and are reported in radians.
    These accuracy fields are omitted when ground truth is unavailable.

    inference_seconds includes initialization, filtering, and any requested
    animation. total_seconds extends through final plotting but precedes NPZ/JSON
    serialization. The experiment runner's wall_seconds includes subprocess
    startup and all output writing. Timings vary with hardware and concurrency.

6. DATA ALIGNMENT AND LIMITATIONS OF THE EXPERIMENTS
--------------------------------------------------

U has shape (2,T), original Ranges is (41,T), X0 is (3,1) or None, and XGT is
(3,T+1) or None. The first XGT column is the initial pose. Input and scan column
k produce means[:,k+1], which is compared with XGT[:,k+1], not XGT[:,k].

XGT is not used to select particles, compute weights, drive motion, or choose
resampling. X0 supplies only the allowed initialization information. Parameter
selection based on the reported experiments is an experimental comparison,
not online access to ground truth by the inference algorithm.

simple_world_path2, loops_world_path2, and rooms_world_path2 provide neither X0
nor XGT. On these datasets, a narrow cloud, a large likelihood, or agreement
between seeds cannot prove correct localization. Symmetric map regions can
produce confidently wrong modes. Report these as concentration/consistency
observations, and distinguish them from measured accuracy on the five datasets
that provide ground truth. A posterior mean is also a limited summary when
several localization modes remain plausible.

The supplied tests exercise the motion equations and variances, collision
handling, circular means, recursive weighting, ESS/resampling, beam components,
underflow handling, and exact ray geometry. From the submission root:

    python -m unittest discover -s tests -v
