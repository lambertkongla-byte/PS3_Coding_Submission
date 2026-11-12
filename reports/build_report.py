"""Build the experiment report from saved run outputs; no experiments are run."""

from pathlib import Path
import json
import math
from html import escape

from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "PS3_Coding_Report.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)
FONTDIR = Path("/usr/share/fonts/truetype/dejavu")
if (FONTDIR / "DejaVuSans.ttf").exists():
    pdfmetrics.registerFont(TTFont("Body", str(FONTDIR / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("BodyBold", str(FONTDIR / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold")
    FONT, BOLD = "Body", "BodyBold"
else:
    FONT, BOLD = "Helvetica", "Helvetica-Bold"

INK = colors.HexColor("#173044")
TEAL = colors.HexColor("#176878")
GREY = colors.HexColor("#52636e")
LIGHT = colors.HexColor("#eff4f7")
W, H, M = 612, 792, 40
BODYW = W - 2 * M
C = canvas.Canvas(str(OUT), pagesize=(W, H))
C.setTitle("PS3 Coding Report: Particle Filter Localization and EKF SLAM")
C.setAuthor("Prepared with ChatGPT assistance")
PAGE = 0
Y = 0


def para(text, size=9.5, color=INK, space=8, width=None, x=M):
    global Y
    style = ParagraphStyle("p", fontName=FONT, fontSize=size, leading=size * 1.38,
                           textColor=color, spaceAfter=space)
    block = Paragraph(text, style)
    _, height = block.wrap(width or BODYW, 700)
    block.drawOn(C, x, Y - height)
    Y -= height + space


def page(title, category):
    global PAGE, Y
    if PAGE:
        if Y < 45:
            raise ValueError(f"Page {PAGE} overflows: y={Y}")
        C.showPage()
    PAGE += 1
    C.setFont(FONT, 8)
    C.setFillColor(GREY)
    C.drawString(M, H - 27, "TTIC 31170  |  Problem Set 3")
    C.drawRightString(W - M, H - 27, "Coding experiments and verification")
    C.setStrokeColor(colors.HexColor("#c7d3da"))
    C.line(M, H - 34, W - M, H - 34)
    C.setFillColor(TEAL)
    C.setFont(BOLD, 8)
    C.drawString(M, H - 54, category.upper())
    Y = H - 69
    para(escape(title), size=16, space=14)
    C.setFillColor(GREY)
    C.setFont(FONT, 7.5)
    C.drawString(M, 23, "All numerical results are from saved, completed experiment runs.")
    C.drawRightString(W - M, 23, str(PAGE))


def table(headers, rows, widths=None, size=8.1):
    global Y
    widths = widths or [BODYW / len(headers)] * len(headers)
    style = ParagraphStyle("cell", fontName=FONT, fontSize=size, leading=size * 1.2,
                           textColor=INK)
    headstyle = ParagraphStyle("head", parent=style, fontName=BOLD, textColor=colors.white)
    values = [[Paragraph(escape(str(cell)), headstyle) for cell in headers]]
    values += [[Paragraph(escape(str(cell)), style) for cell in row] for row in rows]
    t = Table(values, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, TEAL),
    ]))
    _, height = t.wrap(BODYW, 700)
    t.drawOn(C, M, Y - height)
    Y -= height + 12


def images(paths, captions, height=350):
    global Y
    gap = 12
    slot = (BODYW - gap * (len(paths) - 1)) / len(paths)
    original_y = Y
    bottoms = []
    for i, (path, caption) in enumerate(zip(paths, captions)):
        x = M + i * (slot + gap)
        iw, ih = Image.open(path).size
        scale = min(slot / iw, height / ih)
        draw_w, draw_h = iw * scale, ih * scale
        C.drawImage(str(path), x + (slot - draw_w) / 2, original_y - draw_h,
                    draw_w, draw_h, preserveAspectRatio=True, mask="auto")
        style = ParagraphStyle("caption", fontName=FONT, fontSize=7.6, leading=10,
                               textColor=GREY)
        p = Paragraph(escape(caption), style)
        _, ph = p.wrap(slot, 100)
        p.drawOn(C, x, original_y - height - 5 - ph)
        bottoms.append(original_y - height - 5 - ph)
    Y = min(bottoms) - 13


def pf(dataset, count, strategy="always", seed=7):
    folder = ROOT / "results" / "pf" / f"{dataset}_N{count}_{strategy}_s{seed}"
    execution = folder / "execution.json"
    if not execution.exists():
        raise RuntimeError(f"Run is not complete: {folder.name}")
    status = json.loads(execution.read_text())
    if status.get("returncode") != 0:
        raise RuntimeError(f"Run did not succeed: {folder.name}")
    metrics_file = next(p for p in folder.glob("*.json") if p.name != "execution.json")
    result = json.loads(metrics_file.read_text())
    result["image"] = next(folder.glob("*.png"))
    result["folder"] = folder.relative_to(ROOT)
    return result


def num(value, places=3):
    return "n/a" if value is None else f"{value:.{places}f}"


KNOWN = ["simple_world_path1", "simple_world_path3", "loops_world_path0",
         "loops_world_path1", "rooms_world_path1"]
GLOBAL = ["simple_world_path2", "loops_world_path2", "rooms_world_path2"]
SELECTED = json.loads((ROOT / "experiments" / "selected_counts.json").read_text())
ROBUST = json.loads((ROOT / "experiments" / "robustness_counts.json").read_text())
FAILURES = {name: (1 if name == "loops_world_path1" else 10) for name in KNOWN}
known_notes = {
    "simple_world_path1": "The ten-particle run retains an incorrect heading hypothesis and stalls near the lower obstacle boundary. Its tiny final cloud is overconfident: the pose is far from ground truth. The 1000-particle run follows the observed path closely.",
    "simple_world_path3": "Ten particles produce a poor trajectory and heading estimate. The final position happens to be relatively near the endpoint, but its smaller final error does not erase the large trajectory RMSE. A sufficient cloud is needed throughout the trajectory.",
    "loops_world_path0": "The ten-particle run finishes in the wrong corridor. Its final effective sample size is almost ten even though its position error is large. ESS measures weight concentration among surviving particles, not correctness of their locations.",
    "loops_world_path1": "Ten particles actually work reasonably well in this particular run; they are not labeled a failure. The added one-particle experiment fails. With one particle, normalized weighting is always one and resampling cannot create a competing pose hypothesis.",
    "rooms_world_path1": "The ten-particle run follows an incorrect room/corridor hypothesis and repeatedly falls back after occupied motion proposals. The 1000-particle result remains close to the reference trajectory and ends near the true pose.",
}


page("Particle filter localization and EKF SLAM", "Implementation and experiment report")
para("This report covers the coding components of <b>Questions 2 and 3(b)</b>, including the required experiment figures and discussions. The separate written-solutions document covers Questions 1 and 3(a). Results here were obtained by running the supplied datasets; no synthetic results are presented as homework experiments.")
para("<b>Question 2(a): prediction.</b> Each particle samples the three independent Gaussian noise terms in the velocity model. The six supplied alpha coefficients define variances, so normal sampling uses their square roots. A sinc-based expression handles exactly zero angular velocity without division by zero. Headings wrap to [-pi, pi). Occupied endpoints are rejected; after ten failed proposals the particle retains its previous pose, following the starter's suggestion.")
para("<b>Question 2(b): observation model.</b> Exact grid traversal predicts beam ranges. The scan likelihood is the independent-beam product of the hit, short, maximum-range and random components. The hit Gaussian and short exponential are normalized on their permitted intervals. The maximum-range term is the textbook point mass with a numerical saturation tolerance. All likelihood arithmetic used by the filter is in log space.")
para("<b>Question 2(c): recursion.</b> Bootstrap importance weights multiply the previous weights by the new scan likelihood. This recurrence is retained when resampling is skipped. Systematic resampling draws N particles with replacement and resets weights to 1/N. Position is a weighted mean; orientation is a circular mean. ESS = 1 / sum(normalized_weight squared).")
table(["Setting", "Value in reported PF runs"], [
    ["Random seed / beam sparsity", "7 / every fifth beam (9 of 41 beams)"],
    ["Known-start initialization", "Gaussian xy, sigma = 0.5 m; uniform heading, as in starter"],
    ["Unknown-start initialization", "Uniform over free map area and [-pi, pi) heading"],
    ["Hit / short / max / random", "0.97 / 0.01 / 0.01 / 0.01"],
    ["Hit sigma / short rate / max range", "0.5 m / 1 per m / 20 m"],
    ["Adaptive resampling", "Resample when ESS < 2N/3"],
], [175, BODYW - 175], size=8.3)
para("<b>Evaluation.</b> PF input column k produces estimate k+1, matched to XGT column k+1. Ground truth is used only for evaluation and figures. RMSE is the square root of the mean squared Euclidean position error over the trajectory. Final cloud spread is the square root of the weighted xy covariance trace. Path2 files have no ground truth, so scan consistency and concentration cannot establish absolute localization accuracy.", size=8.7)

page("Particle-count sweep", "Questions 2(d) and 2(e)")
para("All entries use always-resampling and seed 7. Asterisks mark the selected baseline counts. These are experimentally chosen counts, not universal minimum requirements.", size=9)
rows = []
for dataset in KNOWN:
    counts = [10, 100, 1000]
    if dataset == "loops_world_path1":
        counts = [1] + counts
    for n in counts:
        d = pf(dataset, n)
        rows.append([dataset.replace("_world_", " "), str(n) + (" *" if n == 1000 else ""),
                     num(d["position_rmse"]), num(d["final_position_error"]), num(d["final_position_spread_rms"])])
table(["Known start", "N", "RMSE (m)", "Final error (m)", "Spread (m)"], rows,
      [160, 54, 99, 116, 103], size=7.7)
rows = []
for dataset in GLOBAL:
    for n in [250, 2000, 5000]:
        d = pf(dataset, n)
        rows.append([dataset.replace("_world_", " "), str(n) + (" *" if n == SELECTED[dataset] else ""),
                     num(d["final_scan_mean_pose_log_likelihood"]), num(d["final_position_spread_rms"]),
                     num(d["final_neff_before_resampling"] / n)])
table(["Unknown start", "N", "Final scan log L", "Spread (m)", "ESS / N"], rows,
      [160, 54, 116, 103, 99], size=7.7)
para("Final scan log L is evaluated at the posterior mean with the same beam model; higher is better for that same scan. It is a diagnostic, not a ground-truth error. Large ESS and a narrow cloud can also occur after the filter has retained the wrong hypothesis.", size=8.3)

for dataset in KNOWN:
    failed, chosen = pf(dataset, FAILURES[dataset]), pf(dataset, 1000)
    page(dataset.replace("_", " "), "Question 2(d): known initial position")
    table(["Run", "Position RMSE", "Final position error", "Heading RMSE"], [
        [f"Failure example: N={FAILURES[dataset]}", num(failed["position_rmse"]) + " m",
         num(failed["final_position_error"]) + " m", num(math.degrees(failed["heading_rmse_rad"]), 2) + " deg"],
        ["Selected: N=1000", num(chosen["position_rmse"]) + " m",
         num(chosen["final_position_error"]) + " m", num(math.degrees(chosen["heading_rmse_rad"]), 2) + " deg"],
    ], [170, 116, 130, 116])
    images([failed["image"], chosen["image"]],
           [f"Failure example, N={FAILURES[dataset]}, seed 7.", "Selected result, N=1000, seed 7."], height=380)
    para(known_notes[dataset], size=9.1)
    para("Black points show the final weighted particle population; the orange star is its weighted mean. The dashed red path and red endpoint are ground truth used only for evaluation.", size=8.3)
    para("Source: results/pf/" + dataset + "_N*_always_s7/ (JSON metrics, PNG figures and NPZ histories).", size=7.7, color=GREY)

global_notes = {
    "simple_world_path2": "The N=250 and N=2000 runs settle near the lower boundary, with final scan log likelihoods -7.81 and -7.99 and collision fallback rates near 19.5% of particle-steps. N=5000 gives a much better final scan fit (-1.67), a plausible free-space endpoint and only one collision fallback. It is the best supported candidate in this sweep; accuracy is not measurable without ground truth.",
    "loops_world_path2": "The occupancy map is exactly invariant under a 180-degree rotation about (5,8). Transforming the N=250 pose by (x,y,heading) -> (10-x,16-y,heading+pi) puts it only 0.0094 m and 0.0015 rad from the N=2000 pose. Their likelihoods are nearly equal. These are symmetry-equivalent modes, not evidence that N=250 is wrong. N=2000 and N=5000 end at essentially the same mode; N=2000 is selected as a representative economical setting.",
    "rooms_world_path2": "N=2000 provides the strongest final scan fit (-2.15) and only 55 collision fallbacks. N=5000 has a worse fit (-7.09) and 1,265,763 fallbacks, about 24% of particle-steps. A larger count can select a worse hypothesis in a single stochastic realization. N=250 is tightly concentrated with ESS/N=0.975 but has poor scan fit (-36.74); concentration alone is insufficient evidence of success.",
}
for dataset in GLOBAL:
    chosen_n = SELECTED[dataset]
    other_n = 5000 if dataset == "rooms_world_path2" else 250
    selected, other = pf(dataset, chosen_n), pf(dataset, other_n)
    page(dataset.replace("_", " "), "Question 2(e): global localization")
    rows = []
    for n in [250, 2000, 5000]:
        d = pf(dataset, n)
        rows.append([n, num(d["final_scan_mean_pose_log_likelihood"]),
                     num(d["final_position_spread_rms"]),
                     ", ".join(num(v) for v in d["final_mean"][:2]), d["collision_fallbacks"]])
    table(["N", "Final scan log L", "Spread (m)", "Final xy (m)", "Fallbacks"], rows,
          [45, 115, 100, 174, 98], size=8)
    images([other["image"], selected["image"]],
           [f"Comparison: N={other_n}.", f"Selected candidate: N={chosen_n}."], height=350)
    para(global_notes[dataset], size=8.8)
    para("The orange path is a history of posterior means. During multimodality its connecting segments can pass through obstacles even when particles lie in free space. No ground-truth accuracy claim is made for this dataset.", size=8.1, color=GREY)

for dataset in KNOWN + GLOBAL:
    n = SELECTED[dataset]
    runs = [pf(dataset, n, method) for method in ["always", "never", "adaptive"]]
    page(dataset.replace("_", " ") + f"  |  N={n}", "Question 2(f): resampling comparison")
    rows = []
    for method, d in zip(["Every step", "Never", "ESS < 2N/3"], runs):
        rows.append([method, num(d.get("position_rmse")), num(d.get("final_position_error")),
                     num(d["final_scan_mean_pose_log_likelihood"]), num(d["final_neff_before_resampling"], 1),
                     f"{d['resampling_count']}/{d['num_steps']}"])
    table(["Strategy", "RMSE (m)", "Final error (m)", "Final log L", "Final ESS", "Resamples"], rows,
          [101, 85, 97, 85, 78, 86], size=7.9)
    images([runs[1]["image"], runs[2]["image"]],
           ["Never resample: point sizes still indicate posterior weights.",
            "Adaptive resampling: ESS is measured before resampling."], height=385)
    if dataset in KNOWN:
        para(f"At the same N, adaptive resampling gives position RMSE {num(runs[2]['position_rmse'])} m, compared with {num(runs[0]['position_rmse'])} m when resampling every step. It resamples {runs[2]['resampling_count']} of {runs[2]['num_steps']} steps. Never-resampling ends with ESS {num(runs[1]['final_neff_before_resampling'], 2)}; its mean is therefore supported by approximately one surviving weighted trajectory.", size=8.8)
        if dataset == "simple_world_path1":
            para("The short simple path retains a small position error without resampling, but its final scan fit and near-unit ESS reveal fragile heading/weight support. Good position error on this short run does not demonstrate a healthy particle approximation.", size=8.6)
    else:
        para(f"Without resampling, final scan log L is {num(runs[1]['final_scan_mean_pose_log_likelihood'])}, versus {num(runs[2]['final_scan_mean_pose_log_likelihood'])} with ESS-triggered resampling. The adaptive result preserves the selected run's scan consistency, while the never-resampled weights collapse to nearly one trajectory. These are scan-fit comparisons, not accuracy measurements.", size=8.8)
    para("The corresponding always-resampling figure appears in the earlier selected-count section. All three strategies use the same dataset, seed, initialization rule, sensor settings and particle count; later random draws diverge as resampling changes the population.", size=8.1, color=GREY)

robust_runs = [(spec, pf(spec["dataset"], spec["particles"], "never", spec["seed"])) for spec in ROBUST]
page("Can more particles rescue never-resampling?", "Question 2(f): increased-count experiments")
para("Seven additional completed runs increase N to 10,000 for the longer known-start cases and 20,000 for global localization. These are bounded follow-up experiments; they do not establish the minimum count needed for convergence.")
rows = []
for spec, d in robust_runs:
    rows.append([spec["dataset"].replace("_world_", " "), spec["particles"],
                 num(d.get("position_rmse")), num(d.get("final_position_error")),
                 num(d.get("tail20_position_rmse")), num(d["final_scan_mean_pose_log_likelihood"]),
                 num(d["final_neff_before_resampling"], 2)])
table(["Dataset", "N", "RMSE m", "Final m", "Tail20 m", "Log L", "ESS"], rows,
      [141, 56, 68, 68, 70, 70, 59], size=7.6)
para("<b>Exploratory position criterion.</b> For known-start recovery, require both final position error &lt; 0.5 m and tail-20% position RMSE &lt; 0.5 m. This is a practical criterion for these experiments, not a course-prescribed threshold or a test of full-posterior reliability. A small terminal error alone can occur by chance after substantial earlier error.")
para("<b>Known-start outcomes.</b> At N=10,000, loops_world_path0 meets that position criterion (final error 0.245 m, tail RMSE 0.322 m), although ESS remains one. simple_world_path3 fails because its final error is 0.757 m; loops_world_path1 and rooms_world_path1 also fail. No successful count is claimed for those cases within the tested range.")
para("<b>Unknown-start outcomes.</b> At N=20,000, simple_world_path2 improves its final scan log likelihood to -2.705, but ESS remains one and no ground truth is available. The increased counts do not restore adequate scan consistency for loops_world_path2 (-16.282) or rooms_world_path2 (-30.846), compared with their selected resampling baselines near -2.4 and -2.15. No tested count cures those failures; a successful threshold was not established.")
para("A near-zero weighted spread is not reassuring when ESS is one: it reflects numerical concentration of the weights, not evidence that the pose is known exactly. The full particle arrays and log-weight-derived normalized weights are saved with each run.")
para("The next two pages include every increased-count final-distribution figure. Original full-size PNG files and the exact commands are in results/pf/&lt;run_id&gt;/; execution.json records successful completion, and the neighboring JSON/NPZ files contain metrics and histories.", size=8.7)
for start in [0, 4]:
    selected_runs = robust_runs[start:start + 4]
    page("Increased-count final distributions", "Question 2(f): supplementary figures")
    for row in range(0, len(selected_runs), 2):
        pair = selected_runs[row:row + 2]
        images([d["image"] for _, d in pair],
               [f"{s['dataset']}, N={s['particles']}, never, seed {s['seed']}." for s, _ in pair],
               height=257)
    para("Particle marker sizes encode normalized weights. An apparently broad raw population can still have approximately one effective particle. Consult the preceding table for ESS and numerical performance.", size=8.2, color=GREY)

for label in ["small", "large"]:
    folder = ROOT / "results" / "ekf" / f"ekf-slam-{label}-noise"
    d = json.loads((folder / "metrics.json").read_text())
    page(f"EKF SLAM: {label}-noise dataset", "Question 3(b): joint pose and landmark estimation")
    table(["Metric", "EKF SLAM", "Dead reckoning"], [
        ["Position RMSE (m)", num(d["slam"]["position_rmse_m"]), num(d["dead_reckoning"]["position_rmse_m"])],
        ["Final position error (m)", num(d["slam"]["final_position_error_m"]), num(d["dead_reckoning"]["final_position_error_m"])],
        ["Heading RMSE (degrees)", num(d["slam"]["heading_rmse_deg"]), num(d["dead_reckoning"]["heading_rmse_deg"])],
        ["Map position RMSE (m)", num(d["map"]["position_rmse_m"]), "No map estimated"],
    ], [252, 140, 140], size=8.4)
    images([folder / "final_map.png", folder / "trajectories.png"],
           ["Final pose and landmark estimates with 3-sigma covariance contours and ground truth.",
            "SLAM, dead-reckoned and ground-truth trajectories."], height=290)
    para(f"The filter processes {d['control_count']} controls and {d['observation_count']} observations, initializes all {d['landmarks_estimated']} landmarks, and performs {d['known_landmark_updates']} updates on already-known landmarks. Initial pose comes from X0. New observations augment the state once; that same observation is not immediately reused for a second update.", size=8.8)
    para("Prediction uses the supplied two-noise distance/heading model and propagates robot-map cross-covariance. Cartesian observation Jacobians update the entire joint state, using a linear solve for the gain and Joseph-form covariance. Landmark initialization rotates the relative Cartesian reading into world coordinates and retains all induced correlations.", size=8.5)
    para(f"Saved R diagonal: {d['process_covariance_R'][0][0]:.6g}, {d['process_covariance_R'][1][1]:.6g}. Saved Q diagonal: {d['measurement_covariance_Q'][0][0]:.6g}, {d['measurement_covariance_Q'][1][1]:.6g}. The file named 'large-noise' has about ten times larger process variances but ten times smaller measurement variances than the 'small-noise' file. The filenames do not mean that every noise source is larger.", size=8.2)

page("Verification, reproducibility and accounting", "Supporting information")
para("<b>34 analytic tests pass:</b> 12 particle-filter tests, 11 LIDAR tests and 11 EKF tests. The test code is in tests/ and uses unittest. It covers zero/near-zero turns, angular wrapping, collisions, noise variances, recursive and log-domain weights, ESS/resampling, beam-model normalization and support, physical ray distances, and finite-difference checks of full joint EKF covariance calculations.")
para("The EKF tests independently check known-landmark updates and augmentation, correlated process noise, initial-time observations, prediction-before-measurement ordering, no double-use of first observations, and independence from supplied ground-truth arrays. Both complete EKF runs retain finite states, symmetric covariance and positive covariance eigenvalues at recorded posteriors.")
table(["Task", "Command from the submission directory"], [
    ["Run analytic tests", "python3 -m unittest discover -s tests -v"],
    ["Run particle filter", "python3 code/pf/RunPF.py data/pf/simple_world_path1.pickle --numParticles 1000 --sparsity 5 --seed 7"],
    ["Run EKF SLAM", "python3 code/ekf-slam/RunEKFSLAM.py data/ekf-slam/ekf-slam-small-noise.pickle"],
    ["Rebuild this report", "python3 reports/build_report.py"],
], [138, 394], size=8.2)
para("<b>Evidence files.</b> Each particle-filter run has execution.json, stdout/stderr, a metrics JSON, a PNG final-cloud plot and an NPZ with particles, weights, estimates, ESS and errors. EKF directories contain metrics.json, results.npz, final_map and trajectories figures. Experiment manifests record the selected settings and follow-up runs. The report reads those saved files and does not alter or rerun inference.", size=9)
para("<b>Question 4 / honest accounting.</b> Human collaborators: enter the actual names, or 'none' if accurate. Time spent: enter your actual time; it cannot be inferred from this report. AI assistance: ChatGPT was used to explain concepts, derive formulas, implement and review the requested code, construct analytic tests, run experiments and prepare this report. Review the implementation and results before submission, and disclose assistance as required by the course.", size=9)
para("<b>References.</b><br/>[1] TTIC 31170, Robot Learning and Estimation, Autumn 2026, Problem Set 3: Questions 2 and 3, supplied ps3-questions.pdf.<br/>[2] Supplied Python starter modules and README files in code/pf, code/ekf-slam and data.<br/>[3] S. Thrun, W. Burgard and D. Fox, <i>Probabilistic Robotics</i>, MIT Press, 2005. Motion and beam algorithms: Tables 5.3 and 6.1, as referenced by the assignment.<br/>[4] The saved experiment metrics, figures and histories bundled with this submission.", size=8.7)

if Y < 45:
    raise ValueError(f"Last page overflows: y={Y}")
C.save()
print(json.dumps({"pdf": str(OUT), "pages": PAGE, "bytes": OUT.stat().st_size}))
