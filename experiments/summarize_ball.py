"""Plot a completed identification run and check error between grid samples.

Usage: .venv/bin/python experiments/summarize_ball.py data/runs/ball-identification-v1
No model is fitted or selected by this script. Original reports are preserved.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from identify_ball import predict, reference, write_json


def integrated_transfer(parameters, config):
    """Integrate squared errors, splitting at both true and predicted impacts.

    Three-point Gauss-Legendre is exact for the degree <= 4 polynomials in
    squared position/velocity error within each fixed-mode interval. This avoids
    over- or under-counting a short event-time mismatch on a coarse time grid.
    """
    evaluation, truth = config["evaluation"], config["simulator"]
    horizon = evaluation["horizon_s"]
    nodes, weights = np.polynomial.legendre.leggauss(3)
    integrated_h, integrated_v, duration = 0., 0., 0.
    for h0 in evaluation["initial_heights_m"]:
        for acceleration in evaluation["accelerations_m_s2"]:
            a = truth["gravity_m_s2"] - acceleration
            taus = [np.sqrt(2 * (h0 - floor) / a)
                    for floor in (truth["floor_m"], parameters["floor_m"])]
            breaks = sorted({0., horizon, *(tau for tau in taus if 0 < tau < horizon)})
            for left, right in zip(breaks[:-1], breaks[1:]):
                half = (right - left) / 2
                times = (left + right) / 2 + half * nodes
                h, v, _ = reference(config, h0, acceleration, times)
                ph, pv = predict(parameters["floor_m"], parameters["restitution"],
                                 h0, acceleration, times, truth["gravity_m_s2"])
                integrated_h += half * np.dot(weights, (ph - h)**2)
                integrated_v += half * np.dot(weights, (pv - v)**2)
            duration += horizon
    return {"time_integrated_position_rmse_m": float(np.sqrt(integrated_h / duration)),
            "time_integrated_velocity_rmse_m_s": float(np.sqrt(integrated_v / duration))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    config = json.loads((args.run / "config.json").read_text())
    report = json.loads((args.run / "report.json").read_text())
    observations = json.loads((args.run / "observations.json").read_text())
    runs = [r for r in report["pinn_runs"] if r["status"] == "complete"]
    checks = {"classical": integrated_transfer(report["classical"], config),
              "pinn": [{"seed": r["seed"], **integrated_transfer(r, config)} for r in runs],
              "note": "Post-fit integration check, splitting at both event times. No training or tuning."}
    write_json(args.run / "integrated-evaluation.json", checks)
    os.environ.setdefault("MPLCONFIGDIR", str((args.run / "matplotlib-cache").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#0072B2", "#D55E00", "#009E73"]
    fig, axes = plt.subplots(2, len(observations), figsize=(13, 7), sharex="col", squeeze=False)
    for index, obs in enumerate(observations):
        times = np.linspace(0, max(obs["times"]), config["evaluation"]["points_per_trajectory"])
        true_h, _, _ = reference(config, obs["initial_height"], obs["acceleration"], times)
        classical_h, _ = predict(report["classical"]["floor_m"], report["classical"]["restitution"],
                                  obs["initial_height"], obs["acceleration"], times,
                                  config["simulator"]["gravity_m_s2"])
        top, bottom = axes[:, index]
        top.plot(times, true_h, color="#222222", linewidth=2.5, label="True trajectory")
        top.scatter(obs["times"], obs["heights"], color="#222222", s=20, zorder=5, label="Noisy positions")
        bottom.axhline(0, color="#aaaaaa", linewidth=1)
        bottom.plot(times, (classical_h - true_h) * 1000, "--", color="#222222", label="Classical fit")
        for j, run in enumerate(runs):
            curves = json.loads((args.run / f"seed-{run['seed']}" / "curves.json").read_text())[index]
            predicted = np.array(curves["pinn_height_m"])
            label = f"PINN seed {run['seed']}"
            color = colors[j % len(colors)]
            top.plot(times, predicted, color=color, linewidth=1.2, alpha=.85, label=label)
            bottom.plot(times, (predicted - true_h) * 1000, color=color, linewidth=1.2, label=label)
        top.axhline(config["simulator"]["floor_m"], color="#777777", linestyle=":", linewidth=1)
        top.set_title(f"Constant input: {obs['acceleration']:+g} m/s²")
        top.set_ylabel("World height (m)")
        bottom.set_ylabel("Position error (mm)")
        bottom.set_xlabel("Time (s)")
        for ax in (top, bottom):
            ax.grid(alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle("Learning a bouncing ball from 36 noisy positions", fontsize=17, y=.98)
    fig.text(.5, .025, "Neural reconstruction on the three observed trajectories • All three initializations shown\n"
             "Floor and restitution are shared unknowns; velocities and impact times are not observed.",
             ha="center", fontsize=10, color="#444444")
    fig.tight_layout(rect=(0, .08, 1, .95))
    fig.savefig(args.run / "identification.png", dpi=160, metadata={"Software": "NonSmoothOperator"})
    plt.close(fig)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
