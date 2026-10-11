"""Run a fixed-data classical versus flow/jump PINN parameter-recovery pilot.

Usage: .venv/bin/python experiments/identify_ball.py --out data/runs/ball-identification-v1
Output directories must be new. All seeds, including failed fits, are retained.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from packages.dynamics.bouncing_ball import BouncingBall
from packages.dynamics.ball_identification import Observations, fit_classical, predict
from packages.pinn.ball import BallPINN, train


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def reference(config, initial_height, acceleration, times):
    truth = config["simulator"]
    ball = BouncingBall(initial_height=initial_height - truth["floor_m"],
                        applied_acceleration=acceleration, restitution=truth["restitution"],
                        gravity=truth["gravity_m_s2"])
    states = [ball.state_at(float(t)) for t in times]
    return (np.array([s.height + truth["floor_m"] for s in states]),
            np.array([s.velocity for s in states]), ball.impact_time)


def generate(config):
    spec = config["observations"]
    rng = np.random.default_rng(spec["seed"])
    times = np.array(spec["times_s"])
    observed = []
    for acceleration in spec["accelerations_m_s2"]:
        height, _, _ = reference(config, spec["initial_height_m"], acceleration, times)
        observed.append(Observations(spec["initial_height_m"], acceleration, times.copy(),
                                     height + rng.normal(0, spec["noise_std_m"], len(times))))
    return observed


def transfer_metrics(parameters, config):
    """Fresh-input predictions from the analytic simulator using fitted parameters.

    These are parameter-transfer metrics, NOT direct neural-network predictions.
    Hidden truth enters evaluation only, after the fit has completed.
    """
    evaluation, truth = config["evaluation"], config["simulator"]
    times = np.linspace(0, evaluation["horizon_s"], evaluation["points_per_trajectory"])
    height_errors, velocity_errors, timing_errors = [], [], []
    for h0 in evaluation["initial_heights_m"]:
        for acceleration in evaluation["accelerations_m_s2"]:
            h, v, tau = reference(config, h0, acceleration, times)
            ph, pv = predict(parameters["floor_m"], parameters["restitution"], h0, acceleration,
                             times, truth["gravity_m_s2"])
            height_errors.extend(ph - h)
            velocity_errors.extend(pv - v)
            fitted_tau = np.sqrt(2 * (h0 - parameters["floor_m"]) / (truth["gravity_m_s2"] - acceleration))
            timing_errors.append(abs(fitted_tau - tau))
    return {
        "floor_absolute_error_m": abs(parameters["floor_m"] - truth["floor_m"]),
        "restitution_absolute_error": abs(parameters["restitution"] - truth["restitution"]),
        "parameter_transfer_position_rmse_m": float(np.sqrt(np.mean(np.square(height_errors)))),
        "parameter_transfer_velocity_rmse_m_s": float(np.sqrt(np.mean(np.square(velocity_errors)))),
        "parameter_transfer_impact_time_mae_s": float(np.mean(timing_errors)),
    }


def reconstruction(model, observations, config):
    rows, errors, timing_errors = [], [], []
    for index, obs in enumerate(observations):
        times = np.linspace(0, obs.times[-1], config["evaluation"]["points_per_trajectory"])
        true_h, true_v, true_tau = reference(config, obs.initial_height, obs.acceleration, times)
        with torch.no_grad():
            ph, pv = model(index, times)
        ph, pv = ph.numpy().ravel(), pv.numpy().ravel()
        errors.extend(ph - true_h)
        timing_errors.append(abs(float(model.event_times[index].detach()) - true_tau))
        rows.append({"acceleration_m_s2": obs.acceleration, "times_s": times.tolist(),
                     "reference_height_m": true_h.tolist(), "pinn_height_m": ph.tolist(),
                     "reference_velocity_m_s": true_v.tolist(), "pinn_velocity_m_s": pv.tolist()})
    return rows, {"neural_reconstruction_position_rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
                  "learned_training_event_time_mae_s": float(np.mean(timing_errors))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/ball-identification.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "config.json", config)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    observations = generate(config)
    # This file deliberately omits simulator parameters, RNG seed, and event times.
    clean_data = [{**asdict(obs), "times": obs.times.tolist(), "heights": obs.heights.tolist()}
                  for obs in observations]
    write_json(args.out / "observations.json", clean_data)
    source_files = ["experiments/identify_ball.py", "packages/pinn/ball.py",
                    "packages/dynamics/ball_identification.py", "packages/dynamics/bouncing_ball.py"]
    report = {
        "study": config["study"], "status": "running",
        "interpretation": "Fixed synthetic dataset, multiple network initializations. No active selection or LLM.",
        "transfer_definition": "Analytic rollouts of learned parameters on fresh inputs; not neural generalization.",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "scipy": scipy.__version__, "device": "cpu", "dtype": "float64"},
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files},
        "observations_sha256": hashlib.sha256((args.out / "observations.json").read_bytes()).hexdigest(),
        "pinn_runs": [],
    }
    inference = config["inference"]
    gravity = config["simulator"]["gravity_m_s2"]
    started = time.perf_counter()
    classical = fit_classical(observations, inference["floor_bounds_m"], inference["restitution_bounds"], gravity)
    classical["elapsed_s"] = time.perf_counter() - started
    classical["evaluation"] = transfer_metrics(classical, config)
    report["classical"] = classical
    report["initial_parameters_evaluation"] = transfer_metrics(
        {"floor_m": inference["initial_floor_m"], "restitution": inference["initial_restitution"]}, config)
    print("Classical fit:", json.dumps(classical), flush=True)
    write_json(args.out / "report.json", report)
    for seed in config["pinn"]["seeds"]:
        seed_dir = args.out / f"seed-{seed}"
        seed_dir.mkdir()
        run = {"seed": seed, "status": "running"}
        report["pinn_runs"].append(run)
        write_json(args.out / "report.json", report)
        model = BallPINN(observations, inference, config["pinn"], gravity, seed)
        try:
            with (seed_dir / "training.jsonl").open("x") as log:
                def progress(row):
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    print(f"PINN seed {seed}: {json.dumps(row)}", flush=True)
                run["training"] = train(model, progress)
            run.update({"status": "complete", "floor_m": float(model.floor.detach()),
                        "restitution": float(model.restitution.detach()),
                        "learned_event_times_s": model.event_times.detach().tolist()})
            run["diagnostics"] = model.diagnostics(config["evaluation"]["diagnostic_points_per_segment"])
            run["evaluation"] = transfer_metrics(run, config)
            curves, direct_metrics = reconstruction(model, observations, config)
            run["evaluation"].update(direct_metrics)
            metrics = {**run["evaluation"], **run["diagnostics"]}
            run["pilot_criteria_pass"] = {name: metrics[name] <= limit
                                         for name, limit in config["pilot_success_criteria"].items()}
            torch.save(model.state_dict(), seed_dir / "model.pt")
            write_json(seed_dir / "curves.json", curves)
            print(f"PINN seed {seed} result:", json.dumps({k: v for k, v in run.items() if k != "training"}), flush=True)
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            run.update({"status": "failed", "error": str(exc)})
            print(f"PINN seed {seed} failed: {exc}", file=sys.stderr, flush=True)
        write_json(seed_dir / "result.json", run)
        write_json(args.out / "report.json", report)
    report["status"] = "complete" if all(r["status"] == "complete" for r in report["pinn_runs"]) else "completed_with_failures"
    write_json(args.out / "report.json", report)
    print(f"Saved results to {args.out}", flush=True)


if __name__ == "__main__":
    main()
