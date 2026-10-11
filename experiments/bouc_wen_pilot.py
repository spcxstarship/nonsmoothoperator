"""Bounded numerical feasibility gate before an LLM/Bouc-Wen comparison.

Run: .venv/bin/python experiments/bouc_wen_pilot.py --out data/research/NEW
Four new synthetic worlds, one noise stream each, baseline plus every candidate
counterfactual. This screens input informativeness; it is NOT an adaptive-policy
contest, a reproduction of the published benchmark, or evidence of LLM gains.
All data are generated locally. No external model is called or trained.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import scipy
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from packages.dynamics import bouc_wen as bw

NOISE = 0.01
STARTS = np.array([[.5, .5, .5, .5], [.2, .7, .3, .8], [.8, .3, .7, .2]])
SEED = 20261009


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fit(history, *, deadline, max_nfev=70):
    """No truth, future observations, reserved profile, or evaluator seeds enter.

Fit in normalized parameter coordinates. All multistart outcomes are retained;
the lowest training SSE selects a fit, regardless of test performance.
"""
    for record in history:
        if set(record) != {"profile", "positions"}:
            raise ValueError("History may contain only profile and positions")
        if record["profile"] not in ("baseline", *bw.CANDIDATES):
            raise ValueError("Reserved profiles cannot enter fitting")
        positions = np.asarray(record["positions"])
        if positions.shape != bw.OBS_TIMES.shape or not np.all(np.isfinite(positions)):
            raise ValueError("Expected 64 finite positions per acquired profile")
    if not history:
        raise ValueError("Empty history")
    calls = 0
    started = time.monotonic()

    def residual(normalized):
        nonlocal calls
        if time.monotonic() > deadline:
            raise TimeoutError("Frozen campaign runtime limit reached")
        calls += 1
        p = bw.LOWER + normalized*(bw.UPPER-bw.LOWER)
        return np.concatenate([
            (bw.simulate(p, r["profile"])[:, 0] - r["positions"])/NOISE
            for r in history])

    attempts = []
    for initial in STARTS:
        # A finite step larger than the integration error is required because
        # abs() switches can amplify adaptive-integration noise in tiny steps.
        result = least_squares(residual, initial, bounds=(0, 1),
                               diff_step=1e-3, max_nfev=max_nfev,
                               ftol=1e-6, xtol=1e-6, gtol=1e-5)
        singular = np.linalg.svd(result.jac, compute_uv=False)
        attempts.append({
            "initial": initial.tolist(), "parameters":
                (bw.LOWER + result.x*(bw.UPPER-bw.LOWER)).tolist(),
            "standardized_sse": float(result.fun @ result.fun),
            "converged": bool(result.success), "status": int(result.status),
            "message": result.message, "nfev": result.nfev,
            "normalized_jacobian_singular_values": singular.tolist(),
        })
    best = min(attempts, key=lambda a: a["standardized_sse"])
    return {"best": best, "starts": attempts, "residual_calls": calls,
            "elapsed_s": time.monotonic()-started,
            "observed_positions": len(history)*len(bw.OBS_TIMES)}


def protocol():
    return {
        "study": "normalized_bouc_wen_input_screen_v1",
        "stage": "development_feasibility_not_policy_comparison",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "worlds": 4, "noise_replicates_per_world": 1,
        "parameters": bw.PARAMETERS, "bounds": [bw.LOWER.tolist(), bw.UPPER.tolist()],
        "known": {"mass": 1, "exponent": 1, "sigma": bw.SIGMA,
                  "initial_state": [0, 0, 0], "model_family": "normalized Bouc-Wen"},
        "equations": ["x'=v", "v'=u-c*v-k*x-h*w",
                      "w'=rho*(v-0.75*abs(v)*w-0.25*v*abs(w))"],
        "noise_std": NOISE, "times": bw.OBS_TIMES.tolist(),
        "baseline": "baseline", "candidates": bw.CANDIDATES,
        "reserved": bw.RESERVED, "candidate_force_rms": float(np.sqrt(.32)),
        "force_peak_limit": 1.4, "duration_per_profile": bw.DURATION,
        "positions_per_profile": 64, "positions_after_one_acquisition": 128,
        "starts": STARTS.tolist(), "max_nfev_per_start": 70,
        "runtime_limit_s": 600, "numerical_position_tolerance": NOISE*.02,
        "primary_metric": "Mean of two reserved-trajectory position RMSEs, dimensionless",
        "evaluation_times": np.linspace(0, bw.DURATION, 321).tolist(),
        "calibration": {"parameters": [.23, .91, .67, 1.31],
                        "profiles": ["slow_cycles", "mixed_cycles"],
                        "max_normalized_parameter_error": .02},
        "failure_rule": "Stop without aggregate rankings on timeout, numerical failure, or nonconverged selected fit",
        "go_rule": "Validated physics and calibration, plus baseline mean prediction error above sensor noise and at least one candidate reducing it by 25%; exploratory engineering gate only",
        "sources": [
            "https://doi.org/10.4121/12967592",
            "https://arxiv.org/abs/2509.17122",
        ],
        "limits": [
            "Dimensionless normalized adaptation, not original benchmark reproduction",
            "Known mechanism and exponent; no unknown-family discovery",
            "One noise stream per world; no significance or superiority claims",
            "Counterfactual candidate outcomes are evaluator-only diagnostics, not an executable selector",
            "Equal measurement count, duration and force RMS; different force peaks and frequency content",
            "No LLM calls, PINN fitting, or neural-weight training",
            "All four worlds become development data; independent confirmation must use new worlds",
        ],
    }


def execute(out, spec):
    started = time.monotonic()
    deadline = started + spec["runtime_limit_s"]
    rng = np.random.default_rng(SEED)
    truths = bw.LOWER + rng.uniform(.15, .85, (spec["worlds"], 4))*(bw.UPPER-bw.LOWER)
    eval_times = np.array(spec["evaluation_times"])
    tolerance = spec["numerical_position_tolerance"]
    reference_checks = []
    # Verify the fixed nominal model and generated worlds across every profile.
    for p in [np.mean([bw.LOWER, bw.UPPER], axis=0), *truths]:
        for profile in bw.PROFILES:
            if time.monotonic() > deadline:
                raise TimeoutError("Preflight exceeded campaign budget")
            regular = bw.simulate(p, profile, eval_times)
            reference = bw.simulate(p, profile, eval_times, reference=True)
            delta = np.max(np.abs(regular[:, 0]-reference[:, 0]))
            check = {"parameters": p.tolist(), "profile": profile,
                     "max_position_difference": float(delta),
                     "max_abs_memory": float(np.max(np.abs(reference[:, 2])))}
            reference_checks.append(check)
            if delta > tolerance or check["max_abs_memory"] > 1+1e-7:
                save(out / "preflight.json", reference_checks)
                raise RuntimeError("Numerical or normalized-memory-bound check failed")
    save(out / "preflight.json", reference_checks)

    calibration_p = np.array(spec["calibration"]["parameters"])
    calibration_history = [{"profile": name, "positions": bw.simulate(
        calibration_p, name, reference=True)[:, 0].tolist()}
        for name in spec["calibration"]["profiles"]]
    calibration = fit(calibration_history, deadline=deadline)
    calibration["max_normalized_parameter_error"] = float(np.max(np.abs(
        np.array(calibration["best"]["parameters"])-calibration_p)/(bw.UPPER-bw.LOWER)))
    save(out / "calibration.json", calibration)
    if (not calibration["best"]["converged"] or
            calibration["max_normalized_parameter_error"] > .02):
        raise RuntimeError("Noiseless parameter recovery gate failed")

    rows = []
    for index, truth in enumerate(truths):
        case = out / f"case-{index+1}"
        case.mkdir()
        save(case / "truth.json", dict(zip(bw.PARAMETERS, truth.tolist())))
        observed = {}
        # Independent profile-indexed noise is generated before fitting.
        for profile in ("baseline", *bw.CANDIDATES):
            positions = bw.simulate(truth, profile, reference=True)[:, 0]
            positions += rng.normal(0, NOISE, len(positions))
            observed[profile] = {"profile": profile, "positions": positions.tolist()}
        save(case / "observations.json", observed)
        reserved = {name: bw.simulate(truth, name, eval_times, reference=True)[:, 0]
                    for name in bw.RESERVED}
        for candidate in ("baseline_only", *bw.CANDIDATES):
            history = [observed["baseline"]]
            if candidate != "baseline_only":
                history.append(observed[candidate])
            fitted = fit(history, deadline=deadline)
            save(case / f"fit-{candidate}.json", fitted)
            if not fitted["best"]["converged"]:
                raise RuntimeError("Selected fit did not converge; no cohort ranking")
            best = fitted["best"]["parameters"]
            errors = {}
            max_reference_delta = 0.0
            for profile, truth_positions in reserved.items():
                predicted = bw.simulate(best, profile, eval_times)[:, 0]
                reference = bw.simulate(best, profile, eval_times, reference=True)[:, 0]
                max_reference_delta = max(max_reference_delta, float(np.max(abs(predicted-reference))))
                errors[profile] = float(np.sqrt(np.mean((reference-truth_positions)**2)))
            if max_reference_delta > tolerance:
                raise RuntimeError("Fitted model failed reference integration check")
            row = {"case": index+1, "acquisition": candidate,
                   "mean_reserved_rmse": float(np.mean(list(errors.values()))),
                   "reserved_rmse": errors,
                   "normalized_parameter_rmse": float(np.sqrt(np.mean((
                       (np.array(best)-truth)/(bw.UPPER-bw.LOWER))**2))),
                   "fitted_reference_max_difference": max_reference_delta,
                   "observed_positions": fitted["observed_positions"]}
            rows.append(row)
            save(case / f"evaluation-{candidate}.json", row)
            print(json.dumps({"case": index+1, "acquisition": candidate,
                              "reserved_rmse": row["mean_reserved_rmse"]}), flush=True)
    means = {a: float(np.mean([r["mean_reserved_rmse"] for r in rows if r["acquisition"] == a]))
             for a in ("baseline_only", *bw.CANDIDATES)}
    best_candidate = min(bw.CANDIDATES, key=means.get)
    reductions = {a: 1-means[a]/means["baseline_only"] for a in bw.CANDIDATES}
    go = means["baseline_only"] > NOISE and reductions[best_candidate] >= .25
    results = {"status": "complete", "rows": rows, "mean_reserved_rmse": means,
               "fractional_reduction_from_baseline": reductions,
               "descriptive_best_fixed_candidate": best_candidate,
               "case_winners": [min((r for r in rows if r["case"] == i+1
                                                      and r["acquisition"] != "baseline_only"),
                                    key=lambda r: r["mean_reserved_rmse"])["acquisition"]
                                for i in range(len(truths))],
               "next_gate": "eligible_for_frozen_selector_pilot" if go else "reconsider_before_selector_pilot",
               "elapsed_s": time.monotonic()-started,
               "interpretation": "Development input screen; no LLM or adaptive-selection advantage measured"}
    save(out / "results.json", results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sources = args.out / "sources"
    sources.mkdir()
    hashes = {}
    for source in (Path(__file__), ROOT / "packages/dynamics/bouc_wen.py",
                   ROOT / "tests/test_bouc_wen.py"):
        if source.exists():
            shutil.copyfile(source, sources / source.name)
            hashes[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
    spec = protocol()
    spec["source_hashes"] = hashes
    spec["environment"] = {"python": sys.version.split()[0],
                           "numpy": np.__version__, "scipy": scipy.__version__}
    save(args.out / "protocol.json", spec)
    try:
        result = execute(args.out, spec)
    except Exception as exc:
        save(args.out / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                                        "aggregate_ranking": "withheld"})
        raise
    print(json.dumps({"status": result["status"], "next_gate": result["next_gate"],
                      "means": result["mean_reserved_rmse"]}), flush=True)


if __name__ == "__main__":
    main()
