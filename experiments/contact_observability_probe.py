"""Small exploratory check: trajectory disagreement versus observable information.

Run: python3 experiments/contact_observability_probe.py

A ball starts at world height 1 m. Two equally likely, otherwise known models
place the floor at 0 or 0.2 m. Choose one constant acceleration, then take ONE
world-position measurement at a specified time, with Gaussian noise (std 0.05 m).
Exact event times and velocities are not observed. All paths stop before the
second impact. The two models are deliberately a minimal prior, not learned PINNs.

Policies maximize either time-averaged squared position disagreement before the
measurement, or squared disagreement at the actual measurement time. For two
known Gaussian means with equal variance and prior, the latter is Bayes optimal.
Thus this is a check of an observation-blind proxy, not a new design algorithm.

The 0.60 s example was chosen after an exploratory 0.20--0.85 s sweep; the whole
sweep is reported to expose selection effects. No claim of preregistration or
general superiority is made. Error rates are exact under the assumed model;
an independent measurement Monte Carlo checks the example. Seeds belong only to
the evaluation harness; there is no LLM or selector context in this experiment.

Optional output: redirect stdout to data/contact_observability_probe.json
(ignored by Git). Python standard library only.
"""

import json
from math import erfc, sqrt
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.dynamics.bouncing_ball import BouncingBall


FLOORS = (0.0, 0.2)
INPUTS = {"downward": -4.0, "neutral": 0.0, "upward": 4.0}
NOISE_STD = 0.05
MODELS = {
    name: tuple(BouncingBall(initial_height=1.0 - floor,
                             applied_acceleration=acceleration)
                for floor in FLOORS)
    for name, acceleration in INPUTS.items()
}


def positions(candidate: str, time: float) -> tuple[float, float]:
    return tuple(floor + model.state_at(time).height
                 for floor, model in zip(FLOORS, MODELS[candidate]))


def evaluate(time: float, intervals: int = 1000) -> dict:
    candidates = {}
    for name in INPUTS:
        means = positions(name, time)
        gap = abs(means[1] - means[0])
        integral = 0.0
        for i in range(intervals + 1):
            pair = positions(name, time * i / intervals)
            weight = 0.5 if i in (0, intervals) else 1.0
            integral += weight * (pair[1] - pair[0]) ** 2
        candidates[name] = {
            "position_means_m": means,
            "mean_squared_trajectory_gap_m2": integral / intervals,
            "measurement_gap_m": gap,
            "exact_identification_error": 0.5 * erfc(gap / (2 * sqrt(2) * NOISE_STD)),
        }

    # Fixed insertion-order tie breaking, rounding only below physical precision.
    dense = max(INPUTS, key=lambda name: round(
        candidates[name]["mean_squared_trajectory_gap_m2"], 14))
    observed = max(INPUTS, key=lambda name: round(
        candidates[name]["measurement_gap_m"], 14))
    return {
        "measurement_time_s": time,
        "candidates": candidates,
        "trajectory_disagreement_choice": dense,
        "measurement_disagreement_choice": observed,
        "trajectory_policy_error": candidates[dense]["exact_identification_error"],
        "measurement_policy_error": candidates[observed]["exact_identification_error"],
        "uniform_random_policy_error": sum(
            c["exact_identification_error"] for c in candidates.values()) / len(INPUTS),
    }


def monte_carlo(time: float, trials_per_model: int = 100000, seed: int = 20261006) -> dict:
    """Classify using Gaussian likelihoods; paired noise across candidate inputs."""
    rng = random.Random(seed)
    means = {name: positions(name, time) for name in INPUTS}
    errors = dict.fromkeys(INPUTS, 0)
    for truth in range(len(FLOORS)):
        for _ in range(trials_per_model):
            noise = rng.gauss(0, NOISE_STD)
            for name, pair in means.items():
                measurement = pair[truth] + noise
                predicted = min(range(len(FLOORS)),
                                key=lambda j: (measurement - pair[j]) ** 2)
                errors[name] += predicted != truth
    total = trials_per_model * len(FLOORS)
    return {
        "seed": seed,
        "trials_per_candidate": total,
        "error_rates": {name: count / total for name, count in errors.items()},
    }


def main() -> None:
    sweep = [evaluate(tick / 100) for tick in range(20, 86)]
    example = evaluate(0.60)
    refined = evaluate(0.60, intervals=2000)
    mc = monte_carlo(0.60)
    quadrature_change = max(abs(
        example["candidates"][name]["mean_squared_trajectory_gap_m2"]
        - refined["candidates"][name]["mean_squared_trajectory_gap_m2"])
        for name in INPUTS)
    mc_z = {}
    for name, candidate in example["candidates"].items():
        p = candidate["exact_identification_error"]
        standard_error = sqrt(p * (1 - p) / mc["trials_per_candidate"])
        mc_z[name] = (mc["error_rates"][name] - p) / standard_error

    report = {
        "study_kind": "exploratory_analytic_observability_probe",
        "config": {"floor_hypotheses_m": FLOORS, "prior": [0.5, 0.5],
                   "initial_world_height_m": 1.0, "initial_velocity_m_s": 0.0,
                   "gravity_m_s2": 9.81, "restitution": 0.8,
                   "acceleration_candidates_m_s2": INPUTS,
                   "measurement_noise_std_m": NOISE_STD,
                   "measurements_per_experiment": 1,
                   "quadrature_intervals": 1000},
        "example": example,
        "example_monte_carlo": mc,
        "checks": {
            "earliest_second_impact_s": min(m.next_impact_time
                for models in MODELS.values() for m in models),
            "quadrature_max_absolute_change_m2": quadrature_change,
            "refined_quadrature_same_choice": (
                example["trajectory_disagreement_choice"] == refined["trajectory_disagreement_choice"]),
            "monte_carlo_error_in_standard_errors": mc_z,
        },
        "sweep_summary": {
            "time_count": len(sweep),
            "trajectory_policy_strictly_worse_count": sum(
                row["trajectory_policy_error"] > row["measurement_policy_error"] + 1e-10
                for row in sweep),
            "mean_error_by_policy": {policy: sum(row[policy] for row in sweep) / len(sweep)
                for policy in ("trajectory_policy_error", "measurement_policy_error",
                               "uniform_random_policy_error")},
        },
        "sweep": sweep,
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
