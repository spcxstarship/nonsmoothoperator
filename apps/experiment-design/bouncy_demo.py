"""Small active-experiment demo using a noisy bouncing-ball observation.

This tests experiment selection with a least-squares parameter estimator. It
does not train or evaluate a PINN.
"""

import argparse
import json
from math import isfinite, sqrt
from pathlib import Path
import random
import sys

from selector import parse_decision, validate_context

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packages.dynamics.bouncing_ball import BouncingBall


ACCELERATIONS = {"downward": -4.0, "neutral": 0.0, "upward": 4.0}
HEIGHT = 1.0
GRAVITY = 9.81


def impact_speed(acceleration: float) -> float:
    return BouncingBall(
        initial_height=HEIGHT, gravity=GRAVITY, applied_acceleration=acceleration
    ).impact_speed


def observation(candidate_id: str, restitution: float, noise: float) -> dict:
    acceleration = ACCELERATIONS[candidate_id]
    speed = impact_speed(acceleration)
    return {
        "candidate_id": candidate_id,
        "acceleration_m_s2": acceleration,
        "pre_impact_speed_m_s": speed,
        "measured_post_impact_speed_m_s": restitution * speed + noise,
    }


def estimate_restitution(observations: list[dict], noise_std: float) -> dict:
    denominator = sum(item["pre_impact_speed_m_s"] ** 2 for item in observations)
    numerator = sum(
        item["pre_impact_speed_m_s"] * item["measured_post_impact_speed_m_s"]
        for item in observations
    )
    return {
        "estimate": numerator / denominator,
        "standard_error": noise_std / sqrt(denominator),
    }


def make_context(restitution: float, noise_std: float, seed: int) -> dict:
    rng = random.Random(seed)
    baseline = observation("neutral", restitution, rng.gauss(0, noise_std))
    return {
        "objective": "Choose one constant vertical input to reduce uncertainty in the restitution estimate.",
        "max_abs_acceleration_m_s2": 4.0,
        "candidates": [
            {
                "id": candidate_id,
                "acceleration_m_s2": acceleration,
                "expected_pre_impact_speed_m_s": impact_speed(acceleration),
                "description": "Constant input from release through the first rebound",
            }
            for candidate_id, acceleration in ACCELERATIONS.items()
        ],
        "history": [baseline],
        "diagnostics": {
            "restitution": estimate_restitution([baseline], noise_std),
            "post_impact_speed_noise_std_m_s": noise_std,
            "note": "The true restitution and future measurements are not provided.",
        },
        "demo_seed": seed,
    }


def validate_demo_context(context: dict) -> None:
    validate_context(context)
    candidates = context["candidates"]
    if len(candidates) != len(ACCELERATIONS):
        raise ValueError("demo candidate set has changed")
    for candidate in candidates:
        candidate_id = candidate["id"]
        if (
            candidate_id not in ACCELERATIONS
            or candidate.get("acceleration_m_s2") != ACCELERATIONS[candidate_id]
        ):
            raise ValueError("demo candidate actuation does not match its ID")
    if context.get("max_abs_acceleration_m_s2") != 4.0:
        raise ValueError("demo actuation limit has changed")


def run_round(context: dict, decision_text: str, restitution: float, noise_std: float) -> dict:
    validate_demo_context(context)
    decision = parse_decision(decision_text, validate_context(context))
    seed = context["demo_seed"]
    rng = random.Random(seed)
    baseline = observation("neutral", restitution, rng.gauss(0, noise_std))
    if baseline != context["history"][0]:
        raise ValueError("context baseline does not match this simulation setup")
    chosen = observation(decision.candidate_id, restitution, rng.gauss(0, noise_std))
    return {
        "candidate_id": decision.candidate_id,
        "observations": [baseline, chosen],
        "before": estimate_restitution([baseline], noise_std),
        "after": estimate_restitution([baseline, chosen], noise_std),
    }


def compare_candidates(restitution: float, noise_std: float, seeds: int) -> dict:
    squared_before = []
    squared_after = {candidate_id: [] for candidate_id in ACCELERATIONS}
    for seed in range(seeds):
        rng = random.Random(seed)
        baseline_noise = rng.gauss(0, noise_std)
        next_noise = rng.gauss(0, noise_std)
        baseline = observation("neutral", restitution, baseline_noise)
        before = estimate_restitution([baseline], noise_std)["estimate"]
        squared_before.append((before - restitution) ** 2)
        for candidate_id in ACCELERATIONS:
            next_observation = observation(candidate_id, restitution, next_noise)
            after = estimate_restitution([baseline, next_observation], noise_std)["estimate"]
            squared_after[candidate_id].append((after - restitution) ** 2)
    return {
        "seed_count": seeds,
        "baseline_parameter_rmse": sqrt(sum(squared_before) / seeds),
        "parameter_rmse_by_candidate": {
            candidate_id: sqrt(sum(errors) / seeds)
            for candidate_id, errors in squared_after.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--true-restitution", type=float, required=True)
    parser.add_argument("--noise-std", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=7)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("prepare", help="print a selection context without the true parameter")
    evaluate = subcommands.add_parser("evaluate", help="evaluate the selected experiment")
    evaluate.add_argument("context", type=Path)
    evaluate.add_argument("decision", type=Path)
    evaluate.add_argument("--seeds", type=int, default=100)
    args = parser.parse_args()

    try:
        if not isfinite(args.true_restitution) or not 0 <= args.true_restitution < 1:
            raise ValueError("true restitution must be in [0, 1)")
        if not isfinite(args.noise_std) or args.noise_std <= 0:
            raise ValueError("noise std must be positive and finite")
        if args.command == "prepare":
            result = make_context(args.true_restitution, args.noise_std, args.seed)
        else:
            if args.seeds < 1:
                raise ValueError("seeds must be positive")
            context = json.loads(args.context.read_text(encoding="utf-8"))
            decision = args.decision.read_text(encoding="utf-8")
            result = {
                "model_kind": "least_squares_restitution_estimator",
                "simulation_config": {
                    "true_restitution": args.true_restitution,
                    "noise_std_m_s": args.noise_std,
                    "seed": args.seed,
                    "height_m": HEIGHT,
                    "gravity_m_s2": GRAVITY,
                },
                "single_round": run_round(context, decision, args.true_restitution, args.noise_std),
                "paired_seed_comparison": compare_candidates(
                    args.true_restitution, args.noise_std, args.seeds
                ),
            }
            result["candidate_id"] = result["single_round"]["candidate_id"]
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        print(f"Demo failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
