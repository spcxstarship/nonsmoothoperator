"""Run a selected constant-actuation experiment in the reference simulator."""

import argparse
import json
from math import isfinite
from pathlib import Path
import sys

from selector import parse_decision, validate_context

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packages.dynamics.bouncing_ball import BouncingBall


def simulate_choice(context: dict, decision_text: str, samples: int = 41) -> dict:
    if samples < 3:
        raise ValueError("samples must be at least 3")
    decision = parse_decision(decision_text, validate_context(context))
    candidate = next(
        item for item in context["candidates"] if item["id"] == decision.candidate_id
    )
    limit = context.get("max_abs_acceleration_m_s2")
    acceleration = candidate.get("acceleration_m_s2")
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not isfinite(limit) or limit <= 0:
        raise ValueError("context needs a positive finite max_abs_acceleration_m_s2")
    if isinstance(acceleration, bool) or not isinstance(acceleration, (int, float)) or not isfinite(acceleration):
        raise ValueError("candidate needs a finite acceleration_m_s2")
    if abs(acceleration) > limit:
        raise ValueError("candidate exceeds the declared actuation limit")

    ball = BouncingBall(applied_acceleration=float(acceleration))
    # Stop halfway through the upward/downward rebound, before a second impact.
    end_time = (ball.impact_time + ball.next_impact_time) / 2
    rows = []
    for index in range(samples):
        time = index * end_time / (samples - 1)
        state = ball.state_at(time)
        rows.append({"time_s": time, "height_m": state.height, "velocity_m_s": state.velocity})
    before = ball.state_at(ball.impact_time, side="pre")
    after = ball.state_at(ball.impact_time, side="post")
    rows.extend([
        {"time_s": ball.impact_time, "height_m": before.height, "velocity_m_s": before.velocity, "side": "pre"},
        {"time_s": ball.impact_time, "height_m": after.height, "velocity_m_s": after.velocity, "side": "post"},
    ])
    rows.sort(key=lambda row: (row["time_s"], 0 if row.get("side") == "pre" else 1))
    return {
        "candidate_id": decision.candidate_id,
        "applied_acceleration_m_s2": acceleration,
        "impact_time_s": ball.impact_time,
        "observations": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("context", type=Path)
    parser.add_argument("decision", type=Path)
    parser.add_argument("--samples", type=int, default=41)
    args = parser.parse_args()
    try:
        context = json.loads(args.context.read_text(encoding="utf-8"))
        decision = args.decision.read_text(encoding="utf-8")
        print(json.dumps(simulate_choice(context, decision, args.samples), indent=2))
        return 0
    except (OSError, ValueError, TypeError) as error:
        print(f"Simulation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
