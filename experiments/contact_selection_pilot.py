"""Frozen numerical worker used by the research loop; outputs development evidence."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.research.ball_design import numerical_checks, run_case
from packages.research.proposals import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--proposal", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--world", type=int, default=0)
    parser.add_argument("--regime", type=int, default=0)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    proposal = validate(json.loads(args.proposal.read_text()))
    if args.check:
        result = numerical_checks(config, proposal)
    else:
        result = run_case(config, proposal, args.world, args.regime)
        refined = run_case(config, proposal, args.world, args.regime,
                           size=2*config["grid_size"]-1, refinement=2)
        comparisons = {}
        for policy, rounds in result["policies"].items():
            other = refined["policies"][policy]
            differences = [abs(a["mean_heldout_rmse_m"]-b["mean_heldout_rmse_m"])
                           for a, b in zip(rounds, other)]
            individual = [abs(x-y) for a, b in zip(rounds, other)
                          for x, y in zip(a["heldout_rmse_m"], b["heldout_rmse_m"])]
            comparisons[policy] = {
                "max_rmse_change_m": max(differences),
                "max_individual_rmse_change_m": max(individual),
                "choices_identical": [r.get("u") for r in rounds] == [r.get("u") for r in other],
                "within_tolerance": max(individual) <= config["refinement_rmse_tolerance_m"],
            }
        result["refinement"] = {"grid_size": refined["grid_size"],
                                "policies": refined["policies"], "comparison": comparisons}
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
