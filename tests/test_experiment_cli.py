import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "apps" / "experiment-design"
CONTEXT = APP / "example-context.json"


class ExperimentCliTests(unittest.TestCase):
    def test_selected_actuation_runs_in_reference_simulator(self) -> None:
        decision = {
            "candidate_id": "upward",
            "hypothesis": "It may delay impact.",
            "rationale": "Compare it with neutral actuation.",
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "decision.json"
            path.write_text(json.dumps(decision), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(APP / "simulate_choice.py"), str(CONTEXT),
                 str(path), "--samples", "5"],
                cwd=ROOT, capture_output=True, text=True, check=True,
            )
        experiment = json.loads(result.stdout)
        self.assertEqual(experiment["candidate_id"], "upward")
        self.assertEqual(experiment["applied_acceleration_m_s2"], 2.0)
        self.assertEqual(len(experiment["observations"]), 7)
        at_impact = [row for row in experiment["observations"]
                     if row["time_s"] == experiment["impact_time_s"]]
        self.assertEqual([row["side"] for row in at_impact], ["pre", "post"])
        self.assertLess(at_impact[0]["velocity_m_s"], 0)
        self.assertGreater(at_impact[1]["velocity_m_s"], 0)

    def test_simulator_rejects_actuation_outside_declared_limit(self) -> None:
        context = json.loads(CONTEXT.read_text(encoding="utf-8"))
        context["candidates"][0]["acceleration_m_s2"] = -20.0
        decision = {
            "candidate_id": "downward",
            "hypothesis": "Test",
            "rationale": "Test",
        }
        with TemporaryDirectory() as directory:
            context_path = Path(directory) / "context.json"
            decision_path = Path(directory) / "decision.json"
            context_path.write_text(json.dumps(context), encoding="utf-8")
            decision_path.write_text(json.dumps(decision), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(APP / "simulate_choice.py"), str(context_path),
                 str(decision_path)], cwd=ROOT, capture_output=True, text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("actuation limit", result.stderr)


if __name__ == "__main__":
    unittest.main()
