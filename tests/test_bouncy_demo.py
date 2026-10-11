import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "apps" / "experiment-design" / "bouncy_demo.py"
DECISION = {
    "candidate_id": "downward",
    "hypothesis": "A faster impact may clarify restitution.",
    "rationale": "The speed is highest for this candidate.",
}


def demo(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(DEMO), "--true-restitution", "0.8",
         "--noise-std", "0.3", "--seed", "7", *args],
        cwd=ROOT, capture_output=True, text=True,
    )


class BouncyDemoTests(unittest.TestCase):
    def test_selection_context_and_paired_comparison(self) -> None:
        prepared = demo("prepare")
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        self.assertNotIn("true_restitution", prepared.stdout)
        context = json.loads(prepared.stdout)

        with TemporaryDirectory() as directory:
            context_path = Path(directory) / "context.json"
            decision_path = Path(directory) / "decision.json"
            context_path.write_text(prepared.stdout, encoding="utf-8")
            decision_path.write_text(json.dumps(DECISION), encoding="utf-8")
            result = demo("evaluate", str(context_path), str(decision_path),
                          "--seeds", "100")

        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        round_result = report["single_round"]
        self.assertEqual(round_result["candidate_id"], "downward")
        self.assertEqual(round_result["observations"][0], context["history"][0])
        self.assertLess(round_result["after"]["standard_error"],
                        round_result["before"]["standard_error"])
        rmse = report["paired_seed_comparison"]["parameter_rmse_by_candidate"]
        self.assertLess(rmse["downward"], rmse["neutral"])
        self.assertLess(rmse["neutral"], rmse["upward"])

    def test_rejects_tampered_candidate_actuation(self) -> None:
        context = json.loads(demo("prepare").stdout)
        context["candidates"][0]["acceleration_m_s2"] = -3.0
        with TemporaryDirectory() as directory:
            context_path = Path(directory) / "context.json"
            decision_path = Path(directory) / "decision.json"
            context_path.write_text(json.dumps(context), encoding="utf-8")
            decision_path.write_text(json.dumps(DECISION), encoding="utf-8")
            result = demo("evaluate", str(context_path), str(decision_path))

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match its ID", result.stderr)


if __name__ == "__main__":
    unittest.main()
