import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "apps" / "experiment-design" / "round_store.py"
CONTEXT = ROOT / "apps" / "experiment-design" / "example-context.json"


def run_store(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(STORE), *args], cwd=ROOT,
        capture_output=True, text=True,
    )


class RoundStoreTests(unittest.TestCase):
    def test_records_model_agnostic_round_and_prevents_overwrite(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            instructions = root / "instructions-source.txt"
            instructions.write_text("Choose one candidate from the context.\n",
                                    encoding="utf-8")
            started = run_store(
                "start", "--root", str(root), "--study", "bounce",
                "--trial", "heuristic-seed-7", "--round", "001",
                "--context", str(CONTEXT), "--selector", "impact-speed-rule",
                "--instructions", str(instructions),
            )
            self.assertEqual(started.returncode, 0, started.stderr)
            round_dir = root / "bounce" / "heuristic-seed-7" / "001"
            self.assertEqual((round_dir / "context.json").read_bytes(), CONTEXT.read_bytes())
            self.assertEqual((round_dir / "instructions.txt").read_bytes(),
                             instructions.read_bytes())
            self.assertEqual(json.loads((round_dir / "manifest.json").read_text())["status"], "open")
            self.assertNotEqual(run_store(
                "start", "--root", str(root), "--study", "bounce",
                "--trial", "heuristic-seed-7", "--round", "001",
                "--context", str(CONTEXT), "--selector", "impact-speed-rule",
            ).returncode, 0)

            decision = {"candidate_id": "downward", "hypothesis": "Higher impact speed.",
                        "rationale": "Maximize measurement information."}
            decision_path = root / "decision-source.json"
            result_path = root / "result-source.json"
            decision_path.write_text(json.dumps(decision), encoding="utf-8")
            result_path.write_text(json.dumps({"candidate_id": "downward", "metric": 0.1}),
                                   encoding="utf-8")
            finished = run_store("finish", str(round_dir), "--decision",
                                 str(decision_path), "--result", str(result_path))
            self.assertEqual(finished.returncode, 0, finished.stderr)
            manifest = json.loads((round_dir / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["selector"], {
                "id": "impact-speed-rule", "model_id": None, "prompt_version": None,
            })
            self.assertIsNone(manifest["usage"])
            self.assertEqual(manifest["selected_candidate_id"], "downward")
            for filename, expected in manifest["files_sha256"].items():
                self.assertEqual(hashlib.sha256((round_dir / filename).read_bytes()).hexdigest(),
                                 expected)
            self.assertNotEqual(run_store("finish", str(round_dir), "--decision",
                                          str(decision_path), "--result",
                                          str(result_path)).returncode, 0)

    def test_rejects_result_for_different_candidate(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            started = run_store("start", "--root", str(root), "--study", "bounce",
                                "--trial", "t1", "--round", "r1", "--context",
                                str(CONTEXT), "--selector", "any-model")
            self.assertEqual(started.returncode, 0, started.stderr)
            decision_path = root / "decision.json"
            result_path = root / "result.json"
            decision_path.write_text(json.dumps({
                "candidate_id": "downward", "hypothesis": "Test", "rationale": "Test",
            }), encoding="utf-8")
            result_path.write_text(json.dumps({"candidate_id": "upward"}), encoding="utf-8")
            round_dir = root / "bounce" / "t1" / "r1"
            finished = run_store("finish", str(round_dir), "--decision",
                                 str(decision_path), "--result", str(result_path))
            self.assertNotEqual(finished.returncode, 0)
            self.assertIn("must match", finished.stderr)
            self.assertFalse((round_dir / "decision.json").exists())
            result_path.write_text(json.dumps({"candidate_id": "downward"}),
                                   encoding="utf-8")
            (round_dir / "context.json").write_text("{}", encoding="utf-8")
            tampered = run_store("finish", str(round_dir), "--decision",
                                 str(decision_path), "--result", str(result_path))
            self.assertNotEqual(tampered.returncode, 0)
            self.assertIn("has changed", tampered.stderr)


if __name__ == "__main__":
    unittest.main()
