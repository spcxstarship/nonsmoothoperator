"""Acquisition blinding, frozen artifacts, and numerical-design checks."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location(
    "bouc_wen_selection", Path(__file__).resolve().parents[1]/"experiments/bouc_wen_selection.py")
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


class SelectionTests(unittest.TestCase):
    def test_numerical_selectors_refine_and_information_is_positive(self):
        p = [.23, .91, .67, 1.31]
        a, b = s.numerical_scores(p, .001), s.numerical_scores(p, .0005)
        self.assertEqual(a["choices"], b["choices"])
        for candidate in s.bw.CANDIDATES:
            for policy in ("fisher_d", "prediction_variance"):
                self.assertGreater(b["scores"][candidate][policy], 0)
                np.testing.assert_allclose(a["scores"][candidate][policy], b["scores"][candidate][policy], rtol=.002)

    def test_decisions_require_every_case_once_and_no_extra_fields(self):
        rows = [{"case_id": f"case-{i}", "candidate_id": "mixed_cycles", "rationale": "Example."}
                for i in range(1, 9)]
        self.assertEqual(s.validate_decisions(json.dumps(rows)), rows)
        bad = json.loads(json.dumps(rows))
        bad[-1]["case_id"] = "case-1"
        with self.assertRaises(ValueError):
            s.validate_decisions(json.dumps(bad))
        rows[0]["true_parameters"] = [1]*4
        with self.assertRaises(ValueError):
            s.validate_decisions(json.dumps(rows))

    def test_cannot_evaluate_without_recorded_llm_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(s, "prepared_check"):
                with self.assertRaises(FileNotFoundError):
                    s.evaluate(Path(directory))
            self.assertFalse((Path(directory)/"evaluation-start.json").exists())

    def test_cannot_replace_recorded_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"decision.json"
            s.write(path, {"choice": "first"})
            with self.assertRaises(FileExistsError):
                s.write(path, {"choice": "second"})
            self.assertEqual(s.read(path), {"choice": "first"})

    def test_cannot_change_choices_after_evaluation_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            s.write(out/"evaluation-start.json", {})
            with patch.object(s, "prepared_check"):
                with self.assertRaises(RuntimeError):
                    s.record(out, "[]")


if __name__ == "__main__":
    unittest.main()
