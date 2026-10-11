import importlib.util
import json
from pathlib import Path
import unittest


SELECTOR_PATH = (
    Path(__file__).resolve().parents[1] / "apps" / "experiment-design" / "selector.py"
)
spec = importlib.util.spec_from_file_location("experiment_selector", SELECTOR_PATH)
selector = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = selector
spec.loader.exec_module(selector)


class SelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = {
            "objective": "Reduce held-out impact-time error",
            "candidates": [{"id": "low"}, {"id": "high"}],
            "history": [],
            "diagnostics": {},
        }

    def test_accepts_a_choice_from_the_candidate_set(self) -> None:
        ids = selector.validate_context(self.context)
        response = json.dumps({
            "candidate_id": "high",
            "hypothesis": "A stronger input may change impact timing.",
            "rationale": "It has not been tried yet.",
        })
        self.assertEqual(selector.parse_decision(response, ids).candidate_id, "high")
        self.assertEqual(selector.build_messages(self.context)[0]["role"], "system")

    def test_rejects_invalid_or_unapproved_choices(self) -> None:
        ids = selector.validate_context(self.context)
        with self.assertRaises(ValueError):
            selector.parse_decision("not json", ids)
        with self.assertRaises(ValueError):
            selector.parse_decision(json.dumps({
                "candidate_id": "unsafe-new-input",
                "hypothesis": "test",
                "rationale": "test",
            }), ids)
        with self.assertRaises(ValueError):
            selector.validate_context({**self.context, "candidates": [{"id": "x"}, {"id": "x"}]})


if __name__ == "__main__":
    unittest.main()
