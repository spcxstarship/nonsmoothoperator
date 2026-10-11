import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
from tempfile import TemporaryDirectory
import unittest

try:
    import numpy as np
    from packages.research.ball_design import (grid, prediction, sensitivity, posterior,
        information_gain, posterior_trajectory, run_case, numerical_checks)
except ImportError:
    np = None

from packages.research.proposals import validate

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = json.loads((ROOT/"experiments/research-proposal.example.json").read_text())
spec = importlib.util.spec_from_file_location("research_loop", ROOT/"experiments/research_loop.py")
loop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loop)


class ProposalTests(unittest.TestCase):
    def test_rejects_executable_unknown_and_nonfinite_formula(self):
        for change in ({"feature": "__import__('os')"}, {"weight": float("nan")},
                       {"transform": "eval"}, {"weight": True}):
            proposal = copy.deepcopy(EXAMPLE)
            proposal["formula"][0].update(change)
            with self.assertRaises(ValueError):
                validate(proposal)
        proposal = copy.deepcopy(EXAMPLE)
        proposal["physics"] = "new gravity"
        with self.assertRaises(ValueError):
            validate(proposal)


@unittest.skipIf(np is None, "requires the scientific environment")
class NumericalResearchTests(unittest.TestCase):
    def test_prediction_matches_independent_reference_and_derivatives(self):
        from packages.dynamics.bouncing_ball import BouncingBall
        theta = grid(9)
        for u in (-4., 0., 4.):
            for t in (.2, .38, .46, .6):
                actual = np.array([z+BouncingBall(initial_height=1-z, restitution=e,
                                  applied_acceleration=u).state_at(t).height for z, e in theta])
                np.testing.assert_allclose(prediction(theta, u, t), actual, atol=1e-14)
                for j in range(2):
                    shift = np.zeros(2); shift[j] = 1e-6
                    numerical = (prediction(theta+shift, u, t)-prediction(theta-shift, u, t))/2e-6
                    np.testing.assert_allclose(sensitivity(theta, u, t)[:, j], numerical, atol=1e-7)

    def test_posterior_and_mixture_prediction(self):
        theta = grid(9)
        weights = posterior(theta, [(0., .1, .95)], .02)
        np.testing.assert_allclose(weights, np.ones(len(theta))/len(theta))
        weights = posterior(theta, [(0., .42, .16)], .02)
        self.assertAlmostEqual(weights.sum(), 1.)
        times = np.linspace(0, .62, 101)
        expected = np.array([weights@prediction(theta, -2., t, .85) for t in times])
        np.testing.assert_allclose(posterior_trajectory(theta, weights, -2., times, .85), expected, atol=1e-14)

    def test_information_gain_limits_and_noise_integration(self):
        self.assertEqual(information_gain(np.zeros(2), np.ones(2)/2, .02), 0)
        separated = information_gain(np.array([0., 1.]), np.ones(2)/2, .02)
        self.assertAlmostEqual(separated, np.log(2), places=8)
        means = np.array([0., .02])
        coarse = information_gain(means, np.ones(2)/2, .02)
        refined = information_gain(means, np.ones(2)/2, .02, 2)
        self.assertLess(abs(coarse-refined), 1e-7)
        self.assertGreater(coarse, 0)
        self.assertLess(coarse, np.log(2))

    def test_repeatability_and_equal_observation_budgets(self):
        config = {"seed": 12, "grid_size": 9, "world_count": 1}
        one = run_case(config, EXAMPLE, 0, 0)
        two = run_case(config, EXAMPLE, 0, 0)
        self.assertEqual(loop.scientific(one), loop.scientific(two))
        self.assertEqual(len(one["policies"]), 7)
        for rounds in one["policies"].values():
            self.assertEqual(len(rounds), 6)
            self.assertTrue(all(r["mean_heldout_rmse_m"] >= 0 for r in rounds))


@unittest.skipIf(np is None, "requires the scientific environment")
class ResearchLifecycleTests(unittest.TestCase):
    def test_snapshot_tamper_rejected_before_execution(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)/"run"
            loop.initialize(directory)
            (directory/"source/packages/research/ball_design.py").write_text("modified")
            with self.assertRaisesRegex(ValueError, "frozen artifact changed"):
                loop.step(directory, json.dumps(EXAMPLE).encode(), "test")
            self.assertEqual(loop.read(directory/"manifest.json")["jobs"], [])

    def test_timeout_is_recorded_and_unresolved_job_blocks_restart(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)/"run"
            loop.initialize(directory)
            manifest = loop.read(directory/"manifest.json")
            record = {"stdout": "out.txt", "stderr": "err.txt"}
            ok = loop.execute(directory, manifest, record,
                              [sys.executable, "-c", "import time; time.sleep(10)"], .05)
            self.assertFalse(ok)
            saved = loop.read(directory/"manifest.json")
            self.assertEqual(saved["jobs"][0]["status"], "timed_out")
            self.assertGreater(saved["execution_s"], 0)
            saved["jobs"][0]["status"] = "running"
            with self.assertRaisesRegex(ValueError, "unresolved running"):
                loop.verify(directory, saved)
            loop.verify(directory, saved, restarting=False)

    def test_exited_controller_cannot_leave_its_child_running(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)/"run"
            loop.initialize(directory)
            marker = Path(tmp)/"child-survived"
            child = "import time; from pathlib import Path; time.sleep(.3); Path("+repr(str(marker))+").write_text('alive')"
            parent = "import subprocess,sys; subprocess.Popen([sys.executable,'-c',"+repr(child)+"],stdin=subprocess.DEVNULL)"
            record = {"stdout": "out.txt", "stderr": "err.txt"}
            try:
                self.assertTrue(loop.execute(directory, loop.read(directory/"manifest.json"), record,
                                             [sys.executable, "-c", parent], 2))
                time.sleep(.4)
                self.assertFalse(marker.exists())
            finally:
                if record.get("pid"):
                    try:
                        os.killpg(record["pid"], signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_grid_and_seed_are_frozen_at_initialization(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)/"run"
            loop.initialize(directory, grid_size=161, seed=20261008)
            config = loop.read(directory/"config.json")
            self.assertEqual((config["grid_size"], config["seed"]), (161, 20261008))
            loop.verify(directory, loop.read(directory/"manifest.json"))
            with self.assertRaises(ValueError):
                loop.initialize(Path(tmp)/"bad", grid_size=10000)

    def test_external_controller_receives_prompt_and_stop_is_honored(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)/"run"
            loop.initialize(directory)
            reply = copy.deepcopy(EXAMPLE); reply["action"] = "stop"
            program = "import json,sys; p=json.load(sys.stdin); assert p['prompt_version']; print("+repr(json.dumps(reply))+")"
            loop.automatic(directory, [sys.executable, "-c", program])
            manifest = loop.read(directory/"manifest.json")
            self.assertEqual(manifest["status"], "stopped_by_controller")
            self.assertEqual(len(manifest["controller_calls"]), 1)
            self.assertEqual(manifest["jobs"], [])

    def test_controller_loop_gets_experiment_feedback_then_stops(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)/"run"
            loop.initialize(directory)
            proposal = copy.deepcopy(EXAMPLE)
            proposal["measurement_times_s"] = [.46]
            program = (
                "import json,sys; p=json.load(sys.stdin); r="+repr(proposal)+"; "
                "assert not p['history'] or p['history'][0]['completed_cases']==4; "
                "r['action']='stop' if p['history'] else 'experiment'; "
                "r['id']='stop-after-feedback' if p['history'] else r['id']; print(json.dumps(r))"
            )
            # This is a deterministic transport fixture, not an LLM experiment.
            loop.automatic(directory, [sys.executable, "-c", program])
            manifest = loop.read(directory/"manifest.json")
            self.assertEqual(manifest["status"], "stopped_by_controller")
            self.assertEqual(len(manifest["controller_calls"]), 2)
            self.assertEqual(len(manifest["jobs"]), 5)
            self.assertTrue(all(j["status"] == "completed" for j in manifest["jobs"]))


if __name__ == "__main__":
    unittest.main()
