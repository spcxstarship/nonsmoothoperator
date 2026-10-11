"""Physics, experimental budgets, and fit-input isolation for the pilot."""
import importlib.util
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import numpy as np

from packages.dynamics import bouc_wen as bw

spec = importlib.util.spec_from_file_location(
    "bouc_wen_pilot", Path(__file__).resolve().parents[1] / "experiments/bouc_wen_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class BoucWenTests(unittest.TestCase):
    def test_memoryless_limit_matches_analytic_forced_oscillator(self):
        # rho=0 => w=0. Undamped x''+k*x=A*sin(omega*t), zero initial state.
        t = np.linspace(0, 16, 161)
        k = 1.3
        omega = 2*np.pi/8
        exact = .8/(k-omega**2)*(np.sin(omega*t)
                                 - omega/np.sqrt(k)*np.sin(np.sqrt(k)*t))
        actual = bw.simulate([0, k, .8, 0], "slow_cycles", t)
        np.testing.assert_allclose(actual[:, 0], exact, atol=2e-7)
        np.testing.assert_array_equal(actual[:, 2], 0)

    def test_reference_agreement_and_memory_bound(self):
        p = [.2, .9, .7, 1.3]
        for name in bw.PROFILES:
            a = bw.simulate(p, name)
            b = bw.simulate(p, name, reference=True)
            self.assertLess(np.max(np.abs(a[:, 0]-b[:, 0])), 2e-4)
            self.assertLessEqual(np.max(np.abs(b[:, 2])), 1+1e-8)

    def test_equal_candidate_energy_and_peak_limit(self):
        t = np.linspace(0, 16, 4097)
        for name in bw.CANDIDATES:
            force = np.array([bw.forcing(name, x) for x in t])
            self.assertAlmostEqual(np.trapezoid(force**2, t)/16, .32, places=10)
            self.assertLessEqual(np.max(abs(force)), 1.4)

    def test_reserved_data_and_metadata_rejected_before_simulation(self):
        for history in ([{"profile": "chirp", "positions": [0.]*64}],
                        [{"profile": "baseline", "positions": [0.]*64, "truth": [1]*4}]):
            with patch.object(bw, "simulate", side_effect=AssertionError("Must not simulate")):
                with self.assertRaises(ValueError):
                    pilot.fit(history, deadline=time.monotonic()+10)

    def test_timeout_is_propagated(self):
        with self.assertRaises(TimeoutError):
            pilot.fit([{"profile": "baseline", "positions": [0.]*64}],
                      deadline=time.monotonic()-1)


if __name__ == "__main__":
    unittest.main()
