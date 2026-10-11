"""Numerical checks for inverse fitting and hybrid neural constraints."""

import importlib.util
import unittest

HAS_SCIENTIFIC = all(importlib.util.find_spec(name) for name in ("numpy", "scipy", "torch"))
if HAS_SCIENTIFIC:
    import numpy as np
    import torch
    from packages.dynamics.ball_identification import Observations, fit_classical, predict
    from packages.dynamics.bouncing_ball import BouncingBall
    from packages.pinn.ball import BallPINN, tensor


@unittest.skipUnless(HAS_SCIENTIFIC, "install requirements-pinn.txt for learning tests")
class BallIdentificationTests(unittest.TestCase):
    def setUp(self):
        self.times = np.linspace(0.08, 0.70, 12)
        self.observations = [Observations(1., u, self.times,
            predict(0.12, 0.8, 1., u, self.times)[0]) for u in (-4., 0., 4.)]

    def test_vectorized_solution_agrees_with_reference_on_both_sides(self):
        for u in (-4., 0., 4.):
            ball = BouncingBall(initial_height=0.88, applied_acceleration=u)
            times = np.array([0, ball.impact_time - 1e-8, ball.impact_time, ball.impact_time + 1e-8, .7])
            h, v = predict(.12, .8, 1., u, times)
            for i, t in enumerate(times):
                reference = ball.state_at(t)
                self.assertAlmostEqual(h[i], reference.height + .12)
                self.assertAlmostEqual(v[i], reference.velocity)

    def test_classical_recovers_noiseless_parameters(self):
        fit = fit_classical(self.observations, (0, .25), (.6, .95))
        self.assertAlmostEqual(fit["floor_m"], .12, places=7)
        self.assertAlmostEqual(fit["restitution"], .8, places=7)
        self.assertGreater(min(fit["jacobian_singular_values"]), 0)

    def test_pinn_initial_conditions_reset_and_gradients(self):
        model = BallPINN(self.observations,
            {"floor_bounds_m": [0, .25], "restitution_bounds": [.6, .95],
             "initial_floor_m": .05, "initial_restitution": .65},
            {"width": 8, "collocation_points_per_segment": 12,
             "physics_weight": 100., "guard_weight": 100.})
        for index in range(3):
            h0, v0 = model.segment(index, tensor([[0.]]))
            self.assertAlmostEqual(h0.item(), 1.)
            self.assertEqual(v0.item(), 0.)
            _, vpre = model.segment(index, tensor([[1.]]))
            hpost, vpost = model.segment(index, tensor([[0.]]), post=True)
            self.assertEqual(hpost.item(), model.floor.item())
            self.assertAlmostEqual(vpost.item(), (-model.restitution * vpre).item())
        model.loss().backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(model.raw_times.grad.abs().sum().item(), 0)

    def test_rejects_second_impact_and_nonfinite_times(self):
        for times in (np.array([0., 2.]), np.array([float("nan")])):
            with self.assertRaises(ValueError):
                predict(.12, .8, 1., -4., times)


if __name__ == "__main__":
    unittest.main()
