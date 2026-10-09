"""Regressions derived from the exposed failed fit, not new validation cases."""
import unittest
import numpy as np
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('event_regression_physics', Path(__file__).resolve().parents[1] / 'experiments/stick_slip_offgrid.py')
physics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(physics)


class EventRegressionTests(unittest.TestCase):
    parameters = {'class': 'static_speed', 'friction': .1624999998113468,
                  'restitution': .8052072595484914,
                  'spring': 2.2974951512564976, 'damping': .22}

    def test_stop_crossing_force_boundary_matches_independent_integration(self):
        # Includes least_squares' inward displacement from the damping bound.
        # The frozen failure leaves a legitimate ~1.9e-12 s interval before t=3.
        times = np.unique(np.r_[physics.TIMES, 3 + np.array([-1e-8, 0, 1e-8])])
        for key in ['friction', 'restitution', 'spring', 'damping']:
            for delta in [-1e-7, -1e-9, -8e-12, 0., 8e-12, 1e-9, 1e-7]:
                with self.subTest(parameter=key, delta=delta):
                    p = {**self.parameters, key: self.parameters[key] + delta}
                    actual, events = physics.simulate(p, physics.PROFILES['release_rest'], times)
                    expected, impacts = physics.reference(p, physics.PROFILES['release_rest'], times)
                    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0)
                    found = [e for e in events if e['kind'] in ['right', 'left']]
                    self.assertEqual([e['kind'] for e in found], [e[1] for e in impacts])
                    np.testing.assert_allclose([e['time'] for e in found],
                                               [e[0] for e in impacts], atol=1e-9, rtol=0)

    def test_collision_immediately_after_force_boundary_is_not_zero_root(self):
        p = physics.bank()[0]
        unit = physics.slip(1., 0., 0., p['friction'] + 1., p, 1.)[0]
        force = p['friction'] + .25 / unit
        for offset in range(-6, 7):
            with self.subTest(offset=offset):
                profile = [force + offset * np.spacing(force), 0., 0., 0.]
                actual, events = physics.simulate(p, profile)
                expected, _ = physics.reference(p, profile, physics.TIMES)
                np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0)
                impacts = [e for e in events if e['kind'] in ['right', 'left']]
                self.assertEqual(len(impacts), 1)
                self.assertEqual(impacts[0]['kind'], 'right')
                self.assertGreater(impacts[0]['incoming_velocity'], 0.)

    def test_exact_initial_state_and_declared_contact_collapse(self):
        p = {**physics.bank()[0], 'restitution': .2}
        x, v = physics.slip(0., .25, -1e-12, 2., p, -1.)
        self.assertEqual(x, .25)
        self.assertEqual(v, -1e-12)
        actual, events = physics.simulate(p, [2.] * 4)
        expected, _ = physics.reference(p, [2.] * 4, physics.TIMES)
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0)
        impacts = [e for e in events if e['kind'] == 'right']
        self.assertGreater(len(impacts), 4)
        self.assertTrue(all(e['incoming_velocity'] > 0 for e in impacts))
        self.assertTrue(np.all(np.diff([e['time'] for e in impacts]) > 0))
        self.assertEqual(sum(e['kind'] == 'low_speed_contact_collapse' for e in events), 1)
        self.assertEqual(actual[-1], .25)


if __name__ == '__main__':
    unittest.main()
