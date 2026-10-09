"""Continuous fit recovery, diagnostics, and strict holdout isolation."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import copy
import json
import tempfile
import numpy as np

spec = importlib.util.spec_from_file_location(
    'continuous', Path(__file__).resolve().parents[1] / 'experiments/stick_slip_continuous.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class ContinuousTests(unittest.TestCase):
    def test_calibration_noiseless_known_class_recovery(self):
        for p in c.calibration_cases():
            fit = c.fit_class(c.calibration_history(p), p['class'])
            self.assertTrue(fit['best_converged'])
            self.assertLess(fit['best']['sse_m2'], 1e-15)
            self.assertEqual(fit['best']['local_jacobian_rank'], 4)
            np.testing.assert_allclose([fit['best']['parameters'][k] for k in c.PARAMETERS],
                                       [p[k] for k in c.PARAMETERS], atol=1e-6)
            self.assertGreater(fit['residual_calls'], sum(r['nfev'] for r in fit['starts']))

    def test_noisy_fit_diagnostics_and_reserved_isolation(self):
        p = c.calibration_cases()[2]
        history = c.calibration_history(p, noise=True)
        original = c.physics.simulate
        acquired = [c.physics.BASELINE, *c.PROFILES.values()]
        def guarded(parameters, profile, **kw):
            self.assertIn(profile, acquired)
            return original(parameters, profile, **kw)
        with patch.object(c.physics, 'simulate', side_effect=guarded):
            fit = c.fit_class(history, p['class'])
            ensemble = c.fitted_ensemble(history, fit)
        self.assertGreater(len(ensemble['members']), 1)
        self.assertEqual(len(fit['best']['raw_residuals_m']), 64)
        self.assertLess(fit['best']['standardized_sse'], 100.)
        for reserved in c.physics.RESERVED:
            with self.assertRaises(ValueError):
                c.fit_class([{'candidate_id': reserved, 'positions': [0.] * 16}], p['class'])

    def test_baseline_preserves_box_and_zero_rank(self):
        history = [{'candidate_id': 'baseline', 'positions': [0.] * 16}]
        for label in c.CLASSES:
            fit = c.fit_class(history, label)
            self.assertEqual(fit['best']['local_jacobian_rank'], 0)
            members = c.fitted_ensemble(history, fit)['members']
            coords = np.array([p['normalized_parameters'] for p in members])
            np.testing.assert_array_equal(coords.min(axis=0), np.zeros(4))
            np.testing.assert_array_equal(coords.max(axis=0), np.ones(4))

    def test_simulation_failure_is_not_physical_mismatch(self):
        history = [{'candidate_id': 'baseline', 'positions': [0.] * 16}]
        with patch.object(c.physics, 'simulate', side_effect=RuntimeError('synthetic failure')):
            fit = c.fit_class(history, c.CLASSES[0])
        self.assertTrue(fit['all_starts_failed'])
        self.assertIsNone(fit['best'])
        self.assertTrue(all('traceback' in r for r in fit['starts']))

    def test_no_impact_restitution_remains_unidentified(self):
        p = c.calibration_cases()[0]
        values, events = c.physics.simulate(p, c.PROFILES['threshold_steps'])
        self.assertFalse(any(e['kind'] in ['left', 'right'] for e in events))
        history = [{'candidate_id': 'threshold_steps', 'positions': values.tolist()}]
        fit = c.fit_class(history, p['class'])
        self.assertLess(fit['best']['local_jacobian_rank'], 4)
        ensemble = c.fitted_ensemble(history, fit)
        restitution = [m['parameters']['restitution'] for m in ensemble['members']]
        self.assertAlmostEqual(min(restitution), c.LOWER[1])
        self.assertAlmostEqual(max(restitution), c.UPPER[1])

    def test_selector_context_does_not_disclose_future_or_test_data(self):
        history = [{'candidate_id': 'baseline', 'positions': [0.] * 16}]
        analysis = c.analyze_history(history)
        context = c.context(history, analysis, 1)
        import json
        text = json.dumps(context)
        for forbidden in ['truth', 'heldout', 'threshold_probe', 'candidate_seed',
                          'incoming_velocity', 'numerical_disagreement', 'posterior_probability']:
            self.assertNotIn(forbidden, text)
        self.assertEqual(len(context['candidates']), 10)

    def test_member_count_does_not_change_class_weight_and_variance_decomposes(self):
        import copy
        analysis = c.analyze_history([{'candidate_id': 'baseline', 'positions': [0.] * 16}])
        original = c.predictive_moments(analysis, c.PROFILES['impact_reversal'])
        duplicate = copy.deepcopy(analysis)
        duplicate['ensembles'][c.CLASSES[0]]['members'] *= 3
        changed = c.predictive_moments(duplicate, c.PROFILES['impact_reversal'])
        for a, b in zip(original, changed):
            np.testing.assert_allclose(a, b, atol=1e-15, rtol=0)
        decision = c.numerical_decision(analysis)
        for score in decision['scores'].values():
            self.assertAlmostEqual(score['total'], score['within_class'] + score['between_class'])
        self.assertGreater(decision['scores']['impact_reversal']['within_class'], 0)

    def test_promoted_probe_preserves_score_and_filters_support(self):
        p = c.calibration_cases()[0]
        history = c.calibration_history(p)
        fit = c.fit_class(history, p['class'], max_nfev=1)
        z = [(p[k] - lo) / (hi-lo) for k, lo, hi in zip(c.PARAMETERS, c.LOWER, c.UPPER)]
        probe = {'parameters': p, 'normalized_parameters': z, 'standardized_sse': 0.}
        support = {'members': [probe, {'parameters': p, 'normalized_parameters': [0.] * 4,
                                      'standardized_sse': 100.}]}
        c.promote_probe(history, fit, support, probe)
        self.assertTrue(fit['unresolved_optimization'])
        self.assertFalse(fit['best_converged'])
        self.assertLess(fit['best']['standardized_sse'], 1e-20)
        self.assertEqual(len(support['members']), 1)
        self.assertEqual(fit['best']['local_jacobian_rank'], 4)
        self.assertEqual(fit['best']['status'], -98)

    def test_failed_repair_retains_finite_candidate_and_failure_trace(self):
        p = c.calibration_cases()[0]
        history = c.calibration_history(p)
        fit = c.fit_class(history, p['class'], max_nfev=1)
        z = (np.array([p[k] for k in c.PARAMETERS])-c.LOWER)/(c.UPPER-c.LOWER)
        probe = {'parameters': p, 'normalized_parameters': z.tolist(), 'standardized_sse': 0.}
        support = {'members': [probe], 'simulator_calls': 0, 'elapsed_s': 0.}
        failure = {'best': None, 'starts': [{'optimizer_success': False, 'traceback': 'simulator failure stack'}],
                   'elapsed_s': 0., 'residual_calls': 1, 'simulator_calls': 1, 'converged_starts': 0}
        with patch.object(c, 'CLASSES', [p['class']]), \
             patch.object(c, 'fit_history', return_value={p['class']: copy.deepcopy(fit)}), \
             patch.object(c, 'fitted_ensemble', return_value=copy.deepcopy(support)), \
             patch.object(c, 'minimize', return_value=SimpleNamespace(success=False, message='budget', fun=0.)), \
             patch.object(c, 'fit_class', return_value=failure):
            result = c.analyze_history(history)
        retained = result['fits'][p['class']]
        self.assertFalse(retained['best_converged'])
        self.assertTrue(retained['unresolved_optimization'])
        self.assertLess(retained['best']['standardized_sse'], 1e-20)
        self.assertEqual(retained['ensemble_repairs'][0]['outcome'], 'verified_probe_retained_unresolved')
        self.assertIn('simulator failure stack', json.dumps(retained))

    def test_unavailable_probe_derivatives_do_not_discard_finite_value(self):
        p = c.calibration_cases()[0]
        history = c.calibration_history(p)
        fit = c.fit_class(history, p['class'], max_nfev=1)
        z = (np.array([p[k] for k in c.PARAMETERS])-c.LOWER)/(c.UPPER-c.LOWER)
        probe = {'parameters': p, 'normalized_parameters': z.tolist(), 'standardized_sse': 0.}
        support = {'members': [probe]}
        original = c.physics.simulate
        def isolated_candidate(par, profile):
            if any(abs(par[k]-p[k]) > 1e-12 for k in c.PARAMETERS):
                raise RuntimeError('invalid adjacent trial')
            return original(par, profile)
        with patch.object(c.physics, 'simulate', side_effect=isolated_candidate):
            c.promote_probe(history, fit, support, probe)
        self.assertLess(fit['best']['standardized_sse'], 1e-20)
        self.assertIsNone(fit['best']['local_jacobian_rank'])
        self.assertIsNone(fit['best']['optimality'])
        self.assertFalse(fit['best']['diagnostic_derivatives_available'])
        self.assertEqual(len(fit['best']['diagnostic_derivative_failures']), 8)
        json.dumps(fit, allow_nan=False)

    def test_fatal_fit_persists_inner_failure_before_cache_write(self):
        history = [{'candidate_id': 'baseline', 'positions': [0.] * 16}]
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(c.physics, 'simulate', side_effect=RuntimeError('controlled event failure')):
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, 'diagnostics'):
                c.cached_analysis(root, history)
            files = list((root/'fit-failures').glob('*.json'))
            self.assertEqual(len(files), 1)
            failure = json.loads(files[0].read_text())
            self.assertIn('controlled event failure', json.dumps(failure['fit_diagnostics']))
            self.assertEqual(failure['history'], history)
            self.assertFalse((root/'fit-cache').exists())

    def test_fatal_support_failure_retains_fits(self):
        history = [{'candidate_id': 'baseline', 'positions': [0.] * 16}]
        with patch.object(c, 'fitted_ensemble', return_value={'members': [], 'simulator_calls': 0, 'elapsed_s': 0.}):
            with self.assertRaises(c.FitAnalysisError) as caught:
                c.analyze_history(history)
        self.assertEqual(set(caught.exception.fit_diagnostics), set(c.CLASSES))
        self.assertIn('Empty class sensitivity support', str(caught.exception))

    def test_timeout_is_never_treated_as_numerical_failure(self):
        history = [{'candidate_id': 'baseline', 'positions': [0.] * 16}]
        with patch.object(c.physics, 'simulate', side_effect=TimeoutError('active budget exhausted')):
            with self.assertRaises(TimeoutError):
                c.fit_class(history, c.CLASSES[0])
        with patch.object(c, 'fitted_ensemble', side_effect=TimeoutError('active budget exhausted')):
            with self.assertRaises(TimeoutError):
                c.analyze_history(history)


if __name__ == '__main__':
    unittest.main()
