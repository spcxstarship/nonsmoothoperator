"""Validation orchestration tests; synthetic selector fixtures are SMOKE only."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('validation', Path(__file__).resolve().parents[1] / 'experiments/stick_slip_revalidation.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


class RevalidationTests(unittest.TestCase):
    def test_generation_is_prespecified_and_independent(self):
        cases = v.generated_cases()
        self.assertEqual(cases, v.generated_cases())
        self.assertEqual(len(cases), 24)
        self.assertEqual(len({json.dumps(c['truth'], sort_keys=True) for c in cases}), 8)
        old = [c['truth'] for c in v.physics.offgrid_cases()] + v.core.calibration_cases()
        previous = v.np.random.Generator(v.np.random.PCG64(2026100807)).uniform(.08, .92, (8,4))
        old += [v.core.parameters(v.core.CLASSES[i//2], z) for i,z in enumerate(previous)]
        for i, case in enumerate(cases):
            self.assertNotIn(case['truth'], old)
            self.assertEqual(case['case_id'], f'case-{i+1}')
            self.assertEqual(case['system_id'], f'system-{i//3+1}')
            self.assertEqual(case['replicate'], i % 3 + 1)
            self.assertEqual(case['baseline_seed'], 7100000+i)
            self.assertEqual(case['candidate_seed_base'], 7200000+10000*i)
            self.assertEqual(case['random_seed'], 7300000+i)
            self.assertTrue(all(.08 <= z <= .92 for z in case['normalized_truth']))
            self.assertEqual(case['truth'], cases[3*(i//3)]['truth'])
        seeds = [c['candidate_seed_base']+100*r+j for c in cases for r in [1,2,3] for j in range(10)]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertTrue(set(seeds).isdisjoint(c['baseline_seed'] for c in cases))

    def test_frozen_dependency_identity_and_context_isolation(self):
        self.assertEqual(v.CORE_PATH.parent, v.HERE.parent)
        for name in v.DEPENDENCIES:
            self.assertTrue((v.CORE_PATH.parent/name).exists())
        history = v.core.calibration_history(v.core.calibration_cases()[0])[:1]
        analysis = v.core.analyze_history(history)
        context = json.dumps(v.core.context(history, analysis, 1))
        for forbidden in ['truth', 'heldout', 'threshold_probe', 'system_id', 'candidate_seed', 'numerical_disagreement']:
            self.assertNotIn(forbidden, context)
        for function in ['fit_class', 'analyze_history', 'fitted_ensemble', 'predictive_moments', 'numerical_decision', 'context', 'stage_metrics']:
            self.assertEqual(Path(getattr(v.core, function).__code__.co_filename), v.CORE_PATH)

    def test_freeze_precedes_physics_and_detects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'campaign'
            with patch.object(v.physics, 'simulate', side_effect=AssertionError('No simulation before freeze')), patch.object(v.core, 'analyze_history', side_effect=AssertionError('No fitting before freeze')):
                v.freeze(root)
            protocol = v.read(root/'protocol.json')
            self.assertEqual(protocol['maximum_selector_calls'], 72)
            self.assertEqual(len(protocol['source_hashes']), 6)
            p = root/'sources/experiments/stick_slip_continuous.py'
            p.write_text(p.read_text()+'\n')
            with self.assertRaises(ValueError):
                v.verify(root)

    def test_smoke_calibration_only_end_to_end(self):
        # Explicitly fake choices exercise acquisition plumbing; no model calls,
        # validation parameter vectors, or validation performance development.
        truth = v.core.calibration_cases()[0]
        cases = [{'case_id': f'case-{i+1}', 'system_id': 'SMOKE-calibration-system',
                  'replicate': i+1, 'truth': truth, 'baseline_seed': 991000+i,
                  'candidate_seed_base': 992000+10000*i, 'random_seed': 993000+i}
                 for i in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'SMOKE-NOT-VALIDATION'
            with patch.object(v, 'generated_cases', return_value=cases):
                v.freeze(root)
            v.preflight(root)
            self.assertEqual(len(v.read(root/'reference-verification.json')), 14)
            self.assertFalse((root/'fit-cache').exists())
            v.prepare(root)
            for r in [1,2,3]:
                entries = []
                for case in cases:
                    decision = Path(directory)/f'SMOKE-{r}-{case["case_id"]}.json'
                    metadata = decision.with_suffix('.metadata.json')
                    v.write(decision, {'candidate_id': v.physics.FIXED[r-1], 'hypothesis': 'SMOKE fixture', 'rationale': 'Explicitly fake; no selector claim'})
                    v.write(metadata, {'SMOKE': True, 'actual_selector_call': False})
                    entries.append({'case_id': case['case_id'], 'decision': str(decision), 'metadata': str(metadata)})
                mapping = Path(directory)/f'SMOKE-map-{r}.json'
                v.write(mapping, entries)
                if r == 1:
                    invalid = Path(directory)/'duplicate-map.json'
                    v.write(invalid, [entries[0]]*len(entries))
                    with self.assertRaises(ValueError):
                        v.acquire(root, r, invalid)
                    self.assertFalse((root/f'round-{r}-lock.json').exists())
                v.acquire(root, r, mapping)
            v.verify(root)
            result = v.read(root/'results.json')
            self.assertEqual(len(result['cases']), 3)
            self.assertEqual(len(result['system_summaries']), 1)
            for row in result['cases']:
                self.assertEqual(row['policies']['fresh_llm']['choices'], v.physics.FIXED)
                a = row['policies']['fresh_llm']['stages'][-1]['heldout_position_rmse_m']
                b = row['policies']['fixed_balanced']['stages'][-1]['heldout_position_rmse_m']
                self.assertEqual(a,b)
                for policy in v.POLICIES:
                    self.assertEqual(len(v.read(root/'cases'/row['case_id']/policy/'history-3.json')), 4)
            self.assertEqual(result['paired_comparisons']['fixed_balanced']['descriptive_block_bootstrap_95pct_m'], [0.,0.])

    def test_analysis_implementation_is_sealed_and_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'campaign'
            analysis = Path(directory)/'analysis'
            analysis.mkdir()
            names = ['record_selectors.py','verify_independent.py','check_fitted_reference.py','report_results.py']
            for name in names:
                (analysis/name).write_text('# SMOKE analysis archive fixture\n')
            v.freeze(root, analysis)
            protocol = v.read(root/'protocol.json')
            self.assertEqual(len(protocol['analysis_sources']), 4)
            for name in names:
                self.assertEqual((root/'sources/analysis'/name).read_bytes(), (analysis/name).read_bytes())
            (root/'sources/analysis/report_results.py').write_text('# changed\n')
            with self.assertRaises(ValueError):
                v.verify(root)


if __name__ == '__main__':
    unittest.main()
