"""Off-grid truth isolation, threshold behavior and misspecified posterior metrics."""
import importlib.util
from pathlib import Path
import unittest
import json
import numpy as np
spec=importlib.util.spec_from_file_location('offgrid',Path(__file__).resolve().parents[1]/'experiments/stick_slip_offgrid.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)

class OffgridTests(unittest.TestCase):
    def test_frozen_cases_are_offgrid_and_independent(self):
        cases=p.offgrid_cases();self.assertEqual(len(cases),8)
        for case in cases:
            self.assertNotIn(case['truth'],p.bank())
            for key in ['friction','restitution','spring','damping']:
                self.assertNotIn(case['truth'][key],{q[key] for q in p.bank()})
        for key in ['baseline_seed','candidate_seed_base','random_seed','mc_seed']:
            self.assertEqual(len({c[key] for c in cases}),8)
        for i in range(0,8,2):self.assertEqual(cases[i]['truth'],cases[i+1]['truth'])

    def test_baseline_ambiguous_threshold_reserved_separates(self):
        self.assertNotIn('threshold_probe',p.PROFILES)
        for c in p.offgrid_cases()[::2]:
            q=c['truth'];np.testing.assert_array_equal(p.simulate(q,p.BASELINE)[0],0)
            y,_=p.simulate(q,p.RESERVED['threshold_probe'])
            if q['class'].startswith('static'):np.testing.assert_array_equal(y,0)
            else:self.assertGreater(max(abs(y)),.01)

    def test_independent_reference_offgrid(self):
        for c in p.offgrid_cases()[::2]:
            for profile in [p.PROFILES['impact_reversal'],p.RESERVED['threshold_probe']]:
                a,_=p.simulate(c['truth'],profile);b,_=p.reference(c['truth'],profile,p.TIMES)
                np.testing.assert_allclose(a,b,atol=1e-7,rtol=0)

    def test_metrics_ties_wrong_confidence_and_extreme_logscore(self):
        particles=p.bank();pred={'baseline':np.zeros((32,16))};held={'threshold_probe':np.zeros((32,16))};truth={'class':p.CLASSES[-1],'threshold_probe':np.ones(16)}
        history=[{'candidate_id':'baseline','positions':[0.]*16}]
        m=p.stage_metrics(history,pred,held,truth,particles)
        self.assertTrue(m['map_tie']);self.assertEqual(m['map_tied_classes'],p.CLASSES);self.assertFalse(m['confident_wrong']);self.assertAlmostEqual(m['heldout_position_rmse_m'],1.)
        pred['baseline'][8:]=1.
        m=p.stage_metrics(history,pred,held,truth,particles)
        self.assertTrue(m['confident_wrong']);self.assertFalse(m['map_correct']);self.assertFalse(m['map_tie']);self.assertGreater(m['class_negative_logscore'],10000);self.assertTrue(np.isfinite(m['class_negative_logscore']))
        history[0]['positions']=[.5]*16
        self.assertAlmostEqual(p.stage_metrics(history,pred,held,truth,particles)['minimum_bank_sse_per_measurement'],.25)

    def test_context_excludes_truth_and_reserved_data(self):
        ctx=p.context([{'candidate_id':'baseline','positions':[0.]*16}],dict.fromkeys(p.CLASSES,.25),1)
        text=json.dumps(ctx)
        for forbidden in ['truth','threshold_probe','heldout','scores','incoming_velocity','event_count','candidate_seed','previous_campaign']:
            self.assertNotIn(forbidden,text)
        self.assertEqual(len(ctx['candidates']),10)

if __name__=='__main__':unittest.main()
