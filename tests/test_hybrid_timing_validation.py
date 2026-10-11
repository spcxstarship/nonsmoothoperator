"""Actual timings, independent controls, reserved metrics and bounded final design."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('timing_validation',Path(__file__).resolve().parents[1]/'experiments/hybrid_timing_validation.py')
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)


class ValidationTests(unittest.TestCase):
    def test_frozen_design_and_disjoint_streams(self):
        rows=pilot.design();main=[r for r in rows if r['kind']=='main'];controls=[r for r in rows if r['kind']=='control']
        self.assertEqual(len(main),8);self.assertEqual(len(controls),2)
        self.assertEqual({(r['floor_m'],r['restitution']) for r in main},{(z,e) for z in [.04,.21] for e in [.68,.88]})
        seeds=[r[k] for r in rows for k in ['baseline_noise_seed','random_policy_seed']]+[s for r in rows for s in r['candidate_noise_seeds'].values()]
        self.assertEqual(len(seeds),len(set(seeds)))
        self.assertEqual(sum(pilot.overrides(r)['budgets']['maximum_training_jobs'] for r in rows),60)
        self.assertEqual(sum(pilot.overrides(r)['budgets']['total_compute_s'] for r in rows),600)

    def baseline(self):
        return {'parameters':{'floor_m':.10,'restitution':.78},'learned_event_times_s':[.43],
                'diagnostics':{},'evaluation':{'forbidden_score':123}}

    def test_actual_schedule_in_data_context_and_information(self):
        p=pilot.physical.common();late=np.linspace(.56,.7,12).tolist();p['data']['times_s']=late
        observations=pilot.core.generate(p,0.,700001);np.testing.assert_array_equal(observations.times,late)
        context=pilot.core.selector_context(self.baseline(),p)
        self.assertEqual(context['history'][0]['times_s'],late)
        self.assertTrue(all(c['times_s']==late for c in context['candidates']))
        self.assertNotIn('evaluation',context);self.assertNotIn('simulator',context)
        self.assertEqual(context['diagnostics']['initial_information_rank'],2)
        original=pilot.core.selector_context(self.baseline(),pilot.physical.common())
        self.assertNotEqual([c['predicted_information_logdet'] for c in context['candidates']],
                            [c['predicted_information_logdet'] for c in original['candidates']])

    def test_early_control_unregularized_rank_is_one(self):
        p=pilot.physical.common();p['data']['times_s']=np.r_[np.linspace(.05,.35,11),.7].tolist()
        context=pilot.core.selector_context(self.baseline(),p)
        self.assertEqual(context['diagnostics']['initial_information_rank'],1)
        self.assertIsNone(context['diagnostics']['initial_information_condition_number'])

    def test_transfer_helper_is_parameter_only_and_true_parameters_zero_error(self):
        p=pilot.physical.common()
        result=pilot.core.parameter_transfer_metrics({'floor_m':.12,'restitution':.8},p)
        self.assertEqual(result[pilot.PRIMARY],0.)
        self.assertEqual(result['parameter_transfer_impact_time_mae_s'],0.)
        self.assertEqual(result['floor_absolute_error_m'],0.)
        self.assertIn('NOT direct neural generalization',result['primary_metric_definition'])

    def rows(self):
        return [{**r,'complete':True,'fits':{name:{'candidate_id':'neutral' if name=='baseline' else 'downward',
                     'evaluation':{pilot.PRIMARY:.002 if r['kind']=='main' else 100.},
                     'classical':{'evaluation':{pilot.PRIMARY:.001 if r['kind']=='main' else 200.}}}
                     for name in ['baseline',*pilot.POLICIES]}} for r in pilot.design()]

    def test_controls_excluded_and_both_fixed_directions_compared(self):
        summary=pilot.summarize(self.rows())
        self.assertAlmostEqual(summary['means']['adaptive_numerical'],.002)
        self.assertAlmostEqual(summary['classical_means']['adaptive_numerical'],.001)
        self.assertEqual(len(summary['setting_means']),4)
        self.assertEqual(set(summary['paired_differences']['fresh_agent']),{'fixed_downward','fixed_upward','uniform_random'})
        self.assertEqual(len(summary['paired_differences']['fresh_agent']['fixed_upward']),8)

    def test_any_failed_control_suppresses_aggregate(self):
        rows=self.rows();rows[-1]['complete']=False
        self.assertIsNone(pilot.summarize(rows)['means'])

    def test_existing_campaign_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):pilot.prepare(Path(directory))


if __name__=='__main__':unittest.main()
