"""Physical design, evaluator isolation, fair numerical policies and hard budgets."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

spec=importlib.util.spec_from_file_location('physical_sweep',Path(__file__).resolve().parents[1]/'experiments/hybrid_physical_sweep.py')
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)


class PhysicalSweepTests(unittest.TestCase):
    def test_six_settings_two_replicates_and_disjoint_new_streams(self):
        rows=pilot.design()
        self.assertEqual(len(rows),12)
        settings={(row['floor_m'],row['restitution']) for row in rows}
        self.assertEqual(settings,{(z,e) for z in [.03,.12,.22] for e in [.65,.90]})
        for setting in settings:
            self.assertEqual({r['noise_replicate'] for r in rows if (r['floor_m'],r['restitution'])==setting},{1,2})
        seeds=[row[key] for row in rows for key in ['baseline_noise_seed','random_policy_seed']]
        seeds += [value for row in rows for value in row['candidate_noise_seeds'].values()]
        self.assertEqual(len(seeds),len(set(seeds)))
        self.assertTrue(set(seeds).isdisjoint(set(range(1001,1007))|set(range(50000,56000))))
        pilot.validate_horizons(pilot.common())

    def test_simulator_truth_is_honored_and_absent_from_context(self):
        p=pilot.common();q=pilot.common();q['simulator'].update({'floor_m':.22,'restitution':.90})
        a=pilot.core.generate(p,0.,100001);b=pilot.core.generate(q,0.,100001)
        self.assertFalse(np.array_equal(a.heights,b.heights))
        pa=pilot.core.predict(.12,.8,1.,0.,a.times)[0];pb=pilot.core.predict(.22,.9,1.,0.,b.times)[0]
        np.testing.assert_allclose(b.heights-a.heights,pb-pa,atol=1e-14)
        model=pilot.core.BallPINN([a],p['inference'],p['pinn'],seed=0)
        ep=pilot.core.evaluate(model,[a],p);eq=pilot.core.evaluate(model,[b],q)
        self.assertAlmostEqual(ep['floor_absolute_error_m'],.07)
        self.assertAlmostEqual(eq['floor_absolute_error_m'],.17)
        self.assertAlmostEqual(eq['restitution_absolute_error'],.25)
        self.assertNotEqual(ep[pilot.PRIMARY],eq[pilot.PRIMARY])
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'campaign'
            with patch.object(pilot.core,'run_job',return_value={'status':'worker_failed'}):
                pilot.core.prepare(target,{'simulator':q['simulator']})
            actual=pilot.core.deserialize(pilot.core.read(target/'baseline/observations.json')[0])
            expected=pilot.core.generate(q,0.,17)
            np.testing.assert_array_equal(actual.heights,expected.heights)
        baseline={'parameters':{'floor_m':.1,'restitution':.75},'learned_event_times_s':[.43],
                  'diagnostics':{},'evaluation':{'hidden_reserved':42},'simulator':q['simulator']}
        context=pilot.core.selector_context(baseline,q)
        self.assertNotIn('simulator',context);self.assertNotIn('evaluation',context)
        self.assertNotIn('trial_spec',context)
        self.assertEqual(context,pilot.core.selector_context(baseline,p))

    def test_explicit_numeric_random_fixed_identities_no_fabricated_llm(self):
        context={'objective':'test','candidates':[{'id':'upward','predicted_information_logdet':5.},
                    {'id':'downward','predicted_information_logdet':1.},{'id':'neutral','predicted_information_logdet':2.}]}
        choices=pilot.choose(context,300000)
        self.assertEqual(set(choices),set(pilot.POLICIES));self.assertNotIn('llm',choices)
        self.assertEqual(choices['adaptive_numerical'],'upward')
        self.assertEqual(choices['fixed_downward'],'downward')
        self.assertEqual(choices['uniform_random'],pilot.core.uniform_random_candidate(300000))

    def test_fifty_second_allowance_and_four_jobs_no_borrowing(self):
        self.assertEqual(pilot.common()['budgets']['total_compute_s']*len(pilot.design()),600)
        self.assertEqual(pilot.common()['budgets']['maximum_training_jobs']*len(pilot.design()),48)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'baseline').mkdir();(root/'adaptive_numerical').mkdir()
            pilot.core.write(root/'protocol.json',{'budgets':pilot.common()['budgets']})
            pilot.core.write(root/'baseline/process.json',{'elapsed_s':49.6})
            with patch.object(pilot.core,'subprocess_job') as launch:
                result=pilot.core.run_job(root,'adaptive_numerical');launch.assert_not_called()
            self.assertEqual(result['status'],'training_budget_unavailable')

    def rows(self):
        return [{**spec,'complete':True,'fits':{name:{'candidate_id':'neutral' if name=='baseline' else 'downward',
                  'evaluation':{pilot.PRIMARY:.002 if name=='baseline' else .001}} for name in ['baseline',*pilot.POLICIES]}}
                for spec in pilot.design()]

    def test_matches_mean_no_adaptation_evidence_and_setting_clusters(self):
        summary=pilot.summarize(self.rows())
        self.assertEqual(summary['adaptive_matches_fixed_downward'],12)
        self.assertEqual(len(summary['setting_means']),6)
        self.assertEqual(summary['paired_differences']['fixed_downward'],[0.]*12)
        self.assertIn('no evidence adaptation helps',summary['conclusion'])

    def test_incomplete_cohort_suppresses_aggregate_ranking(self):
        rows=self.rows();rows[3]['complete']=False
        for cohort in [rows,self.rows()[:-1]]:
            summary=pilot.summarize(cohort)
            self.assertIsNone(summary['means']);self.assertIsNone(summary['paired_differences']);self.assertIsNone(summary['setting_means'])

    def test_refuses_existing_campaign(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'preserved').write_text('keep')
            with self.assertRaises(FileExistsError):pilot.run(root)
            self.assertEqual((root/'preserved').read_text(),'keep')

    def test_backward_defaults_and_fair_matching_initialization(self):
        p=pilot.common();obs=pilot.core.generate(p,0.,17)
        model=pilot.core.BallPINN([obs],p['inference'],p['pinn'],seed=0)
        baseline={'model_state_dict':model.state_dict()}
        updates=[pilot.core.warm_start(baseline,[obs,pilot.core.generate(p,-4.,200101)],p) for _ in range(3)]
        states=[m.state_dict() for m in updates]
        for key in states[0]:
            np.testing.assert_array_equal(states[0][key].detach().numpy(),states[1][key].detach().numpy())
            np.testing.assert_array_equal(states[0][key].detach().numpy(),states[2][key].detach().numpy())
        self.assertEqual(p['inference']['floor_bounds_m'],[0.,.25])
        self.assertEqual(p['pinn']['adam_steps'],1500)


if __name__=='__main__':unittest.main()
