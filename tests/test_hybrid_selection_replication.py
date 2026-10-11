"""Predeclared seeds, random policy, trial budgets, and complete-cohort summaries."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

spec=importlib.util.spec_from_file_location('hybrid_replication',Path(__file__).resolve().parents[1]/'experiments/hybrid_selection_replication.py')
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)


class ReplicationTests(unittest.TestCase):
    def test_noise_streams_never_overlap_and_are_repeatable(self):
        schedule=pilot.seed_schedule()
        self.assertEqual(schedule,pilot.seed_schedule())
        baselines={item['baseline_noise_seed'] for item in schedule}
        new=[seed for item in schedule for seed in item['candidate_noise_seeds'].values()]
        self.assertEqual(len(baselines),6);self.assertEqual(len(set(new)),18)
        self.assertTrue(baselines.isdisjoint(new))
        self.assertEqual([item['baseline_noise_seed'] for item in schedule],list(range(1001,1007)))

    def test_random_policy_is_exact_unbalanced_uniform_rng_rule(self):
        for item in pilot.seed_schedule():
            seed=item['random_policy_seed']
            expected=pilot.round_impl.CANDIDATES[int(np.random.default_rng(seed).integers(0,3))][0]
            self.assertEqual(pilot.round_impl.uniform_random_candidate(seed),expected)
            self.assertEqual(pilot.round_impl.uniform_random_candidate(seed),pilot.round_impl.uniform_random_candidate(seed))

    def test_exhausted_trial_does_not_borrow_or_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'baseline').mkdir();(root/'random').mkdir()
            pilot.round_impl.write(root/'protocol.json',{'budgets':pilot.override(pilot.seed_schedule()[0])['budgets']})
            pilot.round_impl.write(root/'baseline/process.json',{'elapsed_s':100.})
            with patch.object(pilot.round_impl,'subprocess_job') as launch:
                result=pilot.round_impl.run_job(root,'random')
                launch.assert_not_called()
            self.assertEqual(result['status'],'training_budget_unavailable')
            self.assertFalse((root/'random/invocation.json').exists())

    def test_four_job_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            pilot.round_impl.write(root/'protocol.json',{'budgets':pilot.override(pilot.seed_schedule()[0])['budgets']})
            for name in ['baseline','llm','numerical','random']:
                (root/name).mkdir();pilot.round_impl.write(root/name/'invocation.json',{})
            with self.assertRaisesRegex(ValueError,'maximum training jobs'):
                pilot.round_impl.run_job(root,'extra')

    def rows(self):
        return [{'complete':True,'policies':{policy:{'evaluation':{pilot.PRIMARY:value}} for policy,value in [('llm',.02),('numerical',.01),('random',.03)]}} for _ in range(6)]

    def test_complete_aggregate_reports_paired_descriptive_differences(self):
        summary=pilot.aggregate(self.rows())
        self.assertEqual(summary['status'],'complete_descriptive_only')
        self.assertAlmostEqual(summary['paired_vs_random']['llm']['mean'],-.01)
        self.assertAlmostEqual(summary['paired_vs_random']['numerical']['mean'],-.02)
        self.assertEqual(len(summary['paired_vs_random']['llm']['per_trial']),6)

    def test_missing_or_failed_trial_suppresses_all_aggregate_ranking(self):
        rows=self.rows();rows[2]['complete']=False
        for cohort in [rows,self.rows()[:-1]]:
            summary=pilot.aggregate(cohort)
            self.assertEqual(summary['status'],'unavailable_incomplete')
            self.assertIsNone(summary['means']);self.assertIsNone(summary['paired_vs_random'])

    def test_existing_campaign_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'preserved.txt').write_text('keep')
            with self.assertRaises(FileExistsError):pilot.prepare(root)
            self.assertEqual((root/'preserved.txt').read_text(),'keep')


if __name__=='__main__':unittest.main()
