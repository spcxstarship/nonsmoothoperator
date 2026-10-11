"""Frozen selector boundaries, warm-start equality and timeout preservation."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch

spec=importlib.util.spec_from_file_location('hybrid_round',Path(__file__).resolve().parents[1]/'experiments/hybrid_selection_round.py')
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)


def protocol():
    return {'simulator':{'floor_m':.12,'restitution':.8},
            'inference':{'floor_bounds_m':[0.,.25],'restitution_bounds':[.6,.95],'initial_floor_m':.05,'initial_restitution':.65},
            'pinn':{'width':16,'collocation_points_per_segment':48,'adam_steps':1500,'adam_learning_rate':.003,
                    'lbfgs_steps':1000,'physics_weight':100.,'guard_weight':100.}}


class HybridRoundTests(unittest.TestCase):
    def test_context_whitelist_excludes_truth_and_reserved_scores(self):
        p=protocol()
        baseline={'parameters':{'floor_m':.1,'restitution':.78},'learned_event_times_s':[.43],
                  'diagnostics':{'flow':.01},'evaluation':{'secret_reserved_error':999},'simulator':{'secret_truth':123}}
        context=pilot.selector_context(baseline,p)
        text=json.dumps(context)
        for forbidden in ['secret_reserved','secret_truth','evaluation','simulator']:
            self.assertNotIn(forbidden,text)
        self.assertEqual({c['id'] for c in context['candidates']},{'downward','neutral','upward'})
        self.assertTrue(all(np.isfinite(c['predicted_information_logdet']) for c in context['candidates']))

    def test_identical_baseline_and_new_modules_across_candidates(self):
        p=protocol();obs=pilot.generate(p,0.,17)
        baseline=pilot.BallPINN([obs],p['inference'],p['pinn'],seed=0)
        with torch.no_grad():
            baseline.raw_floor.add_(.123);baseline.raw_times.add_(.17)
        checkpoint={'model_state_dict':baseline.state_dict()}
        models=[pilot.warm_start(checkpoint,[obs,pilot.generate(p,a,118)],p) for a in [-4.,4.]]
        old=baseline.state_dict()
        for model in models:
            state=model.state_dict()
            for key,value in old.items():
                if key in ['raw_times','tau_bounds']:continue
                self.assertTrue(torch.equal(state[key],value),key)
            self.assertTrue(torch.equal(state['raw_times'][:1],old['raw_times']))
        for key in models[0].state_dict():
            if key.startswith(('pre.1.','post.1.')):
                self.assertTrue(torch.equal(models[0].state_dict()[key],models[1].state_dict()[key]),key)

    def test_candidate_noise_is_repeatable_but_neutral_repeat_is_fresh(self):
        p=protocol()
        self.assertTrue(np.array_equal(pilot.generate(p,0.,219).heights,pilot.generate(p,0.,219).heights))
        self.assertFalse(np.array_equal(pilot.generate(p,0.,17).heights,pilot.generate(p,0.,219).heights))

    def test_unknown_choice_rejected_before_output_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'sources').mkdir()
            pilot.write(root/'protocol.json',{'source_sha256':{}})
            pilot.write(root/'prepare-manifest.json',{'artifacts_sha256':{}})
            target=root/'selector/ball/trial/round';target.mkdir(parents=True)
            pilot.write(target/'context.json',{'objective':'test','candidates':[{'id':'neutral'}]})
            (target/'instructions.txt').write_text('choose one')
            pilot.write(target/'manifest.json',{'status':'open','files_sha256':{'context.json':pilot.sha(target/'context.json'),'instructions.txt':pilot.sha(target/'instructions.txt')}})
            decision=root/'bad.json';pilot.write(decision,{'candidate_id':'unknown','hypothesis':'test','rationale':'test'})
            with self.assertRaisesRegex(ValueError,'unknown candidate'):
                pilot.finish(root,decision,root/'absent-metadata.json')
            self.assertFalse((root/'finish-lock.json').exists())

    def test_integrity_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);pilot.write(root/'input.json',{'one':1})
            pilot.write(root/'manifest.json',{'artifacts_sha256':{'input.json':pilot.sha(root/'input.json')}})
            pilot.check_manifest(root,'manifest.json')
            with self.assertRaises(FileExistsError):pilot.write(root/'input.json',{'two':2})
            (root/'input.json').write_text('tampered')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):pilot.check_manifest(root,'manifest.json')

    def test_prepared_selector_rejects_added_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            target=root/'selector/ball/trial/round';target.mkdir(parents=True)
            pilot.write(target/'context.json',{'objective':'test','candidates':[{'id':'neutral'}]})
            (target/'instructions.txt').write_text('choose one')
            pilot.write(target/'manifest.json',{'status':'open','files_sha256':{'context.json':pilot.sha(target/'context.json'),'instructions.txt':pilot.sha(target/'instructions.txt')}})
            pilot.write(root/'prepare-manifest.json',{'artifacts_sha256':{'selector/ball/trial/round/context.json':pilot.sha(target/'context.json')}})
            pilot.verify_prepared(root)
            (target/'truth.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'inventory mismatch'):
                pilot.verify_prepared(root)

    def test_timeout_retains_partial_stdout_and_stack(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            result=pilot.subprocess_job(root,[sys.executable,'-c','import time; print("partial",flush=True); time.sleep(5)'],.1)
            self.assertEqual(result['error_type'],'TimeoutExpired')
            self.assertIn('partial',(root/'stdout.log').read_text())
            self.assertIn('Traceback',(root/'controller-stack.txt').read_text())


if __name__=='__main__':unittest.main()
