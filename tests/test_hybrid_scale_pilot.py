"""Frozen menu design, predictive information and fail-closed acquisition checks."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
spec=importlib.util.spec_from_file_location('scale_pilot',Path(__file__).resolve().parents[1]/'experiments/hybrid_scale_pilot.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)

class ScaleTests(unittest.TestCase):
    def protocol(self):
        value=p.physical.common();value['data']['times_s']=np.linspace(.56,.7,12).tolist();return value

    def test_nested_design_and_separate_streams(self):
        rows=p.design();self.assertEqual(len(rows),4);self.assertTrue(set(p.MENUS[3])<set(p.MENUS[9]))
        seeds=[]
        for r in rows:
            seeds.append(r['baseline_noise_seed'])
            for field in ['candidate_noise_seeds','random_policy_seeds','design_mc_seeds']:
                for value in r[field].values():seeds.extend(value.values() if isinstance(value,dict) else [value])
        self.assertEqual(len(seeds),len(set(seeds)));self.assertEqual(len(p.POLICIES)*2*4+4,44)
        for u in p.MENUS[9]:p.core.validate_observations([p.core.Observations(1.,u,np.linspace(.56,.7,12),np.zeros(12))],[0,.25],[.6,.95],9.81)

    def test_posterior_likelihood_and_known_model(self):
        protocol=self.protocol()
        params=np.array([[.12,.8]])
        protocol['simulator'].update({'floor_m':.12,'restitution':.8})
        obs=p.core.generate(protocol,0.,91)
        expected,_=p.core.predict(.12,.8,1.,0.,obs.times)
        np.testing.assert_allclose(p.predictions(params,0.,obs.times)[0],expected,atol=1e-14)
        post=p.posterior([p.core.serialize(obs)],protocol['inference'],33)
        self.assertAlmostEqual(float(post['weights'].sum()),1.)
        self.assertGreater(post['summary']['retained_mass'],1-2e-12)
        self.assertLess(abs(post['summary']['parameter_summary']['floor_m']['mean']-.12),.04)
        with self.assertRaises(ValueError):p.posterior([{**p.core.serialize(obs),'initial_height':1.1}],protocol['inference'],17)

    def test_mi_degenerate_and_nested_common_random(self):
        times=np.linspace(.56,.7,12)
        post={'parameters':np.array([[.1,.8]]),'weights':np.array([1.])}
        score=p.information_scores(post,times,p.MENUS[3],19,32)
        self.assertTrue(all(abs(s['mean'])<1e-8 for s in score['scores'].values()))
        post={'parameters':np.array([[.07,.7],[.18,.86]]),'weights':np.array([.4,.6])}
        three=p.information_scores(post,times,p.MENUS[3],31,32);nine=p.information_scores(post,times,p.MENUS[9],31,32)
        for key in three['scores']:self.assertEqual(three['scores'][key],nine['scores'][key])
        chosen=nine['selected_candidate_id'];runner=nine['selected_gap']['runner_up'];diff=np.array(nine['contributions'][chosen])-np.array(nine['contributions'][runner])
        self.assertAlmostEqual(nine['selected_gap']['paired_mc_standard_error'],float(diff.std(ddof=1)/np.sqrt(32)))

    def test_context_whitelist_and_stale_diagnostics(self):
        with tempfile.TemporaryDirectory() as d:
            case=Path(d);(case/'baseline').mkdir();protocol=self.protocol()
            p.core.write(case/'protocol.json',protocol);p.core.write(case/'baseline/result.json',{'diagnostics':{'loss':1},'evaluation':{'secret':2}})
            history=[p.core.serialize(p.core.generate(protocol,0.,91))]
            post=p.posterior(history,protocol['inference'],17)
            fit={k:0 for k in ['floor_m','restitution','observation_rmse_m','optimizer_success','jacobian_singular_values']};fit['evaluation']={'secret':3}
            context=p.context_for(case,case,9,2,history,post,fit)
            import json
            serialized=json.dumps(context)
            for forbidden in ['secret','simulator','candidate_noise','scores','selected_candidate_id','evaluation']:self.assertNotIn(forbidden,serialized)
            self.assertIn('stale',serialized);self.assertEqual(len(context['candidates']),9)

    def test_phase_guards_do_not_mutate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p.core.write(root/'round-1-lock.json',{})
            before=list(root.iterdir())
            with self.assertRaisesRegex(ValueError,'already started'):p.acquire(root,1,root/'missing')
            self.assertEqual(before,list(root.iterdir()))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p.core.write(root/'round-2-contexts.json',{})
            with self.assertRaisesRegex(ValueError,'round1 must complete'):p.acquire(root,2,root/'missing')

    def test_nonborrowable_budget_and_source_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p.core.write(root/'protocol.json',{'budgets':{},'source_sha256':{'x.py':'wrong'}});(root/'sources').mkdir();(root/'sources/x.py').write_text('x')
            with self.assertRaisesRegex(ValueError,'source mismatch'):p.source_check(root)
            (root/'baseline').mkdir();p.core.write(root/'baseline/process.json',{'elapsed_s':149.6})
            job=root/'routes/3/fixed_downward';job.mkdir(parents=True)
            with patch.object(p.core,'subprocess_job') as run:
                self.assertEqual(p.launch(root,job,['unused'])['status'],'training_budget_unavailable');run.assert_not_called()

    def test_fresh_initialization_is_equal_across_action_histories(self):
        protocol=self.protocol();fingerprints=[]
        for actions in [[0,-4,4],[0,4,4],[0,-1,2]]:
            history=[p.core.generate(protocol,u,92+i) for i,u in enumerate(actions)]
            model=p.core.BallPINN(history,protocol['inference'],protocol['pinn'],seed=0)
            state=model.state_dict();fingerprints.append(p.core.state_digest(state,[key for key in state if key!='tau_bounds']))
        self.assertEqual(len(set(fingerprints)),1)

    def test_incomplete_aggregation_and_interaction(self):
        self.assertEqual(p.summarize([])['status'],'unavailable_incomplete')
        rows=[]
        for i in range(4):
            menus={str(size):{policy:{'evaluation':{p.PRIMARY:.001*(j+1)+(size==9)*.002},'classical':{'evaluation':{p.PRIMARY:.001}}} for j,policy in enumerate(p.POLICIES)} for size in p.MENUS}
            rows.append({'complete':True,'menus':menus})
        summary=p.summarize(rows);self.assertEqual(summary['status'],'complete_descriptive_only')
        self.assertAlmostEqual(summary['interactions']['fixed_downward']['mean'],0.)
        rows[0]['complete']=False;self.assertIsNone(p.summarize(rows)['means'])

if __name__=='__main__':unittest.main()
