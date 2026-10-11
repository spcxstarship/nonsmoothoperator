"""Contact mechanics, class-focused information and position-only contexts."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
spec=importlib.util.spec_from_file_location('stick_slip',Path(__file__).resolve().parents[1]/'experiments/stick_slip_pilot.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)

class StickSlipTests(unittest.TestCase):
    def test_balanced_bank_and_initial_ambiguity(self):
        particles=p.bank();self.assertEqual(len(particles),32)
        pred={'baseline':np.array([p.simulate(q,p.BASELINE)[0] for q in particles])}
        np.testing.assert_array_equal(pred['baseline'],0.)
        w,prob=p.posterior([{'candidate_id':'baseline','positions':[.001]*16}],pred,particles)
        np.testing.assert_allclose(w,1/32);np.testing.assert_allclose(list(prob.values()),.25)

    def test_slip_equation_and_impact_energy(self):
        q=p.bank()[0];t=.23;x=.03;v=.2;f=.7;eps=1e-6
        a,b=p.slip(t,x,v,f,q,1);aa,bb=p.slip(t+eps,x,v,f,q,1)
        self.assertAlmostEqual(float((aa-a)/eps),float(b),places=5)
        self.assertAlmostEqual(float((bb-b)/eps),float(f-q['spring']*a-q['damping']*b-q['friction']),places=5)
        y,events=p.simulate(q,p.PROFILES['impact_reversal']);self.assertLessEqual(max(abs(y)),.25)
        self.assertEqual(sum(e['kind']=='left' for e in events),2);self.assertEqual(sum(e['kind']=='right' for e in events),2)
        for e in events:
            if e['kind'] in ['left','right']:self.assertLess(e['restitution']**2,1.)

    def test_static_breakaway_differs_only_at_release(self):
        a={**p.bank()[0],'class':'coulomb_constant'};b={**a,'class':'static_constant'}
        ya,ea=p.simulate(a,[.20]*4);yb,eb=p.simulate(b,[.20]*4)
        self.assertGreater(max(abs(ya)),0.);np.testing.assert_array_equal(yb,0.)
        np.testing.assert_allclose(p.slip(.3,.03,.2,.7,a,1),p.slip(.3,.03,.2,.7,b,1))

    def test_wall_rest_and_contact_refinement(self):
        q={**p.bank()[0],'restitution':.2}
        y,e=p.simulate(q,[20.,20.,-20.,-20.]);self.assertTrue(any(x['kind']=='wall_rest' for x in e));self.assertLessEqual(max(abs(y)),.25)
        y2,e2=p.simulate(q,[20.,20.,-20.,-20.],collapse_speed=2.5e-7,root_tolerance=2.5e-13)
        np.testing.assert_allclose(y,y2,atol=1e-5)
        for particle in p.bank():
            for profile in p.PROFILES.values():
                a,_=p.simulate(particle,profile);b,_=p.simulate(particle,profile,collapse_speed=2.5e-7,root_tolerance=2.5e-13)
                np.testing.assert_allclose(a,b,atol=1e-5)

    def test_class_mi_not_nuisance_mi(self):
        particles=[{'class':'coulomb_constant'},{'class':'coulomb_constant'}];w=np.array([.5,.5]);pred={name:np.array([[0.]*16,[.1]*16]) for name in p.PROFILES}
        result=p.class_information(w,pred,particles,91,32)
        self.assertTrue(all(abs(x['mean'])<1e-10 for x in result['scores'].values()))
        particles[1]['class']='static_constant';result2=p.class_information(w,pred,particles,91,32)
        self.assertTrue(all(x['mean']>.5 for x in result2['scores'].values()))
        self.assertEqual(result2,p.class_information(w,pred,particles,91,32))

    def test_position_only_context_and_exclusive_outputs(self):
        c=p.context([{'candidate_id':'baseline','positions':[0.]*16}],dict.fromkeys(p.CLASSES,.25),1)
        import json
        text=json.dumps(c)
        for forbidden in ['truth','scores','incoming_velocity','event_count','mc_seed','candidate_seed']:self.assertNotIn(forbidden,text)
        self.assertEqual(len(c['candidates']),10);self.assertEqual(len(c['diagnostics']['particle_posterior']),32);self.assertAlmostEqual(sum(x['probability'] for x in c['diagnostics']['particle_posterior']),1.)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p.write(root/'x',{'a':1})
            with self.assertRaises(FileExistsError):p.write(root/'x',{'a':2})
            p.write(root/'round-1-lock.json',{})
            with self.assertRaisesRegex(ValueError,'alreadystarted'):p.acquire(root,1,root/'absent')
            p.write(root/'m',{'artifacts':{'x':p.sha(root/'x')}});p.check_hashes(root,'m');(root/'x').write_text('changed')
            with self.assertRaises(ValueError):p.check_hashes(root,'m')

if __name__=='__main__':unittest.main()
