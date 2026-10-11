"""Actual timing, normalized sensitivities, prior support and saved integrity."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('timing_audit',Path(__file__).resolve().parents[1]/'experiments/hybrid_timing_audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class TimingAuditTests(unittest.TestCase):
    def test_schedule_generation_and_impact_prior(self):
        declared=audit.schedules();self.assertEqual(len(declared),25)
        self.assertEqual(len({row['id'] for row in declared}),25)
        for row in declared:audit.validate_schedule(row['times_s'])
        self.assertIn('early-parent-0.35',{row['id'] for row in declared})
        with self.assertRaises(ValueError):audit.validate_schedule(np.linspace(.05,.8,12))

    def test_information_uses_exact_schedule_not_uniform_placeholder(self):
        p=np.array([[.10,.78]])
        a=audit.scores(p,audit.ORIGINAL)
        b=audit.scores(p,np.r_[np.linspace(.05,.35,11),.7])
        self.assertFalse(np.allclose(a,b))
        uniform=np.linspace(.59,.7,12)
        self.assertFalse(np.allclose(a,audit.scores(p,uniform)))

    def test_normalized_derivatives_agree_away_from_guard_and_refine(self):
        p=np.array([[.113,.771],[.023,.923],[.224,.653]])
        for _,u in audit.CANDIDATES:
            times=np.array([.05,.10,.18,.24,.29,.31,.47,.61,.64,.66,.68,.7])
            approximate=audit.jacobian(p,u,times,1e-5)
            smaller=audit.jacobian(p,u,times,5e-6)
            exact=audit.analytic_jacobian(p,u,times)
            np.testing.assert_allclose(approximate,exact,rtol=1e-7,atol=1e-7)
            np.testing.assert_allclose(smaller,exact,rtol=1e-7,atol=1e-7)

    def test_grid_refinement_contains_primary_grid(self):
        primary=audit.grid(17);fine=audit.grid(33)
        self.assertEqual(len(primary),289);self.assertEqual(len(fine),1089)
        self.assertTrue(all(np.any(np.all(np.isclose(fine,p,atol=1e-15),axis=1)) for p in primary))

    def test_saved_manifest_detects_modified_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'sources').mkdir()
            source=root/'sources/hybrid_timing_audit.py';source.write_text('synthetic test source')
            audit.write(root/'protocol.json',{'frozen_source_sha256':audit.sha(source)})
            audit.trace(root,'hypothesis','Synthetic integrity test',[root/'protocol.json',source])
            audit.write(root/'manifest.json',{'elapsed_s':0.,'artifacts_sha256':audit.inventory(root)})
            audit.verify(root)
            (root/'protocol.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):audit.verify(root)

    def test_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaises(FileExistsError):audit.run(root)


if __name__=='__main__':unittest.main()
