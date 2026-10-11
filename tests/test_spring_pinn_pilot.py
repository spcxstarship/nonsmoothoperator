"""Trace integrity and subprocess failure retention, without training loops."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('spring_pilot', Path(__file__).resolve().parents[1]/'experiments/spring_pinn_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class PilotTraceTests(unittest.TestCase):
    def campaign(self, root):
        pilot.write(root/'source.py', {'synthetic':'source'})
        pilot.write(root/'protocol.json', {'frozen_sha256':{'source.py':pilot.sha(root/'source.py')}})
        pilot.trace(root,'hypothesis','Test integrity',[root/'source.py',root/'protocol.json'])
        pilot.write(root/'manifest.json', {'artifacts_sha256':{p.name:pilot.sha(p) for p in root.iterdir()}})

    def test_valid_then_modified_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.campaign(root)
            pilot.validate(root)
            (root/'source.py').write_text('tampered')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                pilot.validate(root)

    def test_bad_trace_link_rejected_even_with_rehashed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.campaign(root)
            row = json.loads((root/'thoughts.jsonl').read_text())
            row['artifacts_sha256']['source.py']='0'*64
            (root/'thoughts.jsonl').write_text(json.dumps(row)+'\n')
            manifest = json.loads((root/'manifest.json').read_text())
            manifest['artifacts_sha256']['thoughts.jsonl']=pilot.sha(root/'thoughts.jsonl')
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'trace link mismatch'):
                pilot.validate(root)

    def test_frozen_source_hash_survives_manifest_rehash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.campaign(root)
            (root/'source.py').write_text('modified source')
            # Change both manifest and trace to isolate the protocol freeze check.
            row = json.loads((root/'thoughts.jsonl').read_text())
            row['artifacts_sha256']['source.py']=pilot.sha(root/'source.py')
            (root/'thoughts.jsonl').write_text(json.dumps(row)+'\n')
            manifest = json.loads((root/'manifest.json').read_text())
            manifest['artifacts_sha256']['source.py']=pilot.sha(root/'source.py')
            manifest['artifacts_sha256']['thoughts.jsonl']=pilot.sha(root/'thoughts.jsonl')
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'input/source hash mismatch'):
                pilot.validate(root)

    def test_unlisted_nested_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.campaign(root)
            (root/'job').mkdir()
            (root/'job'/'manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'inventory mismatch'):
                pilot.validate(root)

    def test_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pilot.write(root/'result.json',{'one':1})
            with self.assertRaises(FileExistsError):
                pilot.write(root/'result.json',{'two':2})
            with self.assertRaises(FileExistsError):
                pilot.campaign(root,root/'absent.ipynb')
            self.assertEqual(json.loads((root/'result.json').read_text()),{'one':1})

    def test_failure_and_timeout_keep_logs_and_stacks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label,code,timeout in [('fail','print("partial",flush=True); raise RuntimeError("synthetic failure")',2),
                                       ('timeout','import time; print("partial",flush=True); time.sleep(5)',.1)]:
                job = root/label; job.mkdir()
                result = pilot.run_job(root,job,[sys.executable,'-c',code],timeout)
                self.assertEqual(result['status'],'worker_failed')
                self.assertIn('partial',(job/'stdout.log').read_text())
                self.assertIn('Traceback',(job/'controller-stack.txt').read_text())
            self.assertIn('RuntimeError',(root/'fail'/'stderr.log').read_text())


if __name__=='__main__':
    unittest.main()
