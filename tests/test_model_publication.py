"""A corrupt candidate must never replace the active model."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from Final.tools.promote_candidate import FILES, promote

ROOT = Path(__file__).resolve().parents[1]


class ModelPublicationTests(unittest.TestCase):
    def test_corrupt_candidate_is_rejected_before_any_publication(self):
        active = ROOT/'backend/artifacts/mushroom_rf300.npz'
        before = hashlib.sha256(active.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory)
            for name in FILES:
                dest = candidate/name; dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/name, dest)
            (candidate/'backend/artifacts/mushroom_rf300.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                promote(candidate)
        self.assertEqual(hashlib.sha256(active.read_bytes()).hexdigest(), before)

    def test_failed_cv_gate_is_rejected_before_publication(self):
        active = ROOT/'backend/artifacts/mushroom_rf300.json'
        before = active.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory)
            for name in FILES:
                dest = candidate/name; dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/name, dest)
            path = candidate/'Final/Mushroom/artifacts/training_run.json'
            run = json.loads(path.read_text()); run['cv']['cv_macro_f1_mean'] = 0
            path.write_text(json.dumps(run))
            with self.assertRaisesRegex(ValueError, 'quality gates'):
                promote(candidate)
        self.assertEqual(active.read_bytes(), before)
