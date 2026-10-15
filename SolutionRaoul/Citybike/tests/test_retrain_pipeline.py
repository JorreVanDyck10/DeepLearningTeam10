"""Een checkout mag een extern referentiemodel niet zonder receiptcontrole laden."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import retrain_pipeline
from citibike.config import sha256_file,fingerprint


class ReferenceIntegrityTests(unittest.TestCase):
    def fixture(self,root):
        (root/'models').mkdir();(root/'reports').mkdir()
        (root/'models/citibike_model.joblib').write_bytes(b'unchanged fixture model')
        selection={'fixed':'RF320'}
        for name,value in [('model_selection.json',selection),('final_evaluation.json',{}),('evaluation_lock.json',{})]:
            (root/'reports'/name).write_text(json.dumps(value),encoding='utf-8')
        receipt={'sha256':sha256_file(root/'models/citibike_model.joblib'),'selection_fingerprint':fingerprint(selection)}
        (root/'reports/model_artifact.json').write_text(json.dumps(receipt),encoding='utf-8')

    def test_foreign_reference_checksum_failure_stops_before_loading_or_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            (root/'models/citibike_model.joblib').write_bytes(b'changed model')
            with patch.object(retrain_pipeline,'load_model') as loader:
                with self.assertRaisesRegex(ValueError,'Referentiemodel'):retrain_pipeline.run(root,root/'output')
                loader.assert_not_called()
            self.assertFalse((root/'output/retrained_fixed_model.joblib').exists())

    def test_foreign_selection_change_stops_before_loading_or_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            (root/'reports/model_selection.json').write_text('{"fixed":"another model"}',encoding='utf-8')
            with patch.object(retrain_pipeline,'load_model') as loader:
                with self.assertRaisesRegex(ValueError,'selectie'):retrain_pipeline.run(root,root/'output')
                loader.assert_not_called()
