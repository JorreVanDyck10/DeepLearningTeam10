"""Controles van de nieuwe voorbereidingsgrens, niet van gekopieerde implementatiedetails."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from citibike.model_data import save_prepared_frame,load_model_data
from citibike.features import make_features


class PreparedDataTests(unittest.TestCase):
    def frame(self):
        times=pd.date_range('2025-11-01','2026-07-01',freq='h',inclusive='left')
        counts=np.arange(len(times),dtype=float)%100
        nonexistent=times.tz_localize('America/New_York',ambiguous=True,nonexistent='NaT').isna()
        counts[nonexistent]=0
        return make_features(pd.DataFrame({'timestamp':times,'rides':counts,'area':'NYC'}))

    def test_preparation_has_no_model_reports_and_preserves_identical_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);frame=self.frame()
            first=save_prepared_frame(frame,'source',root/'data',root/'reports')
            second=save_prepared_frame(frame,'source',root/'data',root/'reports')
            self.assertEqual(first['sha256'],second['sha256'])
            training,validation,_=load_model_data(root/'data',root/'reports')
            self.assertTrue((training.day<'2026-05-01').all())
            self.assertTrue((validation.day>='2026-05-01').all())
            self.assertTrue((validation.day<'2026-07-01').all())
            self.assertEqual(len(list((root/'reports').iterdir())),1)

    def test_changed_model_input_stops_before_overwriting(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);frame=self.frame()
            receipt=save_prepared_frame(frame,'source',root/'data',root/'reports')
            changed=frame.copy();changed.loc[0,'lag_1d']+=1
            with self.assertRaises(AssertionError):save_prepared_frame(changed,'source',root/'data',root/'reports')
            _,_,original=load_model_data(root/'data',root/'reports')
            self.assertEqual(receipt['sha256'],original['sha256'])

    def test_test_labels_cannot_be_in_prepared_training_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);frame=self.frame()
            frame.loc[len(frame)-1,'day']=pd.Timestamp('2026-07-01')
            with self.assertRaises(ValueError):save_prepared_frame(frame,'source',root/'data',root/'reports')
            self.assertFalse((root/'data/modeling_development_validation.parquet').exists())
