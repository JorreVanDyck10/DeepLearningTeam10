"""Validate and install only the declared candidate files; never touch research artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from backend.mushroom_model import PortableMushroom, FEATURES

FILES = ['Final/Mushroom/artifacts/current_pipeline.joblib', 'Final/Mushroom/artifacts/training_run.json',
         'backend/artifacts/mushroom_rf300.npz', 'backend/artifacts/mushroom_rf300.json',
         'backend/artifacts/mushroom_export_audit.json', 'backend/artifacts/mushroom_reference_predictions.csv']


def promote(folder):
    folder = Path(folder).resolve()
    for name in FILES:
        source = (folder/name).resolve()
        if not source.is_relative_to(folder) or not source.is_file() or source.stat().st_size >= 100_000_000:
            raise ValueError(f'Invalid/missing candidate file: {name}')
    model = PortableMushroom(folder/'backend/artifacts/mushroom_rf300.json')
    expected_hash = model.metadata['source_pipeline_sha256']
    if hashlib.sha256((folder/FILES[0]).read_bytes()).hexdigest() != expected_hash:
        raise ValueError('Candidate pipeline checksum mismatch')
    reference = pd.read_csv(folder/FILES[-1], float_precision='round_trip')
    study_rows = pd.read_csv(ROOT/'MushroomFinal/comparison_correct/test_predictions.csv', usecols=['source_row'])
    if reference.source_row.tolist() != study_rows.source_row.tolist():
        raise ValueError('Candidate references do not match the frozen development-test rows')
    raw = pd.read_csv(ROOT/'MushroomFinal/mushroom_project_dataset.csv')
    raw.columns = raw.columns.str.replace('-', '_', regex=False)
    if len(reference) != 900 or reference.source_row.duplicated().any():
        raise ValueError('Invalid candidate reference records')
    expected_labels = raw['class'].map({'e': 0, 'p': 1}).iloc[reference.source_row]
    np.testing.assert_array_equal(reference.true_class, expected_labels)
    values = model.predict_proba(raw[FEATURES].iloc[reference.source_row])
    np.testing.assert_allclose(values[:, 1], reference.probability_p, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(np.argmax(values, axis=1), reference.prediction)
    config = json.loads((ROOT/'Final/config/mushroom.json').read_text())
    run = json.loads((folder/FILES[1]).read_text())
    if (run['cv']['cv_macro_f1_mean']+config['tolerance'] < config['minimum_cv_macro_f1']
            or run['cv']['cv_recall_p_mean']+config['tolerance'] < config['minimum_cv_recall_p']):
        raise ValueError('Candidate did not pass the training-CV quality gates')
    for name in FILES:
        target = (ROOT/name).resolve()
        if not target.is_relative_to(ROOT): raise ValueError('Unsafe candidate destination')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((folder/name).read_bytes())
    print('Validated candidate installed:', expected_hash)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder'); args = parser.parse_args(); promote(args.folder)
