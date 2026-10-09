"""Export the evaluated RF without fitting or reading final-test observations."""
from pathlib import Path
import hashlib
import json
import sys
import time
import zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT/'SolutionRaoul/Citybike'
sys.path.insert(0, str(PROJECT))
from citibike.inference import load_model, predict_day
from citibike.features import FEATURES
from backend.raoul_citibike import CompactForest, NODE_DTYPE, MODEL_ID, checksum


def export():
    started = time.perf_counter()
    artifact = load_model()
    source_sha = checksum(PROJECT/'models/citibike_model.joblib')
    receipt = json.loads((PROJECT/'reports/model_artifact.json').read_text(encoding='utf-8'))
    assert receipt['sha256'] == source_sha
    forest = artifact['model'].estimator_
    assert len(forest.estimators_) == 320 and forest.n_outputs_ == 1
    directory = PROJECT/'models/serving'
    directory.mkdir(parents=True, exist_ok=False)
    lengths = [tree.tree_.node_count for tree in forest.estimators_]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype('<i8')
    nodes = np.lib.format.open_memmap(directory/'forest.npy', mode='w+', dtype=NODE_DTYPE, shape=(int(offsets[-1]),))
    for tree, start, stop in zip(forest.estimators_, offsets[:-1], offsets[1:]):
        current = nodes[int(start):int(stop)]
        current['left'] = tree.tree_.children_left
        current['right'] = tree.tree_.children_right
        current['feature'] = tree.tree_.feature
        current['threshold'] = tree.tree_.threshold
        current['value'] = tree.tree_.value[:,0,0]
    nodes.flush()
    del current, nodes
    np.save(directory/'offsets.npy', offsets, allow_pickle=False)
    history = pd.read_parquet(PROJECT/'data/NYC_hourly_development.parquet')
    demo = history[(history.timestamp >= '2026-04-01') & (history.timestamp < '2026-05-01')]
    demo.to_csv(directory/'development_history.csv', index=False)
    evaluation = json.loads((PROJECT/'reports/final_evaluation.json').read_text(encoding='utf-8'))
    metadata = {'model_id': MODEL_ID, 'name': 'Raoul — getunede Random Forest (320 bomen)',
                'features': FEATURES, 'original_model_sha256': source_sha,
                'training_period': [artifact['training_start'], artifact['trained_through']],
                'test_period': evaluation['test_period'], 'metrics': evaluation['metrics'],
                'baseline_metrics': evaluation['baseline_metrics'],
                'RMSE_improvement_percent': evaluation['RMSE_improvement_percent'],
                'demo_start': '2026-04-15', 'demo_end': '2026-04-30', 'default_date': '2026-04-30',
                'history_days': 14, 'nodes': int(offsets[-1]), 'trees': 320,
                'export': 'lossless tree splits and leaves; float32 inputs; float64 thresholds and means',
                'dataset_fingerprint': artifact['selection']['dataset_fingerprint'],
                'selection_fingerprint': evaluation['selection_fingerprint']}
    (directory/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    compact = CompactForest(directory)
    checking = pd.read_parquet(PROJECT/'data/modeling_development_validation.parquet')
    checking = checking[(checking.day >= '2026-05-01') & (checking.day < '2026-07-01')]
    original = artifact['model'].predict(checking[FEATURES])
    actual = np.maximum(compact.predict(checking[FEATURES]), 0)
    actual = np.where(checking.clock_hours.to_numpy() == 0, 0, actual)
    np.testing.assert_allclose(actual, original, rtol=1e-12, atol=1e-9)
    fixtures = []
    for date in ['2026-04-30', '2026-03-08', '2026-11-01']:
        origin = pd.Timestamp(date)
        recent = history[(history.timestamp >= origin-pd.Timedelta(days=14)) & (history.timestamp < origin)]
        if len(recent) != 336:
            recent = pd.DataFrame({'timestamp': pd.date_range(origin-pd.Timedelta(days=14), periods=336, freq='h'),
                                   'rides': np.tile(np.arange(24)*10+100,14), 'area': 'NYC'})
        expected = predict_day(origin, recent, artifact)
        fixtures.append({'date': date, 'history': [{**row, 'timestamp': row['timestamp'].isoformat()} for row in recent.to_dict('records')],
                         'expected': expected.predicted_rides.tolist(), 'clock_hours': expected.clock_hours.tolist()})
    fixture_dir = ROOT/'backend/fixtures'
    fixture_dir.mkdir(exist_ok=True)
    (fixture_dir/'citibike_parity.json').write_text(json.dumps(fixtures), encoding='utf-8')
    archive = PROJECT/'models/raoul-citibike-rf-20261009.zip'
    names = ['forest.npy','offsets.npy','metadata.json','development_history.csv']
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for name in names:
            zipped.write(directory/name, arcname=name)
    deployment = {'model_id': MODEL_ID, 'url': 'https://github.com/JorreVanDyck10/DeepLearningTeam10/releases/download/raoul-citibike-rf-20261009/'+archive.name,
                  'archive_sha256': checksum(archive), 'archive_bytes': archive.stat().st_size,
                  'original_model_sha256': source_sha,
                  'files': {name: checksum(directory/name) for name in names},
                  'feature_code_sha256': {f'citibike/{name}': checksum(PROJECT/'citibike'/name)
                                          for name in ['features.py','inference.py','datasets.py','config.py']}}
    (ROOT/'backend/citibike_artifact.json').write_text(json.dumps(deployment,indent=2),encoding='utf-8')
    (ROOT/'frontend/citibike_metrics.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    results = {'model_id': MODEL_ID, 'original_model_sha256': source_sha,
               'validation_parity_rows': len(checking), 'prediction_max_absolute_difference': float(np.max(np.abs(actual-original))),
               'archive_bytes': archive.stat().st_size, 'compact_forest_bytes': (directory/'forest.npy').stat().st_size,
               'export_seconds': time.perf_counter()-started, 'test_observations_read': False,
               'training_performed': False, 'model_changed': False, 'release_url': deployment['url']}
    (PROJECT/'reports/frontend_model_export.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2), flush=True)


if __name__ == '__main__':
    export()
