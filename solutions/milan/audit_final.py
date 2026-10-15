"""Read-only audit of Final and its saved evidence; writes only our own report."""
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mushroom_alternative import development_data, scores, ROOT, HERE


def run():
    findings = {'scope': 'Local source, stored predictions, serving export and synthetic causality checks. No new cloud training or live-hosting check.'}
    X, y, ids, folds = development_data()
    findings['training_rows'] = len(X)
    manifest = json.loads((ROOT / 'Final/source_manifest.json').read_text(encoding='utf-8'))
    findings['source_hash_mismatches'] = []
    for row in manifest['copies']:
        digest = hashlib.sha256((ROOT / row['source']).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if digest != row['source_sha256_lf']:
            findings['source_hash_mismatches'].append(row['source'])
    findings['source_copies_checked'] = len(manifest['copies'])
    notebooks, broken = [], []
    for path in sorted((ROOT / 'Final').rglob('*.ipynb')):
        nb = json.loads(path.read_text(encoding='utf-8'))
        cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
        notebooks.append({'path': path.relative_to(ROOT).as_posix(),
            'code_cells': len(cells), 'nonempty_code_cells': sum(bool(''.join(c['source']).strip()) for c in cells),
            'execution_counts_present': sum(c.get('execution_count') is not None for c in cells),
            'stored_errors': sum(o.get('output_type') == 'error' for c in cells for o in c.get('outputs', []))})
        for i, c in enumerate(nb['cells']):
            if c['cell_type'] != 'markdown':
                continue
            for url in re.findall(r'\]\(([^)]+)\)', ''.join(c['source'])):
                if '://' in url or url.startswith('#'):
                    continue
                if not (path.parent / url.split('#')[0]).exists():
                    broken.append({'notebook': path.relative_to(ROOT).as_posix(), 'cell_index': i, 'link': url})
    findings['notebooks'] = notebooks
    findings['broken_notebook_links'] = broken
    out = ROOT / 'MushroomFinal/comparison_correct'
    reference = pd.read_csv(out / 'model_comparison.csv').set_index('model')
    cv_reference = pd.read_csv(out / 'cv_folds.csv')
    oof = pd.read_csv(out / 'oof_predictions.csv', float_precision='round_trip')
    test = pd.read_csv(out / 'test_predictions.csv', float_precision='round_trip')
    assert oof.source_row.tolist() == ids.tolist()
    comparisons = 0
    for key, row in reference.iterrows():
        values = scores(test.true_class, test[key + '_probability_p'], prediction=test[key + '_prediction'])
        # Saved label arrays are authoritative for potential exact-probability ties.
        assert np.array_equal(test[key + '_prediction'], (test[key + '_probability_p'] > .5).astype(int))
        for metric in ['accuracy', 'macro_f1', 'recall_p', 'precision_p', 'roc_auc', 'tn', 'fp', 'fn', 'tp']:
            assert abs(values[metric] - row[metric]) < 1e-12, (key, metric)
            comparisons += 1
        for fold, (_, val) in enumerate(folds, 1):
            values = scores(oof.true_class.iloc[val], oof[key + '_probability_p'].iloc[val], prediction=oof[key + '_prediction'].iloc[val])
            saved = cv_reference[(cv_reference.model == key) & (cv_reference.fold == fold)].iloc[0]
            for metric in ['accuracy', 'macro_f1', 'recall_p', 'precision_p', 'roc_auc', 'fn', 'fp']:
                assert abs(values[metric] - saved[metric]) < 1e-12, (key, fold, metric)
                comparisons += 1
    findings['mushroom_saved_metric_comparisons'] = comparisons
    sys.path.insert(0, str(ROOT))
    from backend.mushroom_model import PortableMushroom, FEATURES
    model = PortableMushroom()
    data = pd.read_csv(ROOT / 'MushroomFinal/mushroom_project_dataset.csv')
    data.columns = data.columns.str.replace('-', '_', regex=False)
    probability = model.predict_proba(data.iloc[test.source_row][FEATURES])[:, 1]
    delta = float(np.max(np.abs(probability - test.rf_final_probability_p)))
    assert delta < 1e-12
    findings['portable_export_rows'] = len(test)
    findings['portable_export_max_probability_difference'] = delta
    artifact = ROOT / model.metadata['source_pipeline']
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == model.metadata['source_pipeline_sha256']
    findings['active_pipeline_hash_matches_metadata'] = True
    folder = ROOT / 'SolutionRaoul/Citybike/reports'
    bike = pd.read_csv(folder / 'test_predictions.csv')
    report = json.loads((folder / 'final_evaluation.json').read_text(encoding='utf-8'))
    recomputed = {'RMSE': float(np.sqrt(mean_squared_error(bike.rides, bike.predicted_rides))),
        'MAE': float(mean_absolute_error(bike.rides, bike.predicted_rides)),
        'R2': float(r2_score(bike.rides, bike.predicted_rides))}
    assert all(abs(v - report['metrics'][k]) < 1e-8 for k, v in recomputed.items())
    findings['citibike_existing_test_metrics_recomputed'] = recomputed
    findings['citibike_existing_test_rows'] = len(bike)
    splits = json.loads((folder / 'chronological_splits.json').read_text(encoding='utf-8'))
    assert all(f['train_stop'] <= f['validation_start'] and
               f['train_last_day'] < f['validation_first_day'] for f in splits['folds'])
    findings['citibike_chronological_folds_checked'] = len(splits['folds'])
    sys.path.insert(0, str(ROOT / 'SolutionRaoul/Citybike'))
    from citibike.features import make_features, FEATURES as BIKE_FEATURES
    times = pd.date_range('2026-01-01', periods=45 * 24, freq='h')
    history = pd.DataFrame({'timestamp': times, 'rides': np.arange(len(times)) % 100 + 10, 'area': 'NYC'})
    before = make_features(history)
    changed = history.copy()
    day = pd.Timestamp('2026-01-25')
    changed.loc[changed.timestamp >= day, 'rides'] += 1000
    after = make_features(changed)
    pd.testing.assert_frame_equal(before.loc[before.timestamp < day + pd.Timedelta(days=1), BIKE_FEATURES],
                                 after.loc[after.timestamp < day + pd.Timedelta(days=1), BIKE_FEATURES])
    assert not before.loc[before.timestamp >= day + pd.Timedelta(days=1), BIKE_FEATURES].equals(
        after.loc[after.timestamp >= day + pd.Timedelta(days=1), BIKE_FEATURES])
    findings['citibike_no_same_day_or_future_target_leakage_synthetic_check'] = True
    path = HERE / 'results/audit_final.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(findings, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in findings.items() if k != 'notebooks'}, indent=2, ensure_ascii=False))
    return findings


if __name__ == '__main__':
    run()
