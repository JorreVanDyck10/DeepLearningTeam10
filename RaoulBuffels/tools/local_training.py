"""Meet lokale hertraining en een begrensd, afzonderlijk vervolgexperiment.

Geen wijzigingen aan het gedeployde model, oorspronkelijke rapporten of eindtest.
"""
from pathlib import Path
import argparse
import copy
import gc
import json
import platform
import sys
import threading
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
import psutil
from lightgbm import LGBMRegressor
from sklearn.base import clone
from citibike.config import DATA, MODELS, REPORTS, atomic_json, code_version, fingerprint, now, sha256_file
from citibike.features import FEATURES, SavedChronologicalCV
from citibike.inference import load_model, predict_features
from citibike.modeling import OperationalRegressor, metrics


class Measurement:
    """Sample proces-RSS plus eventuele subprocessen iedere 0,2 seconde."""
    def __enter__(self):
        self.started = time.perf_counter()
        self.cpu_start = self.cpu_seconds()
        self.peak = 0
        self.samples = 0
        self.stop = threading.Event()
        self.worker = threading.Thread(target=self.sample, daemon=True)
        self.worker.start()
        return self

    @staticmethod
    def cpu_seconds():
        cpu = psutil.Process().cpu_times()
        return cpu.user + cpu.system

    def sample(self):
        while True:
            root = psutil.Process()
            total = 0
            for process in [root, *root.children(recursive=True)]:
                try:
                    total += process.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            self.peak = max(self.peak, total)
            self.samples += 1
            if self.stop.wait(.2):
                break

    def __exit__(self, *args):
        self.stop.set()
        self.worker.join()
        self.result = {'seconds': time.perf_counter() - self.started,
                       'process_cpu_seconds': self.cpu_seconds() - self.cpu_start,
                       'sampled_process_tree_peak_RSS_GiB': self.peak / 1024**3,
                       'RSS_samples': self.samples, 'sampling_interval_seconds': .2}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_frame(split):
    frame = pd.read_parquet(DATA / 'modeling_development_validation.parquet')
    if not frame.area.eq('NYC').all() or not (frame.day < pd.Timestamp(split['test_start'])).all():
        raise ValueError('Alleen NYC-ontwikkeling en validatie toegestaan')
    if frame.timestamp.duplicated().any() or not frame.timestamp.is_monotonic_increasing:
        raise ValueError('Uurreeks is niet uniek en chronologisch')
    return frame


def retrain(report_dir, model_dir, split):
    print('Hertraining: vaste geëvalueerde configuratie; geen selectie of testlabels.', flush=True)
    with Measurement() as total:
        with Measurement() as loading:
            artifact = load_model()
            frame = load_frame(split)
            frame = frame[frame.day >= pd.Timestamp(artifact['training_start'])].reset_index(drop=True)
            assert len(frame) == artifact['training_rows']
        with Measurement() as fitting:
            reproduced = clone(artifact['model']).fit(frame[FEATURES], frame.rides)
        print(f'Fit voltooid: {fitting.result["seconds"]:.1f} seconden.', flush=True)
        checking = frame[frame.day >= pd.Timestamp(split['validation_start'])]
        original_predictions = predict_features(artifact['model'], checking, FEATURES)
        refit_predictions = predict_features(reproduced, checking, FEATURES)
        np.testing.assert_allclose(refit_predictions, original_predictions, rtol=1e-12, atol=1e-9)
        reproduced_artifact = copy.copy(artifact)
        reproduced_artifact['model'] = reproduced
        path = model_dir / 'retrained_fixed_model.joblib'
        with Measurement() as saving:
            joblib.dump(reproduced_artifact, path)
        # Werkelijk herladen bestand controleren; deze rijen zijn in de refit gebruikt,
        # dus deze vergelijking is uitsluitend reproductie, geen prestatiemeting.
        del reproduced_artifact, reproduced
        gc.collect()
        reloaded = load_model(path)
        np.testing.assert_allclose(predict_features(reloaded['model'], checking, FEATURES),
                                   original_predictions, rtol=1e-12, atol=1e-9)
        result = {'started_at': now(), 'rows': len(frame), 'training_end_exclusive': split['test_start'],
                  'model_parameters': artifact['model'].get_params(deep=True),
                  'loading': loading.result, 'fit': fitting.result, 'saving': saving.result,
                  'prediction_max_absolute_difference': float(np.max(np.abs(refit_predictions-original_predictions))),
                  'saved_model': str(path), 'saved_model_sha256': sha256_file(path),
                  'saved_model_MiB': path.stat().st_size / 1024**2,
                  'reloaded_predictions_verified': True,
                  'comparison_is_accuracy_evaluation': False, 'test_used': False}
    result['total'] = total.result
    atomic_json(report_dir / 'retraining.json', result)
    return result


def experiment(report_dir, model_dir, split):
    # Bevries de kleine zoekruimte vóór fitting. Geen keuze uit eindtestresultaten.
    common = dict(n_estimators=400, num_leaves=31, learning_rate=.05,
                  min_child_samples=50, n_jobs=2, verbosity=-1, random_state=42,
                  deterministic=True, force_col_wise=True)
    configurations = {
        'count_poisson': {**common, 'objective': 'poisson'},
        'robust_huber': {**common, 'objective': 'huber', 'alpha': .9},
        'regularized_squared_error': {**common, 'objective': 'regression',
                                     'num_leaves': 15, 'reg_lambda': 10.},
    }
    protocol = {'created_at': now(), 'configurations': configurations,
                'folds': split['folds'], 'fold_fingerprint': split['fold_fingerprint'],
                'selection': 'minimum mean chronological CV RMSE; validation only after CV selection',
                'status': 'exploratory follow-up after original study; not independent confirmation',
                'same_features': FEATURES, 'test_used': False, 'deployment_changed': False}
    assert fingerprint(split['folds']) == split['fold_fingerprint']
    atomic_json(report_dir / 'experiment_protocol.json', protocol)
    with Measurement() as total:
        frame = load_frame(split)
        training = frame[(frame.day >= pd.Timestamp(split['train_start'])) &
                         (frame.day < pd.Timestamp(split['train_end_exclusive']))].reset_index(drop=True)
        validation = frame[(frame.day >= pd.Timestamp(split['validation_start'])) &
                           (frame.day < pd.Timestamp(split['validation_end_exclusive']))].reset_index(drop=True)
        assert len(training) == split['train_rows'] and len(validation) == split['validation_rows']
        audit = read_json(REPORTS / 'pycaret_split_audit.json')
        assert fingerprint(training.timestamp.astype(str).tolist()) == audit['train_data_fingerprint']
        assert fingerprint(validation.timestamp.astype(str).tolist()) == audit['validation_data_fingerprint']
        rows = []
        for name, parameters in configurations.items():
            scores = []
            with Measurement() as measured:
                for fold, (train_idx, val_idx) in enumerate(SavedChronologicalCV(split['folds']).split(training), 1):
                    model = OperationalRegressor(LGBMRegressor(**parameters))
                    start = time.perf_counter()
                    model.fit(training.iloc[train_idx][FEATURES], training.iloc[train_idx].rides)
                    score = metrics(training.iloc[val_idx].rides, model.predict(training.iloc[val_idx][FEATURES]))
                    scores.append({'fold': fold, **score, 'seconds': time.perf_counter()-start})
                    del model
            row = {'name': name, 'parameters': parameters, 'folds': scores,
                   'cv_RMSE': float(np.mean([score['RMSE'] for score in scores])),
                   'measurement': measured.result}
            rows.append(row)
            atomic_json(report_dir / 'experiment_progress.json', rows)
            print(f'{name}: CV-RMSE {row["cv_RMSE"]:.2f}; {measured.result["seconds"]:.1f}s.', flush=True)
        winner = min(rows, key=lambda row: row['cv_RMSE'])
        with Measurement() as final_fit:
            selected = OperationalRegressor(LGBMRegressor(**winner['parameters'])).fit(training[FEATURES], training.rides)
        predictions = selected.predict(validation[FEATURES])
        validation_metrics = metrics(validation.rides, predictions)
        baseline_metrics = metrics(validation.rides, validation.lag_7d)
        original_comparison = read_json(REPORTS / 'model_comparison.json')
        original = next(row for row in original_comparison if row['name'] == 'tuned_random_forest')
        original_lgbm = next(row for row in original_comparison if row['name'] == 'lightgbm')
        # Geen refit op validatie, geen wijziging aan het productiemodel.
        path = model_dir / 'experimental_development_model.joblib'
        joblib.dump({'model': selected, 'features': FEATURES, 'area': 'NYC',
                     'target': 'hourly_demand', 'protocol': protocol,
                     'trained_through': str(training.day.max()),
                     'deployment_approved': False}, path)
        pd.DataFrame({'timestamp': validation.timestamp, 'actual_rides': validation.rides,
                      'predicted_rides': predictions}).to_csv(report_dir / 'experimental_validation_predictions.csv', index=False)
        result = {'protocol': protocol, 'candidates': rows, 'selected_by_CV': winner['name'],
                  'selected_validation_metrics': validation_metrics, 'final_fit': final_fit.result,
                  'baseline_validation_metrics': baseline_metrics,
                  'original_RF_validation_metrics': {key: original[key] for key in ['RMSE','MAE','R2']},
                  'original_LightGBM_validation_metrics': {key: original_lgbm[key] for key in ['RMSE','MAE','R2']},
                  'RMSE_change_percent_vs_original_RF': 100*(validation_metrics['RMSE']/original['RMSE']-1),
                  'saved_experimental_model': str(path), 'model_sha256': sha256_file(path),
                  'train_rows': len(training), 'validation_rows': len(validation),
                  'original_model_training_end_exclusive': split['train_end_exclusive'],
                  'test_used': False, 'deployment_changed': False,
                  'independent_test_for_new_candidate': 'not yet available; original July-August evaluation is spent'}
    result['total'] = total.result
    atomic_json(report_dir / 'experiment.json', result)
    print(f'CV-winnaar {winner["name"]}: validatie-RMSE {validation_metrics["RMSE"]:.2f}.', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['retrain', 'experiment', 'all'], default='all', nargs='?')
    args = parser.parse_args()
    identifier = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    report_dir = REPORTS / 'local_runs' / identifier
    model_dir = MODELS / 'local_runs' / identifier
    report_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=False)
    protected = [MODELS/'citibike_model.joblib', REPORTS/'model_selection.json',
                 REPORTS/'model_artifact.json', REPORTS/'evaluation_lock.json', REPORTS/'final_evaluation.json']
    before = {str(path): sha256_file(path) for path in protected}
    manifest = read_json(REPORTS / 'data_manifest.json')
    split = read_json(REPORTS / 'chronological_splits.json')
    summary = {'started_at': now(), 'command': args.command, 'python': sys.executable,
               'platform': platform.platform(), 'logical_CPUs': psutil.cpu_count(),
               'physical_CPUs': psutil.cpu_count(logical=False),
               'RAM_GiB': psutil.virtual_memory().total/1024**3,
               'source_fingerprint': manifest['source_fingerprint'],
               'dataset_fingerprint': manifest['dataset_fingerprint'],
               'modeling_parquet_sha256': sha256_file(DATA/'modeling_development_validation.parquet'),
               'code_version': code_version([Path(__file__)]),
               'fold_fingerprint': split['fold_fingerprint'], 'reports': str(report_dir)}
    atomic_json(report_dir / 'run.json', summary)
    print(f'Rapporten: {report_dir}', flush=True)
    started = time.perf_counter()
    if args.command in ['retrain', 'all']:
        summary['retraining'] = retrain(report_dir, model_dir, split)
        gc.collect()
    if args.command in ['experiment', 'all']:
        summary['experiment'] = experiment(report_dir, model_dir, split)
    after = {str(path): sha256_file(path) for path in protected}
    assert before == after, 'Oorspronkelijke studie of model is gewijzigd'
    summary.update(finished_at=now(), total_seconds=time.perf_counter()-started,
                   original_model_and_evaluation_unchanged=True, protected_checksums=after)
    atomic_json(report_dir / 'run.json', summary)
    atomic_json(REPORTS / 'local_runs' / 'latest.json', {'run_report': str(report_dir/'run.json')})
    print(f'Klaar: {summary["total_seconds"]:.1f} seconden. Origineel model en eindtest ongewijzigd.', flush=True)


if __name__ == '__main__':
    main()
