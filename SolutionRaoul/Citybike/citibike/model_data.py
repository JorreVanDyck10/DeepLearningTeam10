"""Zelfstandige modelvoorbereiding; geen model- of onderzoeksresultaten nodig."""
import json
from pathlib import Path

import pandas as pd

from .config import DATA, REPORTS, SOURCE, TRAIN_END, TEST_START, TEST_END, atomic_json, fingerprint, now, sha256_file
from .datasets import historical_hourly, manifest
from .features import FEATURES, make_features, folds
from .prepare import prepare

DATASET_NAME = 'modeling_development_validation.parquet'
MANIFEST_NAME = 'prepared_model_data.json'
CONTRACT = {
    'target': 'hourly_demand', 'area': 'NYC',
    'prediction_time': '00:00 America/New_York op voorspeldag D',
    'history_days': 14, 'features': FEATURES,
    'window_years': None,
    'train_end_exclusive': TRAIN_END, 'validation_end_exclusive': TEST_START,
    'test_start': TEST_START, 'test_end_exclusive': TEST_END,
}


def save_prepared_frame(frame, dataset_fingerprint, data_dir=DATA, report_dir=REPORTS):
    """Bestaande modelinvoer mag alleen worden hergebruikt als alle waarden gelijk zijn."""
    data_dir, report_dir = Path(data_dir), Path(report_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    frame = frame.reset_index(drop=True)
    training = frame[frame.day < TRAIN_END].reset_index(drop=True)
    validation = frame[(frame.day >= TRAIN_END) & (frame.day < TEST_START)].reset_index(drop=True)
    if len(frame) != len(training) + len(validation) or set(frame.area) != {'NYC'}:
        raise ValueError('Modeldata mogen uitsluitend NYC-training en validatie bevatten')
    if training.empty or validation.empty or frame.timestamp.duplicated().any() or not frame.timestamp.is_monotonic_increasing:
        raise ValueError('Onvolledige of niet-chronologische modeldata')
    bounds = folds(training)
    path = data_dir / DATASET_NAME
    reused = path.exists()
    if reused:
        # Stop vóór overschrijven bij een inhoudelijke wijziging na de bekeken eindtest.
        pd.testing.assert_frame_equal(pd.read_parquet(path), frame, check_exact=True)
    else:
        staging = path.with_suffix('.parquet.tmp')
        frame.to_parquet(staging, index=False)
        staging.replace(path)
    receipt = {
        'prepared_at': now(), 'contract': CONTRACT,
        'path': str(path.resolve()), 'sha256': sha256_file(path),
        'dataset_fingerprint': dataset_fingerprint, 'existing_dataset_equal': reused,
        'rows': len(frame), 'train_rows': len(training), 'validation_rows': len(validation),
        'train_start': str(training.day.min()), 'train_end_exclusive': TRAIN_END,
        'validation_start': TRAIN_END, 'validation_end_exclusive': TEST_START,
        'test_start': TEST_START, 'test_end_exclusive': TEST_END,
        'folds': bounds, 'fold_fingerprint': fingerprint(bounds),
        'training_results_required': False, 'test_labels_read': False,
        'test_preparation': 'Ruwe testpartitions technisch opgeschoond; testtoegang blijft vergrendeld. Geen testfeatures of scores in modelnotebooks.',
    }
    atomic_json(report_dir / MANIFEST_NAME, receipt)
    return receipt


def prepare_model_data(source=SOURCE):
    """ZIPs → cleaning → uuraggregatie → historische features → vaste splits."""
    # De bestaande streaminglaag gebruikt een stagingdatabase in deze directory.
    (DATA / 'temporary').mkdir(parents=True, exist_ok=True)
    raw = prepare(source)
    history = historical_hourly(include_validation=True)
    frame = make_features(history)
    receipt = save_prepared_frame(frame, raw['dataset_fingerprint'])
    # Dit uurbestand is ook de invoer voor de bestaande EDA/deploymentdemo.
    development = history[history.timestamp < TRAIN_END].reset_index(drop=True)
    history_path = DATA / 'NYC_hourly_development.parquet'
    if history_path.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(history_path), development, check_exact=True)
    else:
        development.to_parquet(history_path, index=False)
    return receipt


def load_model_data(data_dir=DATA, report_dir=REPORTS):
    """Eén gecontroleerde ingang voor notebooks en vaste hertraining."""
    path = Path(data_dir) / DATASET_NAME
    receipt = json.loads((Path(report_dir) / MANIFEST_NAME).read_text(encoding='utf-8'))
    if receipt['contract'] != CONTRACT or sha256_file(path) != receipt['sha256']:
        raise ValueError('Voorbereide dataset of featurecontract gewijzigd; voer notebook 02 uit')
    frame = pd.read_parquet(path)
    training = frame[frame.day < TRAIN_END].reset_index(drop=True)
    validation = frame[(frame.day >= TRAIN_END) & (frame.day < TEST_START)].reset_index(drop=True)
    if len(frame) != receipt['rows'] or len(training) != receipt['train_rows'] or len(validation) != receipt['validation_rows']:
        raise ValueError('Onjuiste rijen in voorbereide modeldata')
    if fingerprint(folds(training)) != receipt['fold_fingerprint']:
        raise ValueError('Chronologische folds gewijzigd')
    return training, validation, receipt
