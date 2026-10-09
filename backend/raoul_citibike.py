"""Serve the selected 320-tree RF with lossless memory-mapped inference arrays."""
import hashlib
import json
import mmap
import os
from pathlib import Path
import shutil
import threading
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from SolutionRaoul.Citybike.citibike.features import FEATURES
from SolutionRaoul.Citybike.citibike.inference import predict_day as shared_predict_day

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT / 'SolutionRaoul/Citybike'
RECEIPT = ROOT / 'backend/citibike_artifact.json'
MODEL_ID = 'raoul_tuned_random_forest_320'
NODE_DTYPE = np.dtype([('left', '<i4'), ('right', '<i4'), ('feature', 'i1'),
                       ('threshold', '<f8'), ('value', '<f8')])
LOAD_LOCK = threading.Lock()


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


class CompactForest:
    """Same tree splits, float32 inputs, float64 thresholds/leaves and averaging."""
    def __init__(self, directory):
        directory = Path(directory)
        self.nodes = np.load(directory/'forest.npy', mmap_mode='r', allow_pickle=False)
        self.offsets = np.load(directory/'offsets.npy', allow_pickle=False)
        if self.nodes.dtype != NODE_DTYPE or self.nodes.ndim != 1:
            raise ValueError('Invalid compact forest schema')
        if (self.offsets.shape != (321,) or self.offsets[0] != 0 or
                self.offsets[-1] != len(self.nodes) or not (np.diff(self.offsets) > 0).all()):
            raise ValueError('Invalid tree boundaries')
        if hasattr(self.nodes._mmap, 'madvise') and hasattr(mmap, 'MADV_RANDOM'):
            self.nodes._mmap.madvise(mmap.MADV_RANDOM)

    def predict(self, frame):
        if list(frame.columns) != FEATURES:
            raise ValueError('Incorrect feature order')
        values = frame.to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError('Non-finite features')
        result = np.zeros(len(frame), dtype=np.float64)
        for start, stop in zip(self.offsets[:-1], self.offsets[1:]):
            tree = self.nodes[int(start):int(stop)]
            indices = np.zeros(len(frame), dtype=np.int32)
            for _ in range(1024):
                active = tree['left'][indices] != -1
                if not active.any():
                    break
                rows = np.flatnonzero(active)
                current = indices[rows]
                columns = tree['feature'][current]
                if ((columns < 0) | (columns >= len(FEATURES))).any():
                    raise ValueError('Invalid split feature')
                indices[rows] = np.where(values[rows, columns] <= tree['threshold'][current],
                                         tree['left'][current], tree['right'][current])
                if ((indices[rows] < 0) | (indices[rows] >= len(tree))).any():
                    raise ValueError('Invalid child index')
            else:
                raise ValueError('Invalid cyclic or excessively deep tree')
            result += tree['value'][indices]
        return result/320


def ensure_artifact(directory=None):
    """Download only the checksum-pinned release; no pickle or training on server."""
    receipt = json.loads(RECEIPT.read_text(encoding='utf-8'))
    directory = Path(directory or os.getenv('RAOUL_CITIBIKE_MODEL_DIR', str(PROJECT/'models/serving')))
    with LOAD_LOCK:
        valid = directory.is_dir() and all((directory/name).is_file() and checksum(directory/name) == digest
                                           for name, digest in receipt['files'].items())
        if not valid:
            directory.parent.mkdir(parents=True, exist_ok=True)
            staging = directory.with_name(directory.name+'.partial')
            if staging.exists():
                raise RuntimeError(f'Incomplete download exists at {staging}; remove it before retrying')
            staging.mkdir()
            archive = staging/'artifact.zip'
            request = urllib.request.Request(receipt['url'], headers={'User-Agent': 'DeepLearningTeam10'})
            try:
                with urllib.request.urlopen(request, timeout=120) as response, archive.open('wb') as output:
                    shutil.copyfileobj(response, output, length=1024*1024)
                if checksum(archive) != receipt['archive_sha256']:
                    raise ValueError('Citi Bike release checksum mismatch')
                with zipfile.ZipFile(archive) as zipped:
                    if set(zipped.namelist()) != set(receipt['files']):
                        raise ValueError('Unexpected archive members')
                    for name, digest in receipt['files'].items():
                        if Path(name).name != name:
                            raise ValueError('Unsafe archive name')
                        with zipped.open(name) as source, (staging/name).open('wb') as output:
                            shutil.copyfileobj(source, output, length=1024*1024)
                        if checksum(staging/name) != digest:
                            raise ValueError(f'Invalid artifact file: {name}')
                archive.unlink()
                if directory.exists():
                    raise ValueError('Existing model directory differs from pinned release; do not overwrite it')
                staging.replace(directory)
            except BaseException:
                # Only remove files in the exact staging directory created above.
                for path in staging.iterdir():
                    if path.is_file():
                        path.unlink()
                staging.rmdir()
                raise
    for relative, digest in receipt['feature_code_sha256'].items():
        if checksum(PROJECT/relative) != digest:
            raise ValueError(f'Feature code changed: {relative}')
    metadata = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
    if metadata['model_id'] != MODEL_ID or metadata['features'] != FEATURES:
        raise ValueError('Incorrect deployment model')
    if metadata['original_model_sha256'] != receipt['original_model_sha256']:
        raise ValueError('Incorrect original model lineage')
    forest = CompactForest(directory)
    history = pd.read_csv(directory/'development_history.csv', parse_dates=['timestamp'])
    return {'model': forest, 'features': FEATURES, 'area': 'NYC', 'target': 'hourly_demand',
            'metadata': metadata, 'demo_history': history}


def model_metadata(artifact):
    metadata = artifact['metadata']
    return {key: metadata[key] for key in ['model_id', 'name', 'training_period', 'test_period',
            'metrics', 'baseline_metrics', 'RMSE_improvement_percent', 'demo_start', 'demo_end',
            'default_date', 'history_days', 'original_model_sha256']}


def predict(artifact, selected_date, hour, mode, history=None):
    origin = pd.Timestamp(selected_date)
    metadata = artifact['metadata']
    if mode == 'demo':
        if history is not None:
            raise ValueError('Kies eigen historie om tellingen aan te leveren.')
        if not pd.Timestamp(metadata['demo_start']) <= origin <= pd.Timestamp(metadata['demo_end']):
            raise ValueError('Demonstratiedatum buiten beschikbare ontwikkelingshistorie. Kies eigen historie voor andere datums.')
        checking = artifact['demo_history']
        checking = checking[(checking.timestamp >= origin-pd.Timedelta(days=14)) & (checking.timestamp < origin)]
        warning = 'Historische demonstratie; deze datum kan in de training zitten en is geen onafhankelijke test.'
    else:
        if history is None:
            raise ValueError('Lever 14 volledige voorafgaande dagen aan: timestamp, rides, area=NYC.')
        checking = pd.DataFrame(history)
        warning = 'Voorspelling met jouw aangeleverde NYC-historie; tellingen moeten vóór 00:00 op de voorspeldag bekend zijn.'
    result = shared_predict_day(origin, checking, artifact)
    rows = [{'hour': int(row.timestamp.hour), 'clock_hours': int(row.clock_hours),
             'predicted_ride_starts': float(row.predicted_rides)} for row in result.itertuples()]
    return {'date': selected_date.isoformat(), 'hour': hour, 'timezone': 'America/New_York',
            'model_id': MODEL_ID, 'mode': mode, 'predicted_ride_starts': rows[hour]['predicted_ride_starts'],
            'daily_predictions': rows, 'daily_total': float(result.predicted_rides.sum()),
            'training_period': metadata['training_period'], 'test_mae': metadata['metrics']['MAE'],
            'warning': warning+' Geregistreerde vertrekken, geen beschikbare fietsen. Weer en evenementen kunnen afwijken.',
            'prediction_origin': f'{selected_date.isoformat()}T00:00:00 America/New_York'}
