"""Serve the selected RF300 through a lossless, version-independent NumPy export."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / 'backend/artifacts/mushroom_rf300.json'
MODEL_ID = 'mushroom_final_rf300'
FEATURES = ['cap_diameter', 'stem_height', 'stem_width', 'spore_print_color',
            'gill_color', 'habitat', 'season', 'ring_type', 'cap_shape',
            'stem_surface', 'jumbled_noise_1']


def clean_measurements(frame):
    """Legacy pickle reference only; the selected RF300 does not use this cleaning."""
    result = frame.copy()
    result[['stem-height', 'stem-width']] = result[['stem-height', 'stem-width']].replace(0, np.nan)
    return result


class PortableMushroom:
    """Original imputation, one-hot encoding, float32 inputs and tree averaging."""
    def __init__(self, manifest_path=DEFAULT_MODEL):
        path = Path(manifest_path)
        self.metadata = json.loads(path.read_text(encoding='utf-8'))
        m = self.metadata
        if m['model_id'] != MODEL_ID or m['features'] != FEATURES or m['n_estimators'] != 300:
            raise ValueError('Unexpected Mushroom model or feature contract')
        if Path(m['forest_file']).name != m['forest_file']:
            raise ValueError('Invalid Mushroom artifact path')
        forest = path.parent / m['forest_file']
        if hashlib.sha256(forest.read_bytes()).hexdigest() != m['forest_sha256']:
            raise ValueError('Mushroom artifact checksum mismatch')
        with np.load(forest, allow_pickle=False) as arrays:
            self.left = arrays['left']; self.right = arrays['right']
            self.feature = arrays['feature']; self.threshold = arrays['threshold']
            self.probabilities = arrays['probabilities']; self.offsets = arrays['offsets']
        count = len(self.left)
        if (self.offsets.shape != (301,) or self.offsets[0] != 0 or self.offsets[-1] != count
                or not (np.diff(self.offsets) > 0).all()
                or any(len(v) != count for v in [self.right, self.feature, self.threshold, self.probabilities])
                or self.probabilities.shape != (count, 2)):
            raise ValueError('Invalid Mushroom forest schema')
        self.feature_names_in_ = np.asarray(FEATURES, dtype=object)
        self.classes_ = np.asarray([0, 1])

    def _encode(self, frame):
        if list(frame.columns) != FEATURES:
            raise ValueError('Incorrect Mushroom input columns/order')
        m = self.metadata
        numeric = frame[m['numeric_features']].apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
        numeric = np.where(np.isnan(numeric), np.asarray(m['numeric_medians']), numeric)
        if not np.isfinite(numeric).all():
            raise ValueError('Numeric Mushroom features must be finite or null')
        pieces = [numeric]
        for spec in m['categorical']:
            values = frame[spec['name']].astype(object).where(frame[spec['name']].notna(), spec['imputation']).to_numpy()
            pieces.append(np.column_stack([values == code for code in spec['categories']]).astype(float))
        return np.column_stack(pieces).astype(np.float32)

    def predict_proba(self, frame):
        values = self._encode(frame)
        result = np.zeros((len(frame), 2), dtype=float)
        for start, stop in zip(self.offsets[:-1], self.offsets[1:]):
            start, stop = int(start), int(stop)
            indices = np.zeros(len(frame), dtype=np.int32)
            for _ in range(21):
                active = self.left[start + indices] != -1
                if not active.any():
                    break
                rows = np.flatnonzero(active)
                current = start + indices[rows]
                columns = self.feature[current]
                if ((columns < 0) | (columns >= values.shape[1])).any():
                    raise ValueError('Invalid Mushroom split feature')
                indices[rows] = np.where(values[rows, columns] <= self.threshold[current], self.left[current], self.right[current])
                if ((indices[rows] < 0) | (start + indices[rows] >= stop)).any():
                    raise ValueError('Invalid Mushroom child index')
            else:
                raise ValueError('Mushroom tree exceeds declared maximum depth')
            result += self.probabilities[start + indices]
        return result / 300

    def predict(self, frame):
        return np.argmax(self.predict_proba(frame), axis=1)

    def predict_record(self, record):
        if set(record) != set(FEATURES):
            raise ValueError('Mushroom prediction requires the eleven selected features')
        for spec in self.metadata['categorical']:
            value = record[spec['name']]
            if value is not None and value not in spec['categories']:
                raise ValueError(f"Onbekende datasetcode voor {spec['name']}.")
        frame = pd.DataFrame([record], columns=FEATURES).replace({None: np.nan})
        proba = self.predict_proba(frame)[0]
        label = 'p' if np.argmax(proba) == 1 else 'e'
        return {'label': label, 'prediction': 'giftig' if label == 'p' else 'eetbaar',
                'probability_poisonous': float(proba[1])}
