"""Single preparation contract for the final 5,000-row Mushroom submission."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'MushroomFinal/mushroom_project_dataset.csv'
SOURCE_SHA256_LF = '1f1a25f2f330458ed95ce9f6ffe3241582312a42cb21797a2b7bd5e5ad281114'
FEATURES = ['cap_diameter', 'stem_height', 'stem_width', 'spore_print_color',
            'gill_color', 'habitat', 'season', 'ring_type', 'cap_shape', 'stem_surface', 'jumbled_noise_1']


def prepare():
    digest = hashlib.sha256(SOURCE.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    if digest != SOURCE_SHA256_LF:
        raise ValueError('Unexpected course source: review the dataset contract before retraining')
    raw = pd.read_csv(SOURCE)
    raw.columns = raw.columns.str.strip().str.lower().str.replace('-', '_', regex=False)
    if raw.shape != (5000, 13) or raw.duplicated().any() or not raw['class'].isin(['e', 'p']).all():
        raise ValueError('Invalid course data schema, labels or duplicate rows')
    X = raw[FEATURES].copy()
    y = raw['class'].map({'e': 0, 'p': 1})
    train, test = train_test_split(np.arange(len(raw)), test_size=.18, random_state=42, stratify=y)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X.iloc[train], y.iloc[train]))
    return raw, X, y, train, test, folds
