"""Download the exact UCI source and reconstruct the AWS train/test membership."""
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
URL = 'https://archive.ics.uci.edu/static/public/848/secondary+mushroom+dataset.zip'
DATA = ROOT / 'secondary_mushroom/data/raw/secondary_data.csv'


def load_data():
    DATA.parent.mkdir(parents=True, exist_ok=True)
    if not DATA.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            archive = response.read()
        with zipfile.ZipFile(io.BytesIO(archive)) as outer:
            inner_bytes = outer.read('MushroomDataset.zip')
        with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
            csv = inner.read('MushroomDataset/secondary_data.csv')
        DATA.write_bytes(csv)
        DATA.with_suffix('.manifest.json').write_text(json.dumps({
            'url': URL, 'archive_sha256': hashlib.sha256(archive).hexdigest(),
            'csv_sha256': hashlib.sha256(csv).hexdigest(),
        }, indent=2) + '\n', encoding='utf-8')
    raw = pd.read_csv(DATA, sep=';', na_values=['?'])
    assert raw.shape == (61069, 21) and raw['class'].isin(['e', 'p']).all()
    first_rows = np.flatnonzero(~raw.duplicated().to_numpy())
    data = raw.drop_duplicates().reset_index(drop=True)
    assert data.shape == (60923, 21)
    X = data.drop(columns='class')
    y = data['class'].map({'e': 0, 'p': 1})
    train, test = train_test_split(np.arange(len(data)), test_size=.2, random_state=42, stratify=y)
    assert len(train) == 48738 and len(test) == 12185
    assert y.iloc[test].value_counts().to_dict() == {1: 6749, 0: 5436}
    return raw, data, X, y, train, test, first_rows
