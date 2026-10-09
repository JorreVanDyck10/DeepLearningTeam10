"""Gedeelde operationele dagvooruit-inference, zonder training."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from .config import MODELS,REPORTS,sha256_file,fingerprint
from .features import make_features, validate_history


def predict_features(estimator,features,columns):
    predicted=np.maximum(np.asarray(estimator.predict(features[columns]),float),0)
    if not np.isfinite(predicted).all():
        raise ValueError('Model geeft niet-eindige voorspellingen')
    return np.where(features.clock_hours.to_numpy()==0,0,predicted)


def load_model(path=MODELS/'citibike_model.joblib'):
    canonical=Path(path).resolve()==(MODELS/'citibike_model.joblib').resolve()
    receipt=None
    if canonical:
        receipt=json.loads((REPORTS/'model_artifact.json').read_text(encoding='utf-8'))
        if sha256_file(Path(path))!=receipt['sha256']:
            raise ValueError('Modelbestand wijkt af van het vastgelegde artefact')
    artifact=joblib.load(path)
    if artifact['area']!='NYC' or artifact['target']!='hourly_demand':
        raise ValueError('Verkeerd modelgebied of doel')
    from .modeling import prediction_code_fingerprint
    if artifact['selection']['prediction_code_fingerprint']!=prediction_code_fingerprint():
        raise ValueError('Model vereist de vastgelegde feature-, inferencecode en packageversies')
    if receipt and fingerprint(artifact['selection'])!=receipt['selection_fingerprint']:
        raise ValueError('Modelselectie wijkt af van het vastgelegde artefact')
    return artifact


def predict_day(date, history: pd.DataFrame, artifact: dict | None=None) -> pd.DataFrame:
    origin=pd.Timestamp(date)
    if origin.tzinfo is not None or origin!=origin.normalize():
        raise ValueError('Voorspeldatum moet een offsetloze kalenderdag zijn')
    checked=validate_history(history)
    if (checked.timestamp>=origin).any():
        raise ValueError('Historie bevat voorspeldag of toekomstinformatie')
    expected=pd.date_range(origin-pd.Timedelta(days=14),origin,freq='h',inclusive='left')
    recent=checked[checked.timestamp>=expected[0]].copy()
    if not pd.DatetimeIndex(recent.timestamp).equals(expected):
        raise ValueError('Precies de veertien aansluitende historische dagen vereist')
    artifact=artifact or load_model()
    if artifact.get('area')!='NYC' or artifact.get('target')!='hourly_demand':
        raise ValueError('Dagvoorspelling vereist het NYC-uurmodel')
    future=pd.DataFrame({'timestamp':pd.date_range(origin,periods=24,freq='h'),
                         'rides':np.nan,'area':'NYC'})
    features=make_features(pd.concat([recent,future],ignore_index=True),future_day=True)
    result=features[features.day==origin][['timestamp','clock_hours']].copy()
    result['predicted_rides']=predict_features(artifact['model'],features[features.day==origin],artifact['features'])
    if len(result)!=24:
        raise AssertionError('Dagvoorspelling moet 24 kloklabels hebben')
    return result
