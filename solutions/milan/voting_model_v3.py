"""Native model files for an optional voting challenger, with the original input.

Unlike the small RF proposal, using this reader requires boosting dependencies.
No pickle loading or automatic training is performed by this serving reader.
"""
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
optional=HERE/'.generated/boosting_packages'
if optional.exists():
    sys.path.insert(0,str(optional))
sys.path.insert(0,str(HERE.parents[1]))
import numpy as np
import pandas as pd
from backend.mushroom_model import PortableMushroom, FEATURES, MODEL_ID


class VotingMushroom:
    def __init__(self,manifest_path):
        path=Path(manifest_path)
        self.metadata=json.loads(path.read_text(encoding='utf-8'))
        m=self.metadata
        if (m['features']!=FEATURES or m['model_id']!=MODEL_ID
                or m['serving_format']!='milan_native_voting_v1'):
            raise ValueError('Unexpected voting input contract')
        recipe={k:m[k] for k in ['components','proposed_decision_threshold','proposed_policy']}
        digest=hashlib.sha256(json.dumps(recipe,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if digest!=m['source_pipeline_sha256']:
            raise ValueError('Voting recipe checksum mismatch')
        self.parts=[]
        for spec in m['components']:
            file=(path.parent/spec['file']).resolve()
            if not file.is_relative_to(path.parent.resolve()):
                raise ValueError('Invalid component path')
            raw=file.read_bytes()
            if file.suffix in ['.txt','.json']:
                raw=raw.replace(b'\r\n',b'\n')
            if hashlib.sha256(raw).hexdigest()!=spec['sha256']:
                raise ValueError('Voting component checksum mismatch')
            if spec['family']=='rf_selection':
                # RF metadata itself is checked here; its array checksum is
                # additionally checked by the existing PortableMushroom reader.
                model=PortableMushroom(file)
            elif spec['family']=='lightgbm':
                from lightgbm import Booster
                model=Booster(model_file=str(file))
            elif spec['family']=='catboost':
                from catboost import CatBoostClassifier
                model=CatBoostClassifier()
                model.load_model(str(file))
            else:
                raise ValueError('Unknown voting family')
            if not np.isfinite(spec['weight']) or spec['weight']<=0:
                raise ValueError('Invalid voting weight')
            self.parts.append((spec,model))
        if not self.parts or not np.isclose(sum(s['weight'] for s,_ in self.parts),1,rtol=0,atol=1e-12):
            raise ValueError('Voting weights must sum to one')
        self.feature_names_in_=np.asarray(FEATURES,dtype=object)
        self.classes_=np.asarray([0,1])

    def predict_proba(self,frame):
        if list(frame.columns)!=FEATURES:
            raise ValueError('Incorrect Mushroom input columns/order')
        probability=np.zeros(len(frame),dtype=float)
        for spec,model in self.parts:
            if spec['family']=='rf_selection':
                p=model.predict_proba(frame)[:,1]
            else:
                values=frame[spec['columns']].copy()
                for column in ['cap_diameter','stem_height','stem_width']:
                    values[column]=pd.to_numeric(values[column],errors='raise').astype(float)
                    if np.isinf(values[column]).any():
                        raise ValueError('Numeric Mushroom features must be finite or null')
                for column,categories in spec['categories'].items():
                    if spec['family']=='lightgbm':
                        values[column]=pd.Categorical(values[column],categories=categories)
                    else:
                        values[column]=values[column].fillna('__missing__').astype(str)
                p=(model.predict(values,num_threads=1) if spec['family']=='lightgbm'
                   else model.predict_proba(values,thread_count=1)[:,1])
            probability+=spec['weight']*np.asarray(p)
        if not np.isfinite(probability).all() or not ((probability>=0)&(probability<=1)).all():
            raise ValueError('Invalid voting probability')
        return np.column_stack([1-probability,probability])

    def predict(self,frame):
        return np.argmax(self.predict_proba(frame),axis=1)

    def predict_record(self,record):
        if set(record)!=set(FEATURES):
            raise ValueError('Mushroom prediction requires the eleven selected features')
        for spec in self.metadata['categorical']:
            value=record[spec['name']]
            if value is not None and value not in spec['categories']:
                raise ValueError(f"Onbekende datasetcode voor {spec['name']}.")
        p=self.predict_proba(pd.DataFrame([record],columns=FEATURES).replace({None:np.nan}))[0]
        label='p' if np.argmax(p)==1 else 'e'
        return {'label':label,'prediction':'giftig' if label=='p' else 'eetbaar',
            'probability_poisonous':float(p[1])}
