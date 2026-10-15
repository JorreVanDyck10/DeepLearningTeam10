"""Vaste RF opnieuw fitten en als gecontroleerd artefact bewaren, zonder live rollout."""
from pathlib import Path
import argparse
import copy
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from citibike.config import atomic_json, code_version, sha256_file, fingerprint
from citibike.features import FEATURES
from citibike.inference import load_model, predict_features
from citibike.model_data import load_model_data


def run(training_root, output):
    training_root, output=Path(training_root).resolve(),Path(output).resolve()
    if output.exists(): raise FileExistsError('Gebruik een nieuwe outputmap per run: '+str(output))
    output.mkdir(parents=True)
    protected_names=['models/citibike_model.joblib','reports/model_selection.json',
                     'reports/model_artifact.json','reports/final_evaluation.json','reports/evaluation_lock.json']
    before={name:sha256_file(training_root/name) for name in protected_names}
    receipt=json.loads((training_root/'reports/model_artifact.json').read_text(encoding='utf-8'))
    selection=json.loads((training_root/'reports/model_selection.json').read_text(encoding='utf-8'))
    if before['models/citibike_model.joblib']!=receipt['sha256'] or fingerprint(selection)!=receipt['selection_fingerprint']:
        raise ValueError('Referentiemodel of selectie wijkt af van de vastgelegde receipt')
    started=time.perf_counter()
    artifact=load_model(training_root/'models/citibike_model.joblib')
    if fingerprint(artifact['selection'])!=receipt['selection_fingerprint']:
        raise ValueError('Referentieartefact bevat een andere modelselectie')
    training,validation,prepared=load_model_data(training_root/'data',training_root/'reports')
    frame=pd.concat([training,validation],ignore_index=True)
    frame=frame[frame.day>=pd.Timestamp(artifact['training_start'])].reset_index(drop=True)
    if len(frame)!=artifact['training_rows'] or not (frame.day<'2026-07-01').all():
        raise ValueError('De vaste hertraining vereist exact de oorspronkelijke train+validatierijen')
    if prepared['dataset_fingerprint']!=artifact['selection']['dataset_fingerprint']:
        raise ValueError('Bronversie gewijzigd: geen reproductie van de oorspronkelijke studie')
    estimator=artifact['model'].estimator_
    if estimator.get_params()['n_estimators']!=320 or estimator.get_params()['random_state']!=42:
        raise ValueError('Onverwachte RF-configuratie')
    expected=predict_features(artifact['model'],validation,FEATURES)
    model_template=clone(artifact['model'])
    candidate=copy.copy(artifact)
    del artifact
    # De referentie wordt later opnieuw geladen voor de serialisatiecontrole.
    candidate.pop('model')
    gc.collect()
    fit_start=time.perf_counter()
    candidate['model']=model_template.fit(frame[FEATURES],frame.rides)
    fit_seconds=time.perf_counter()-fit_start
    actual=predict_features(candidate['model'],validation,FEATURES)
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-9)
    path=output/'retrained_fixed_model.joblib'
    joblib.dump(candidate,path,compress=3)
    del candidate,model_template
    gc.collect()
    reloaded=load_model(path)
    np.testing.assert_allclose(predict_features(reloaded['model'],validation,FEATURES),expected,rtol=1e-12,atol=1e-9)
    after={name:sha256_file(training_root/name) for name in protected_names}
    if before!=after: raise AssertionError('Oorspronkelijke studie gewijzigd')
    result={'completed_at':datetime.now(timezone.utc).isoformat(),'status':'passed',
            'training_root':str(training_root),'rows':len(frame),'features':FEATURES,
            'prepared_data_sha256':prepared['sha256'],'dataset_fingerprint':prepared['dataset_fingerprint'],
            'reference_model_sha256':before['models/citibike_model.joblib'],
            'artifact':str(path),'artifact_sha256':sha256_file(path),
            'fit_seconds':fit_seconds,'total_seconds':time.perf_counter()-started,
            'prediction_max_absolute_difference':float(np.max(np.abs(actual-expected))),
            'reloaded_predictions_verified':True,'original_study_unchanged':True,
            'test_used':False,'new_model_selection':False,'live_model_replaced':False,
            'comparison_is_accuracy_evaluation':False,'code_version':code_version(),
            'pipeline_script_sha256':sha256_file(Path(__file__)),
            'github_run_id':os.environ.get('GITHUB_RUN_ID'),'github_sha':os.environ.get('GITHUB_SHA')}
    atomic_json(output/'run.json',result)
    # Kleine lokale receipt; modelbestanden blijven uitsluitend in de genegeerde data-map.
    if training_root==ROOT:
        atomic_json(ROOT/'reports/retraining_pipeline_latest.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,default=ROOT/'data/pipeline_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    args=parser.parse_args()
    run(args.training_root,args.output)
