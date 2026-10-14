"""Evaluate only the team's provided 5,000-row CSV; keep the old UCI runs historical."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from MushroomFinal.final_model_support import preprocessing, ZeroStemMeasurements, ProximitySporeImputer

FOLDER = ROOT / 'MushroomFinal'
OUT = FOLDER / 'comparison_correct'
SOURCE = FOLDER / 'mushroom_project_dataset.csv'
NAMES = {'rf_final': 'Final Random Forest 300', 'gb_final': 'Final Gradient Boosting',
    'lr_final': 'Final Logistic Regression', 'rf_500_previous': 'Eerdere Andrew RF500',
    'rf_200_api': 'RF200 API-configuratie', 'aws_tuned_local': 'AWS-configuratie getuned (lokale fit)',
    'aws_baseline_local': 'AWS-configuratie baseline (lokale fit)',
    'tree_baseline': 'Beslisboom depth 5', 'dummy': 'Meerderheidsreferentie'}


def digest(path):
    b = path.read_bytes()
    return {'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest(),
        'sha256_lf': hashlib.sha256(b.replace(b'\r\n', b'\n')).hexdigest()}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def load_data():
    raw = pd.read_csv(SOURCE)
    assert raw.shape == (5000, 13) and raw['class'].isin(['e','p']).all()
    raw.columns = raw.columns.str.strip().str.lower().str.replace('-', '_', regex=False)
    X = raw.drop(columns=['class','jumbled_noise_0'])
    y = raw['class'].map({'e':0,'p':1})
    assert X.shape == (5000,11) and 'jumbled_noise_1' in X and not raw.duplicated().any()
    train,test = train_test_split(np.arange(5000),test_size=.18,random_state=42,stratify=y)
    assert len(train)==4100 and len(test)==900
    assert y.iloc[test].value_counts().to_dict()=={0:559,1:341}
    folds = list(StratifiedKFold(5,shuffle=True,random_state=42).split(X.iloc[train],y.iloc[train]))
    return raw,X,y,train,test,folds


def evaluate(y,pred,prob):
    tn,fp,fn,tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'accuracy':float(accuracy_score(y,pred)), 'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
        'precision_p':float(precision_score(y,pred,zero_division=0)),
        'recall_p':float(recall_score(y,pred,zero_division=0)),
        'f1_p':float(f1_score(y,pred,zero_division=0)),
        'roc_auc':float(roc_auc_score(y,prob)) if prob is not None else None,
        'average_precision':float(average_precision_score(y,prob)) if prob is not None else None,
        'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}


def predict(model,X):
    return model.predict(X).astype(int),model.predict_proba(X)[:,list(model.classes_).index(1)]


def macro_f1_metric(y_true,y_pred,**kwargs):
    # PyCaret 3.3.2 does not reliably forward add_metric(average='macro') to its scorer.
    # Keep the averaging inside the callable so the column is genuinely macro F1.
    return f1_score(y_true,y_pred,average='macro',zero_division=0)


def candidates_for(X):
    memory = joblib.Memory(OUT/'cache',verbose=0)
    def pipe(model,aws=False,scale=False,proximity=False,zero=False):
        steps=[]
        if zero: steps.append(('zero_stems',ZeroStemMeasurements()))
        if proximity: steps.append(('proximity',ProximitySporeImputer()))
        return Pipeline([*steps,('preprocessor',preprocessing(X,aws,scale)),('classifier',model)],memory=memory)
    return {
        'rf_final':pipe(RandomForestClassifier(n_estimators=300,max_depth=20,max_features=None,
            min_samples_split=4,class_weight='balanced',random_state=42,n_jobs=2)),
        'gb_final':pipe(GradientBoostingClassifier(n_estimators=200,learning_rate=.1,max_depth=3,
            min_samples_leaf=1,subsample=.8,random_state=42),proximity=True),
        'lr_final':pipe(LogisticRegression(C=10,penalty='l1',solver='liblinear',class_weight='balanced',
            max_iter=2000,random_state=42),proximity=True,scale=True),
        'rf_500_previous':pipe(RandomForestClassifier(n_estimators=500,max_depth=20,min_samples_split=3,
            class_weight='balanced',random_state=42,n_jobs=2),zero=True),
        'rf_200_api':pipe(RandomForestClassifier(n_estimators=200,class_weight='balanced',random_state=42,n_jobs=2),zero=True),
        'aws_tuned_local':pipe(RandomForestClassifier(n_estimators=100,max_depth=24,min_samples_leaf=2,
            random_state=42,n_jobs=2),aws=True),
        'aws_baseline_local':pipe(RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=2),aws=True),
        'tree_baseline':pipe(DecisionTreeClassifier(max_depth=5,random_state=42)),
        'dummy':pipe(DummyClassifier(strategy='most_frequent'))}


def prepare():
    OUT.mkdir(exist_ok=True)
    raw,X,y,train,test,folds=load_data()
    manifest=pd.DataFrame({'source_row':np.arange(5000),'class':raw['class'],'split':'train',
        'training_order':pd.Series([pd.NA]*5000,dtype='Int64'),
        'cv_validation_fold':pd.Series([pd.NA]*5000,dtype='Int64')})
    manifest.loc[test,'split']='test'
    manifest.loc[train,'training_order']=np.arange(4100)
    for fold,(_,val) in enumerate(folds,1): manifest.loc[train[val],'cv_validation_fold']=fold
    manifest.to_csv(OUT/'split_manifest.csv',index=False)
    protocol={'source':'MushroomFinal/mushroom_project_dataset.csv','dataset':digest(SOURCE),
        'raw_rows':5000,'features':X.columns.tolist(),'removed':['jumbled_noise_0'],
        'class_encoding':{'e':0,'p':1},'train_rows':4100,'test_rows':900,'test_size':.18,
        'random_state':42,'stratified':True,'folds':5,'fold_shuffle':True,
        'primary_metric':'mean CV macro F1','tie_breakers':['mean CV recall p','fixed candidate order'],
        'test_previously_used':True,'species_grouping':'Unavailable; source row is not a species ID',
        'aws_status':('Corrected SageMaker exports received and artifact audited; original comparison rows remain local refits'
            if (OUT/'aws_artifact_audit.json').exists() else 'Local refit of prior AWS settings; new SageMaker run still required'),
        'old_61069_row_results':'Not valid for the final course comparison'}
    write_json(OUT/'protocol.json',protocol)
    return raw,X,y,train,test,folds,protocol


def manual_run():
    assert sklearn.__version__=='1.4.2', 'Use Python 3.11 and MushroomFinal/requirements-comparison.txt'
    raw,X,y,train,test,folds,protocol=prepare()
    candidates=candidates_for(X)
    fold_records=[]; summary=[]; warning_log={}
    oof=pd.DataFrame({'source_row':train,'true_class':y.iloc[train].to_numpy()})
    test_predictions=pd.DataFrame({'source_row':test,'true_class':y.iloc[test].to_numpy()})
    models={}
    for key,prototype in candidates.items():
        labels=np.zeros(len(train),dtype=int); prob=np.zeros(len(train))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            for fold,(tr,val) in enumerate(folds,1):
                print(key,'fold',fold,'/5',flush=True)
                model=clone(prototype);start=perf_counter()
                model.fit(X.iloc[train[tr]],y.iloc[train[tr]])
                elapsed=perf_counter()-start
                labels[val],prob[val]=predict(model,X.iloc[train[val]])
                fold_records.append({'model':key,'fold':fold,'fit_seconds':elapsed,
                    **evaluate(y.iloc[train[val]],labels[val],prob[val])})
            models[key]=clone(prototype).set_params(memory=None).fit(X.iloc[train],y.iloc[train])
        warning_log[key]=[str(w.message) for w in caught]
        oof[key+'_prediction']=labels; oof[key+'_probability_p']=prob
        rows=[r for r in fold_records if r['model']==key]
        summary.append({'model':key,'cv_macro_f1_mean':float(np.mean([r['macro_f1'] for r in rows])),
            'cv_macro_f1_std':float(np.std([r['macro_f1'] for r in rows],ddof=1)),
            'cv_recall_p_mean':float(np.mean([r['recall_p'] for r in rows])),
            'cv_balanced_accuracy_mean':float(np.mean([r['balanced_accuracy'] for r in rows])),
            'mean_fold_fit_seconds':float(np.mean([r['fit_seconds'] for r in rows]))})
    ranking=pd.DataFrame(summary).query("model != 'dummy'").sort_values(
        ['cv_macro_f1_mean','cv_recall_p_mean'],ascending=False,kind='stable')
    selected=str(ranking.iloc[0].model)
    write_json(OUT/'cv_selection.json',{'selected_model':selected,'ranking':ranking.to_dict('records'),
        'selection_made_before_test_predictions':True})
    for row in summary:
        key=row['model']; predicted,prob=predict(models[key],X.iloc[test])
        row.update(evaluate(y.iloc[test],predicted,prob))
        test_predictions[key+'_prediction']=predicted;test_predictions[key+'_probability_p']=prob
        train_pred,train_prob=predict(models[key],X.iloc[train])
        row['train_accuracy']=evaluate(y.iloc[train],train_pred,train_prob)['accuracy']
    pd.DataFrame(summary).to_csv(OUT/'model_comparison.csv',index=False)
    pd.DataFrame(fold_records).to_csv(OUT/'cv_folds.csv',index=False)
    oof.to_csv(OUT/'oof_predictions.csv',index=False)
    test_predictions.to_csv(OUT/'test_predictions.csv',index=False)
    selected_model=models[selected]
    joblib.dump(selected_model,OUT/'selected_pipeline.joblib',compress=3)
    wrong=test_predictions[selected+'_prediction']!=test_predictions.true_class
    errors=X.iloc[test[wrong.to_numpy()]].copy()
    errors.insert(0,'source_row',test[wrong.to_numpy()])
    errors['true_class']=test_predictions.loc[wrong,'true_class'].to_numpy()
    errors['prediction']=test_predictions.loc[wrong,selected+'_prediction'].to_numpy()
    errors['probability_p']=test_predictions.loc[wrong,selected+'_probability_p'].to_numpy()
    errors.to_csv(OUT/'selected_errors.csv',index=False)
    write_json(OUT/'comparison_summary.json',{'protocol':protocol,'selected_model':selected,
        'candidates':list(candidates),'model_comparison':summary,'warnings':warning_log,
        'environment':{'python':platform.python_version(),'sklearn':sklearn.__version__,'numpy':np.__version__,
            'pandas':pd.__version__,'joblib':joblib.__version__},
        'classifier_parameters':{k:v.named_steps['classifier'].get_params() for k,v in candidates.items()},
        'source_notebook_commit':'0b34861','artifact':digest(OUT/'selected_pipeline.joblib'),
        'artifact_training_rows':4100,'artifact_class_encoding':{'e':0,'p':1}})
    print(pd.DataFrame(summary)[['model','cv_macro_f1_mean','accuracy','macro_f1','recall_p','fn']],flush=True)


def pycaret_run():
    from pycaret.classification import ClassificationExperiment
    import pycaret
    raw,X,y,train,test,folds,protocol=prepare()
    experiment=ClassificationExperiment()
    training=X.iloc[train].assign(target=y.iloc[train])
    testing=X.iloc[test].assign(target=y.iloc[test])
    experiment.setup(data=training,test_data=testing,target='target',index=True,
        session_id=42,fold_strategy=StratifiedKFold(5,shuffle=True,random_state=42),
        numeric_imputation='median',categorical_imputation='mode',max_encoding_ohe=25,
        n_jobs=2,html=False,verbose=False,system_log=False,log_experiment=False)
    assert experiment.get_config('X_train').index.tolist()==train.tolist()
    assert experiment.get_config('X_test').index.tolist()==test.tolist()
    actual_folds=list(experiment.get_config('fold_generator').split(experiment.get_config('X_train'),experiment.get_config('y_train')))
    assert all(np.array_equal(a,c) and np.array_equal(b,d) for (a,b),(c,d) in zip(folds,actual_folds))
    assert abs(macro_f1_metric([0,0,1],[0,0,0])-.4)<1e-12
    experiment.add_metric(id='macro_f1',name='Macro F1',score_func=macro_f1_metric,target='pred')
    print('PyCaret: LR, RF, Gradient Boosting on the exact shared split',flush=True)
    three=experiment.compare_models(include=['lr','rf','gbc'],sort='Macro F1',n_select=3,round=10,errors='raise',verbose=False)
    comparison=experiment.pull();comparison.to_csv(OUT/'pycaret_three.csv')
    print(comparison.to_string(),flush=True)
    print('PyCaret: all available fast model families',flush=True)
    best=experiment.compare_models(sort='Macro F1',n_select=5,round=10,errors='ignore',verbose=False)
    all_results=experiment.pull();all_results.to_csv(OUT/'pycaret_all.csv')
    assert float(all_results.loc['dummy','Macro F1'])>.3
    returned=best if isinstance(best,list) else [best]
    experiment.save_model(returned[0],str(OUT/'pycaret_best_pipeline'),verbose=False)
    test_rows=[]
    for i,model in enumerate(returned):
        prediction=experiment.predict_model(model,data=testing,raw_score=True,round=10,verbose=False)
        prediction.to_csv(OUT/f'pycaret_holdout_{i}.csv')
        prob_column=next((c for c in prediction if c in ['prediction_score_1','prediction_score_1.0']),None)
        test_rows.append({'rank':i+1,'model':type(model).__name__,
            **evaluate(y.iloc[test],prediction.prediction_label.astype(int),prediction[prob_column] if prob_column else None)})
    pd.DataFrame(test_rows).to_csv(OUT/'pycaret_test_metrics.csv',index=False)
    write_json(OUT/'pycaret_audit.json',{'version':pycaret.__version__,'train_rows':len(train),'test_rows':len(test),
        'train_source_rows':train.tolist(),'test_source_rows':test.tolist(),'folds':5,'random_state':42,
        'class_encoding':{'e':0,'p':1},'primary_metric':'Macro F1','default_models_not_gridsearch_tuned':True,
        'dataset':digest(SOURCE),'all_model_rows':int(len(all_results)),
        'shared_fold_membership_verified':True,
        'available_turbo_model_ids':experiment.models().query('Turbo').index.tolist(),
        'not_returned_by_compare_models':[k for k in experiment.models().query('Turbo').index if k not in all_results.index],
        'best_pipeline':digest(OUT/'pycaret_best_pipeline.pkl'),
        'macro_metric_validation':'Custom callable fixes macro averaging; majority reference must score >0.3'})
    print(all_results.to_string(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['manual','pycaret','all'],default='all')
    args=parser.parse_args()
    if args.stage in ['manual','all']: manual_run()
    if args.stage in ['pycaret','all']: pycaret_run()
