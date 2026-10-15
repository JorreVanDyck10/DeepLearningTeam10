"""Build the stronger voting challenger only if the prespecified goal is met."""
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode=True
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score
from improve_v3 import experiments, WEIGHTS, f1_threshold, threshold_at_fpr, OUT
from mushroom_alternative import HERE, FEATURES, development_data, write_json

DEST=HERE/'voting_artifacts_v3'


def digest(path):
    raw=path.read_bytes()
    if path.suffix in ['.json','.txt','.py']:
        raw=raw.replace(b'\r\n',b'\n')
    return hashlib.sha256(raw).hexdigest()


def export_rf(model,directory):
    directory.mkdir(parents=True,exist_ok=True)
    pre=model.named_steps['preprocessor']; clf=model.named_steps['classifier']
    trees=[t.tree_ for t in clf.estimators_]
    file=directory/'mushroom_rf300.npz'
    np.savez_compressed(file,left=np.concatenate([t.children_left for t in trees]).astype(np.int32),
        right=np.concatenate([t.children_right for t in trees]).astype(np.int32),
        feature=np.concatenate([t.feature for t in trees]).astype(np.int16),threshold=np.concatenate([t.threshold for t in trees]),
        probabilities=np.concatenate([t.value[:,0,:] for t in trees]),offsets=np.r_[0,np.cumsum([t.node_count for t in trees])])
    meta=json.loads((HERE/'artifacts/mushroom_rf300.json').read_text(encoding='utf-8'))
    allowed={s['name']:s for s in meta['categorical']}
    columns={n:c for n,_,c in pre.transformers_ if n!='remainder'}
    cat=pre.named_transformers_['cat']
    specs=[{'name':n,'imputation':str(mode),'categories':[str(v) for v in codes]}
        for n,mode,codes in zip(columns['cat'],cat.named_steps['impute'].statistics_,cat.named_steps['encode'].categories_)]
    ignored=[f for f in FEATURES if f not in model.named_steps['inputs'].columns_]
    specs.extend(allowed[f] for f in ignored)
    meta.update(forest_sha256=digest(file),forest_file=file.name,categorical=specs,
        numeric_features=list(columns['num']),numeric_medians=pre.named_transformers_['num'].statistics_.tolist(),
        ignored_features=ignored,metrics={},model_name='Internal RF component of Milan voting',
        source_pipeline=file.name,source_pipeline_sha256=digest(file),
        proposed_decision_threshold=.5,proposed_policy='default',proposed_policy_metrics={},promotion_recommended=False,
        validation_note='Internal fulltrain voting component. No separate validation claim; evaluate the complete voting procedure.')
    path=directory/'mushroom_rf300.json'; write_json(path,meta)
    return path


def run():
    assert json.loads((OUT/'run.json').read_text())['complete']
    table=pd.read_csv(OUT/'summary.csv').set_index('model')
    reference=table.loc['rf_compatible_f1_threshold']
    candidates=table.loc[['voting_default','voting_f1','voting_fpr_budget']]
    eligible=candidates[(candidates.macro_f1_mean>reference.macro_f1_mean)
        &(candidates.fn_total<reference.fn_total)&(candidates.fp_total<=reference.fp_total)]
    if not len(eligible):
        write_json(OUT/'voting_package.json',{'packaged':False,'goal_met':False,
            'reason':'No voting procedure meets all prespecified goals against v2.','deployment_performed':False})
        print('No qualifying voting challenger.')
        return
    selected=eligible.macro_f1_mean.idxmax(); policy=selected.removeprefix('voting_')
    X,y,ids,_=development_data()
    provenance={'protocol':digest(OUT/'protocol.json'),'code':digest(Path(__file__))}
    cv=StratifiedKFold(5,shuffle=True,random_state=42)
    models={}; inside=[]; parameters={}
    for name,(pipeline,grid) in experiments().items():
        print('Fulltrain voting component:',name,flush=True)
        cache=HERE/'.generated/v3'/('fulltrain_'+name+'.joblib')
        cache.parent.mkdir(parents=True,exist_ok=True)
        scorefile=cache.with_suffix('.npz')
        statefile=cache.with_suffix('.json')
        if cache.exists() and scorefile.exists() and statefile.exists():
            state=json.loads(statefile.read_text())
            assert state['provenance']==provenance,'Fulltrain cache changed: remove only this study cache before rerunning.'
            model=joblib.load(cache)
            with np.load(scorefile) as data:
                p=data['probability']
            parameters[name]=state['parameters']
        else:
            search=GridSearchCV(pipeline,grid,scoring='f1_macro',cv=cv,n_jobs=2,pre_dispatch=2,error_score='raise')
            search.fit(X,y); model=search.best_estimator_
            pd.DataFrame(search.cv_results_).to_csv(OUT/('final_voting_'+name+'_search.csv'),index=False)
            parameters[name]=search.best_params_
            p=cross_val_predict(clone(model),X,y,cv=cv,n_jobs=2,pre_dispatch=2,method='predict_proba')[:,1]
            joblib.dump(model,cache,compress=3); np.savez_compressed(scorefile,probability=p)
            write_json(statefile,{'provenance':provenance,'parameters':search.best_params_})
        models[name]=model; inside.append(p)
    matrix=np.column_stack(inside)
    options=[]
    for weights in WEIGHTS:
        p=matrix@np.asarray(weights); threshold=f1_threshold(y,p)
        options.append((float(f1_score(y,p>=threshold,average='macro',zero_division=0)),weights,threshold))
    _,weights,tf=max(enumerate(options),key=lambda item:(item[1][0],-item[0]))[1]
    threshold=.5 if policy=='default' else (tf if policy=='f1' else threshold_at_fpr(y,matrix@np.asarray(weights)))
    DEST.mkdir(parents=True,exist_ok=True)
    components=[]; expected=np.zeros(len(X))
    for (name,model),weight in zip(models.items(),weights):
        if not weight:
            continue
        selector=model.named_steps['inputs']
        if name=='rf_selection':
            path=export_rf(model,DEST/'rf')
        elif name=='lightgbm':
            path=DEST/'lightgbm.txt'; model.named_steps['classifier'].booster_.save_model(str(path))
        else:
            path=DEST/'catboost.cbm'; model.named_steps['classifier'].save_model(str(path))
        components.append({'family':name,'weight':weight,'file':path.relative_to(DEST).as_posix(),
            'sha256':digest(path),'columns':selector.columns_,'categories':selector.categories_})
        expected+=weight*model.predict_proba(X)[:,1]
    metadata=json.loads((HERE/'artifacts/mushroom_rf300.json').read_text(encoding='utf-8'))
    for key in ['forest_file','forest_sha256','numeric_medians','source_pipeline','n_estimators']:
        metadata.pop(key,None)
    raw=table.loc['voting_default']; chosen=table.loc[selected]
    recipe={'components':components,'proposed_decision_threshold':threshold,'proposed_policy':policy}
    revision=hashlib.sha256(json.dumps(recipe,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    metadata.update(**recipe,source_pipeline_sha256=revision,model_name='Milan — gewogen voting v3',
        serving_format='milan_native_voting_v1',
        metrics={'accuracy':raw.accuracy_mean,'macro_f1':raw.macro_f1_mean,'recall_poisonous':raw.recall_p_mean,
            'fn':int(raw.fn_total),'fp':int(raw.fp_total),'metrics_scope':'outer CV, 4100 development rows, raw voting'},
        proposed_policy_metrics=chosen.to_dict(),reported_metrics_policy='default',
        validation_note='Derde ontwikkelstudie op eerder gebruikte 4100 trainingsrijen; geen onafhankelijke eindtest.',
        concept_only=True,promotion_recommended=False,
        policy_note='Stronger exploratory voting challenger; new model reader and boosting dependencies required. RF v2 remains the minimal-integration alternative.')
    manifest=DEST/'voting.json'; write_json(manifest,metadata)
    from voting_model_v3 import VotingMushroom
    portable=VotingMushroom(manifest)
    actual=portable.predict_proba(X)[:,1]
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
    np.testing.assert_array_equal(portable.predict(X),expected>.5)
    edge=X.iloc[:10].copy(); edge.iloc[0,:]=np.nan
    for f in FEATURES[3:]:
        edge.loc[1,f]='never_seen'
    edge_expected=sum(weight*model.predict_proba(edge)[:,1] for (name,model),weight in zip(models.items(),weights) if weight)
    np.testing.assert_allclose(portable.predict_proba(edge)[:,1],edge_expected,rtol=0,atol=1e-12)
    portable.predict_record(metadata['example'])
    pd.DataFrame({'source_row':ids,'probability_p':expected,'raw_prediction':(expected>.5).astype(int)}).to_csv(DEST/'training_input_export_references.csv',index=False)
    write_json(OUT/'voting_package.json',{'packaged':True,'goal_met':True,'policy':policy,
        'fulltrain_weights':weights,'fulltrain_threshold':threshold,'parameters':parameters,
        'revision':revision,'export_rows_checked':len(X),'export_max_probability_difference':float(np.max(np.abs(actual-expected))),
        'null_and_unknown_category_export_check':True,'custom_pickle_classes':False,
        'new_serving_dependencies':any(s['family']!='rf_selection' for s in components),
        'artifact_bytes':sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file()),
        'validation_scope':'Outer CV evaluates the procedure; fulltrain export equivalence is not independent validation.',
        'deployment_performed':False,'development_test_rows_scored':0})
    print('Native voting challenger packaged and checked; production unchanged.',flush=True)


if __name__=='__main__':
    with joblib.parallel_config(backend='threading'):
        run()
