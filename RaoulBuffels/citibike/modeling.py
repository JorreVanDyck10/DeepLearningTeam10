"""Chronologische vergelijking, begrensde tuning en eenmalige eindtest."""
import json
import time
import importlib.metadata
import inspect
import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterSampler
from sklearn.inspection import permutation_importance

from .config import (ROOT,REPORTS,MODELS,DATA,TRAIN_END,TEST_START,SEED,atomic_json,
                     code_version,fingerprint,now,sha256_file)
from .datasets import historical_hourly,hourly,manifest
from .features import FEATURES,CALENDAR,make_features,folds,SavedChronologicalCV,leakage_audit
from .inference import predict_features,predict_day,load_model

SEARCH_BUDGET={'random_forest':6,'lightgbm':20,'ridge':20}


class SeasonalNaive(RegressorMixin,BaseEstimator):
    def fit(self,X,y=None):
        self.n_features_in_=X.shape[1]
        return self
    def predict(self,X):
        return X['lag_7d'].to_numpy(float)


class OperationalRegressor(RegressorMixin,BaseEstimator):
    """Clipping en DST ook in sklearn/PyCaret-CV, identiek aan deployment."""
    def __init__(self,estimator,columns=None):
        self.estimator=estimator
        self.columns=columns
    def fit(self,X,y):
        self.columns_=self.columns or FEATURES
        self.estimator_=clone(self.estimator).fit(X[self.columns_],y)
        self.n_features_in_=X.shape[1]
        return self
    def predict(self,X):
        prediction=np.maximum(self.estimator_.predict(X[self.columns_]),0)
        return np.where(X.clock_hours.to_numpy()==0,0,prediction)


def prediction_code_fingerprint():
    """Vergrendel featureberekening, data-afscherming en operationele predictregels."""
    version=code_version([ROOT/'citibike'/name for name in ['features.py','inference.py','datasets.py','config.py']])
    return fingerprint({'files':version['files'],'packages':version['packages'],
                        'estimators':inspect.getsource(OperationalRegressor)+inspect.getsource(SeasonalNaive)})


def metrics(actual,predicted):
    return {'RMSE':float(np.sqrt(mean_squared_error(actual,predicted))),
            'MAE':float(mean_absolute_error(actual,predicted)),'R2':float(r2_score(actual,predicted))}


def candidates():
    return {
      'seasonal_naive':OperationalRegressor(SeasonalNaive()),
      'ridge':OperationalRegressor(make_pipeline(StandardScaler(),Ridge(alpha=10))),
      'random_forest':OperationalRegressor(RandomForestRegressor(n_estimators=160,max_depth=16,
                        min_samples_leaf=8,max_features=.8,n_jobs=2,random_state=SEED)),
      'lightgbm':OperationalRegressor(LGBMRegressor(n_estimators=250,num_leaves=31,
                       learning_rate=.05,min_child_samples=40,n_jobs=2,verbosity=-1,random_state=SEED,
                       deterministic=True,force_col_wise=True)),
    }


def cv_scores(model,frame,bounds):
    results=[]
    for i,(train_indices,validation_indices) in enumerate(SavedChronologicalCV(bounds).split(frame),1):
        start=time.perf_counter();training=frame.iloc[train_indices];checking=frame.iloc[validation_indices]
        fitted=clone(model).fit(training[FEATURES],training.rides)
        results.append({'fold':i,**metrics(checking.rides,fitted.predict(checking[FEATURES])),
                        'seconds':time.perf_counter()-start})
    return results


def run_experiment(name,model,train,validation,bounds):
    start=time.perf_counter()
    scores=cv_scores(model,train,bounds)
    fitted=clone(model).fit(train[FEATURES],train.rides)
    train_scores=metrics(train.rides,fitted.predict(train[FEATURES]))
    predictions=fitted.predict(validation[FEATURES]);valid_scores=metrics(validation.rides,predictions)
    MODELS.mkdir(parents=True,exist_ok=True)
    joblib.dump(fitted,MODELS/f'{name}.joblib')
    row={'name':name,**valid_scores,**{f'train_{k}':v for k,v in train_scores.items()},
         'cv_RMSE':float(np.mean([r['RMSE'] for r in scores])),
         'fit_cv_seconds':time.perf_counter()-start,'cv':scores,'parameters':model.get_params(deep=True)}
    print(f"{name}: validatie-RMSE {row['RMSE']:.2f}",flush=True)
    return row


def pycaret_compare(train,validation,bounds):
    from pycaret.regression import RegressionExperiment
    experiment=RegressionExperiment()
    experiment.setup(data=train[FEATURES+['rides']],test_data=validation[FEATURES+['rides']],target='rides',
                     fold_strategy=SavedChronologicalCV(bounds),fold=len(bounds),
                     data_split_shuffle=False,fold_shuffle=False,preprocess=False,session_id=SEED,
                     n_jobs=2,html=False,verbose=False,system_log=False,log_experiment=False,
                     index=False)
    # Controleer de daadwerkelijke interne rijvolgorde, niet alleen setup-argumenten.
    for label,expected in [('train',train),('test',validation)]:
        actual_X=experiment.get_config('X_'+label)
        actual_y=experiment.get_config('y_'+label)
        assert list(actual_X.columns)==FEATURES and len(actual_X)==len(expected)
        np.testing.assert_array_equal(actual_X.to_numpy(dtype=np.float32),
                                      expected[FEATURES].to_numpy(dtype=np.float32))
        np.testing.assert_array_equal(np.asarray(actual_y,dtype=np.float32),
                                      expected.rides.to_numpy(dtype=np.float32))
    shortlist=candidates()
    ranked=experiment.compare_models(include=list(shortlist.values()),sort='RMSE',n_select=4,
                                     errors='raise',verbose=False)
    table=experiment.pull()
    # De wrappers hebben dezelfde klassenaam; benoem het werkelijk teruggegeven binnenmodel.
    labels=[next(name for name,candidate in shortlist.items()
                 if type(candidate.estimator) is type(fitted.estimator)) for fitted in ranked]
    assert len(labels)==len(table)==len(shortlist)
    table.insert(0,'candidate',labels)
    table.to_csv(REPORTS/'pycaret_comparison.csv',index=True)
    atomic_json(REPORTS/'pycaret_split_audit.json',{
        'train_rows':len(train),'validation_rows':len(validation),'folds':bounds,
        'fold_fingerprint':fingerprint(bounds),'train_data_fingerprint':fingerprint(train.timestamp.astype(str).tolist()),
        'validation_data_fingerprint':fingerprint(validation.timestamp.astype(str).tolist()),
        'shuffle':False,'preprocess':False,'preprocessing':'Ridge scaler inside per-fold estimator pipeline',
        'actual_internal_rows_checked':True,'candidate_labels':labels,
        'operational_prediction_rules':'same OperationalRegressor as sklearn and saved inference'})
    return table


def tune(name,model,train,bounds):
    if name=='random_forest':
        space={'estimator__n_estimators':[160,240,320], 'estimator__max_depth':[10,16,22,None],
               'estimator__min_samples_leaf':[4,8,16], 'estimator__max_features':[.6,.8,1.]}
    elif name=='lightgbm':
        space={'estimator__n_estimators':[200,400,650], 'estimator__num_leaves':[15,31,63],
               'estimator__learning_rate':[.025,.05,.1], 'estimator__min_child_samples':[20,50,100]}
    elif name=='ridge':
        space={'estimator__ridge__alpha':np.logspace(-2,5,20).tolist()}
    else:
        return model,[]
    rows=[];best_score=np.inf;best=model
    budget=SEARCH_BUDGET[name]
    for number,params in enumerate(ParameterSampler(space,n_iter=budget,random_state=SEED),1):
        candidate=clone(model).set_params(**params);scores=cv_scores(candidate,train,bounds)
        score=float(np.mean([r['RMSE'] for r in scores]))
        rows.append({'configuration':number,'cv_RMSE':score,'parameters':params,'folds':scores})
        atomic_json(REPORTS/f'{name}_tuning.json',rows)
        print(f'Tuning {name} {number}/{budget}: CV-RMSE {score:.2f}',flush=True)
        if score<best_score:
            best_score=score;best=candidate
    pd.DataFrame([{**r,'parameters':json.dumps(r['parameters']), 'folds':json.dumps(r['folds'])} for r in rows]).to_csv(REPORTS/f'{name}_tuning.csv',index=False)
    return best,rows


def choose_model(rows):
    minimum=min(r['RMSE'] for r in rows)
    simplicity={'seasonal_naive':0,'ridge':1,'ridge_calendar':1,'ridge_no_14d':1,
                'random_forest':2,'lightgbm':3}
    eligible=[r for r in rows if r['RMSE']==minimum or r['RMSE']<minimum*1.01]
    return min(eligible,key=lambda r:(simplicity.get(r['name'].replace('tuned_',''),4),r['RMSE']))


def residual_table(frame,predictions):
    result=frame[['timestamp','rides']].copy();result['predicted_rides']=predictions
    result['residual']=result.rides-result.predicted_rides
    result['absolute_error']=result.residual.abs();result['hour']=result.timestamp.dt.hour
    result['weekday']=result.timestamp.dt.dayofweek;result['month']=result.timestamp.dt.month
    result['season']=result['month'].map({12:'winter',1:'winter',2:'winter',3:'lente',4:'lente',5:'lente',6:'zomer',7:'zomer',8:'zomer',9:'herfst',10:'herfst',11:'herfst'})
    return result


def train():
    if (REPORTS/'final_evaluation.json').exists() or (REPORTS/'evaluation_lock.json').exists():
        raise RuntimeError('Eindtest is al geopend; nieuwe selectie vereist een nieuwe onafhankelijke eindtest')
    decision=json.loads((REPORTS/'target_decision.json').read_text(encoding='utf-8'))
    if decision['target']!='hourly_demand' or decision['dataset_fingerprint']!=manifest()['dataset_fingerprint']:
        raise ValueError('Doelkeuze ontbreekt of brondata gewijzigd')
    start=time.perf_counter()
    atomic_json(REPORTS/'leakage_audit.json',leakage_audit())
    full=make_features(historical_hourly(include_validation=True))
    full.to_parquet(DATA/'modeling_development_validation.parquet',index=False)
    validation=full[(full.day>=TRAIN_END)&(full.day<TEST_START)].reset_index(drop=True)
    training=full[full.day<TRAIN_END].reset_index(drop=True)
    window_rows=[]
    for years in [2,5,None]:
        subset=training if years is None else training[training.day>=pd.Timestamp(TRAIN_END)-pd.DateOffset(years=years)]
        subset=subset.reset_index(drop=True);bounds=folds(subset)
        for name in ['ridge','lightgbm']:
            scores=cv_scores(candidates()[name],subset,bounds)
            row={'years':years,'model':name,'rows':len(subset), 'cv_RMSE':float(np.mean([r['RMSE'] for r in scores])),'folds':scores}
            window_rows.append(row);print(f"Venster {years or 'alle'} jaren / {name}: {row['cv_RMSE']:.2f}",flush=True)
    atomic_json(REPORTS/'training_windows.json',window_rows)
    window=min(window_rows,key=lambda r:r['cv_RMSE'])['years']
    train_frame=training if window is None else training[training.day>=pd.Timestamp(TRAIN_END)-pd.DateOffset(years=window)]
    train_frame=train_frame.reset_index(drop=True);bounds=folds(train_frame)
    atomic_json(REPORTS/'chronological_splits.json',{
        'window_years':window,'train_start':str(train_frame.day.min()),'train_end_exclusive':TRAIN_END,
        'validation_start':TRAIN_END,'validation_end_exclusive':TEST_START,
        'test_start':TEST_START,'test_end_exclusive':'2026-09-01','train_rows':len(train_frame),
        'validation_rows':len(validation),'folds':bounds,'fold_fingerprint':fingerprint(bounds)})
    rows=[]
    for name,model in candidates().items():
        rows.append(run_experiment(name,model,train_frame,validation,bounds))
        atomic_json(REPORTS/'model_comparison.json',rows)
    pycaret_compare(train_frame,validation,bounds)
    preliminary=choose_model(rows)
    preliminary_model=joblib.load(MODELS/f"{preliminary['name']}.joblib")
    preliminary_errors=residual_table(validation,preliminary_model.predict(validation[FEATURES]))
    preliminary_errors.to_csv(REPORTS/'preliminary_validation_residuals.csv',index=False)
    hour_errors=preliminary_errors.groupby('hour').absolute_error.mean()
    weekday_errors=preliminary_errors.groupby('weekday').absolute_error.mean()
    atomic_json(REPORTS/'validation_error_investigation.json',{
        'model_before_improvement':preliminary['name'],'mean_bias':float(preliminary_errors.residual.mean()),
        'worst_hour':int(hour_errors.idxmax()),'worst_hour_MAE':float(hour_errors.max()),
        'worst_weekday':int(weekday_errors.idxmax()),'worst_weekday_MAE':float(weekday_errors.max()),
        'hour_MAE':hour_errors.to_dict(),'weekday_MAE':weekday_errors.to_dict(),
        'largest_errors':preliminary_errors.nlargest(10,'absolute_error').astype(str).to_dict('records'),
        'improvement_experiments':'Tune model capacity; compare calendar-only with historical features; test removal of less recent 14-day lag. Compare improvement only on validation.',
        'test_used':False})
    promising=sorted([r for r in rows if r['name']!='seasonal_naive'],key=lambda r:r['cv_RMSE'])[:2]
    atomic_json(REPORTS/'tuning_protocol.json',{
        'created_at':now(),'models':[r['name'] for r in promising],
        'configurations':{r['name']:SEARCH_BUDGET[r['name']] for r in promising},
        'seed':SEED,'fold_fingerprint':fingerprint(bounds),'test_used':False,
        'reason':'Resource budget fixed before tuning: Random Forest comparison measured about five minutes; LightGBM about seven seconds. Six seeded RF configurations, up to twenty for other candidates. Same data and folds; no search budget chosen using test results.'})
    for row in promising:
        name=row['name'];tuned,_=tune(name,candidates()[name],train_frame,bounds)
        rows.append(run_experiment('tuned_'+name,tuned,train_frame,validation,bounds))
    # Feature-ablaties, uitsluitend validatie en dezelfde chronologische folds.
    for label,columns in [('ridge_calendar',CALENDAR),('ridge_no_14d',[c for c in FEATURES if c!='lag_14d'])]:
        model=OperationalRegressor(make_pipeline(StandardScaler(),Ridge(alpha=10)),columns=columns)
        rows.append(run_experiment(label,model,train_frame,validation,bounds))
    atomic_json(REPORTS/'model_comparison.json',rows)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ('parameters','cv')} for r in rows]).to_csv(REPORTS/'model_comparison.csv',index=False)
    winner=choose_model(rows)
    fitted=joblib.load(MODELS/f"{winner['name']}.joblib")
    residuals=residual_table(validation,fitted.predict(validation[FEATURES]))
    residuals.to_csv(REPORTS/'validation_residuals.csv',index=False)
    for column in ['hour','weekday','season']:
        residuals.groupby(column).agg(MAE=('absolute_error','mean'),bias=('residual','mean'),n=('rides','size')).to_csv(REPORTS/f'validation_errors_by_{column}.csv')
    importance=permutation_importance(fitted,validation[FEATURES],validation.rides,
                  scoring='neg_root_mean_squared_error',n_repeats=5,random_state=SEED,n_jobs=1)
    pd.DataFrame({'feature':FEATURES,'importance_RMSE':importance.importances_mean,
                  'sd':importance.importances_std}).sort_values('importance_RMSE',ascending=False).to_csv(REPORTS/'permutation_importance.csv',index=False)
    baseline=next(r for r in rows if r['name']=='seasonal_naive')
    selection={'created_at':now(),'name':winner['name'],'validation_metrics':{k:winner[k] for k in ['RMSE','MAE','R2']},
               'baseline_validation_metrics':{k:baseline[k] for k in ['RMSE','MAE','R2']},
               'validation_RMSE_improvement_percent':100*(baseline['RMSE']-winner['RMSE'])/baseline['RMSE'],
               'parameters':fitted.get_params(deep=True),'window_years':window,'features':FEATURES,
               'contract':decision['contract'],'dataset_fingerprint':manifest()['dataset_fingerprint'],
               'modeling_code_version':code_version(),'fold_fingerprint':fingerprint(bounds),
               'prediction_code_fingerprint':prediction_code_fingerprint(),
               'selection_rule':'lowest validation RMSE; strictly less than 1% difference choose simpler model',
               'training_seconds':time.perf_counter()-start,'test_seen':False}
    atomic_json(REPORTS/'model_selection.json',selection)
    # Fit definitieve kandidaat op train+validatie, met venster verschoven tot teststart.
    refit=full if window is None else full[full.day>=pd.Timestamp(TEST_START)-pd.DateOffset(years=window)]
    final=clone(fitted).fit(refit[FEATURES],refit.rides)
    artifact={'model':final,'features':FEATURES,'area':'NYC','target':'hourly_demand','contract':decision['contract'],
              'selection':selection,'training_rows':len(refit),'trained_through':'2026-06-30',
              'training_start':str(refit.day.min()),
              'history_days':14,'schema_version':'1.0'}
    artifact['input_contract']={
        'prediction_day':'offsetloze kalenderdatum; voorspelling om 00:00 America/New_York',
        'history_columns':{'timestamp':'offsetloos lokaal uur, uniek en aaneengesloten',
                           'rides':'eindig niet-negatief aantal geregistreerde vertrekken',
                           'area':'uitsluitend NYC'},
        'history_requirement':'minimaal 14 volledige aansluitende dagen strikt vóór de voorspeldag',
        'DST':'24 kloklabels; ontbrekend uur nul, herhaald uur samengevoegd',
        'output_columns':{'timestamp':'lokaal kloklabel','clock_hours':'0, 1 of 2 fysieke uren',
                          'predicted_rides':'niet-negatieve verwachting, ritten per klokvak'},
        'training_at_inference':False}
    atomic_json(REPORTS/'inference_contract.json',artifact['input_contract'])
    joblib.dump(artifact,MODELS/'citibike_model.joblib')
    atomic_json(REPORTS/'model_artifact.json',{'path':str(MODELS/'citibike_model.joblib'),
              'sha256':sha256_file(MODELS/'citibike_model.joblib'),'selection_fingerprint':fingerprint(selection)})
    print('Modelkeuze en artefact vastgelegd; eindtest nog gesloten.',flush=True)
    return selection


def evaluate():
    output=REPORTS/'final_evaluation.json'
    if output.exists():
        previous=json.loads(output.read_text(encoding='utf-8'))
        artifact=load_model()
        if (previous['model_sha256']!=sha256_file(MODELS/'citibike_model.joblib') or
            previous['selection_fingerprint']!=fingerprint(artifact['selection']) or
            artifact['selection']['dataset_fingerprint']!=manifest()['dataset_fingerprint']):
            raise ValueError('Bestaande eindtest hoort bij een ander model, andere selectie of dataset')
        print('Bestaande eindtestuitvoer geladen; geen nieuwe evaluatie.',flush=True)
        return previous
    selection=json.loads((REPORTS/'model_selection.json').read_text(encoding='utf-8'))
    if selection['prediction_code_fingerprint']!=prediction_code_fingerprint():
        raise ValueError('Feature-, inference-, datatoegangs- of predictioncode gewijzigd na modelvergrendeling')
    receipt=json.loads((REPORTS/'model_artifact.json').read_text(encoding='utf-8'))
    if receipt['sha256']!=sha256_file(MODELS/'citibike_model.joblib') or receipt['selection_fingerprint']!=fingerprint(selection):
        raise ValueError('Model of selectie gewijzigd vóór eindtest')
    if selection['dataset_fingerprint']!=manifest()['dataset_fingerprint']:
        raise ValueError('Dataset gewijzigd vóór eindtest')
    lock=REPORTS/'evaluation_lock.json'
    if lock.exists():
        previous=json.loads(lock.read_text(encoding='utf-8'))
        expected={'model_sha256':receipt['sha256'],'selection_fingerprint':fingerprint(selection),
                  'dataset_fingerprint':selection['dataset_fingerprint']}
        if any(previous.get(key)!=value for key,value in expected.items()):
            raise RuntimeError('Onderbroken evaluatie heeft andere fingerprints; herstel uitsluitend het oorspronkelijke experiment')
        print('Technisch herstel van hetzelfde vergrendelde experiment; geen nieuwe modelkeuze.',flush=True)
    else:
        atomic_json(lock,{'started_at':now(),'model_sha256':receipt['sha256'],
                         'selection_fingerprint':fingerprint(selection),'dataset_fingerprint':selection['dataset_fingerprint']})
    artifact=load_model();history=historical_hourly(include_validation=True)
    checking=hourly('NYC','test')
    # De vaste dataset wordt dag voor dag geëvalueerd; latere testhistorie pas na die dag bekend.
    prediction_rows=[]
    for day,observations in checking.groupby(checking.timestamp.dt.normalize()):
        forecast=predict_day(day,history,artifact)
        baseline_history=history.set_index('timestamp').rides
        forecast['baseline_rides']=[float(baseline_history[t-pd.Timedelta(days=7)]) for t in forecast.timestamp]
        forecast.loc[forecast.clock_hours==0,'baseline_rides']=0
        forecast['rides']=observations.rides.to_numpy()
        prediction_rows.append(forecast)
        history=pd.concat([history,observations],ignore_index=True)
    prediction=pd.concat(prediction_rows,ignore_index=True)
    diagnostic=residual_table(prediction,prediction.predicted_rides)
    diagnostic['baseline_rides']=prediction.baseline_rides
    diagnostic.to_csv(REPORTS/'test_predictions.csv',index=False)
    for column in ['hour','weekday','season']:
        diagnostic.groupby(column).agg(MAE=('absolute_error','mean'),bias=('residual','mean'),n=('rides','size')).to_csv(REPORTS/f'test_errors_by_{column}.csv')
    score=metrics(prediction.rides,prediction.predicted_rides)
    baseline=metrics(prediction.rides,prediction.baseline_rides)
    daily=diagnostic.assign(day=diagnostic.timestamp.dt.normalize()).groupby('day').apply(
        lambda g:np.mean((g.rides-g.baseline_rides)**2)-np.mean((g.rides-g.predicted_rides)**2))
    weekly=daily.groupby(daily.index-pd.to_timedelta(daily.index.dayofweek,unit='D')).mean().to_numpy()
    rng=np.random.default_rng(SEED);boot=rng.choice(weekly,size=(10000,len(weekly)),replace=True).mean(axis=1)
    result={'finished_at':now(),'rows':len(prediction),'metrics':score,'baseline_metrics':baseline,
            'RMSE_improvement_percent':100*(baseline['RMSE']-score['RMSE'])/baseline['RMSE'],
            'MSE_gain_week_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist(),
            'ci_limitation':'Slechts circa negen weekblokken; onzekerheidsinterval is indicatief en veronderstelt beperkte afhankelijkheid.',
            'selected_model':selection['name'],'test_period':['2026-07-01','2026-08-31'],
            'fixed_model_rolling_observation':True,'retuned_after_test':False,
            'selection_fingerprint':receipt['selection_fingerprint'],'model_sha256':receipt['sha256']}
    # Zonder herfit na test: dit is precies het geëvalueerde artefact.
    write_model_card(selection,result)
    atomic_json(output,result)
    print(json.dumps(result,indent=2),flush=True)
    return result


def write_model_card(selection,result):
    text=f"""# Citi Bike-modelkaart

Gekozen model: **{selection['name']}**. Het voorspelt geregistreerde NYC-vertrekken
per lokaal uurvak, voor één volledige kalenderdag vanaf 00:00 New York-tijd.
Veertien volledige voorafgaande dagen zijn nodig. Jersey City is uitgesloten.

| Metriek | Eindmodel | Vorig weekuur |
|---|---:|---:|
| RMSE, ritten/uur | {result['metrics']['RMSE']:.2f} | {result['baseline_metrics']['RMSE']:.2f} |
| MAE, ritten/uur | {result['metrics']['MAE']:.2f} | {result['baseline_metrics']['MAE']:.2f} |
| R² | {result['metrics']['R2']:.4f} | {result['baseline_metrics']['R2']:.4f} |

RMSE-verbetering: {result['RMSE_improvement_percent']:.2f}%. Eindtest juli–augustus 2026,
{result['rows']} klokvakken. Selectie en tuning gebruikten deze uitkomsten niet.
Het model blijft vast; eerder gemeten testdagen worden daarna historische invoer.
Het artefact is na de eindtest niet opnieuw getraind.

Dit is geregistreerd gebruik, geen onvervulde vraag of beschikbaarheid per station.
Weer, evenementen en capaciteit ontbreken. Feature importance toont geen causaliteit.
Historische schema's en gebruik veranderen; twee testmaanden dekken niet alle seizoenen.
Het weekbootstrapinterval bevat weinig blokken en geeft beperkte zekerheid.
De actuele maandarchieven leveren niet automatisch de recente historie voor morgen.
De backtest veronderstelt een externe bron met vóór 00:00 beschikbare uurtellingen.
Publicatie-, ingestie- en rekenlatentie zijn niet gemodelleerd. De prestaties gelden
onder dit invoercontract; een operationele tellingenfeed moet apart worden gerealiseerd.

Hertrain offline met de README-commando's. Modelbestanden staan lokaal en buiten Git.
Bronnen, code, packageversies, seeds, splits en fingerprints zijn vastgelegd.
AWS-training blijft open voor de volledige schoolopdracht.
"""
    (REPORTS/'MODEL_CARD.md').write_text(text,encoding='utf-8')
