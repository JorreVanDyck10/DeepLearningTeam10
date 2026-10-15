"""Leesbare, uitvoerbare notebookstructuur rond de bestaande studie."""
import json
import sys
import nbformat as nbf
from nbclient import NotebookClient
from .config import ROOT

NAMES=['00_projectoverzicht.ipynb','01a_dataonderzoek_en_cleaning.ipynb',
       '01b_EDA_en_hypothesen.ipynb','02_definitieve_voorbereiding.ipynb',
       '03_baseline_en_Ridge.ipynb','04_PyCaret_vergelijking.ipynb',
       '05_Random_Forest.ipynb','06_LightGBM.ipynb','07_eindvergelijking.ipynb']
CONTRIBUTIONS='''## Bijdragen — Messi10

**Raoul Buffels:** dataverwerking en cleaning, EDA en hypothesen, features, modellen,
tuning, evaluatie en documentatie; Citi Bike-frontend en backend.
**Jorre Van Dyck:** hulp bij deployment op Render.
Codex hielp met implementatie, uitleg, tests en uitvoering. Ieder teamlid moet de
eigen keuzes mondeling kunnen uitleggen.'''
INIT='''from pathlib import Path
import sys, json
import pandas as pd
from IPython.display import display, Markdown
locations=[Path.cwd(),Path.cwd().parent,Path.cwd()/'Citybike_full']
ROOT=next(p for p in locations if (p/'citibike').exists())
sys.path.insert(0,str(ROOT))
from citibike.config import REPORTS, DATA, MODELS
def report(name):
    return json.loads((REPORTS/name).read_text(encoding='utf-8'))
def table(name):
    display(pd.read_csv(REPORTS/name,dtype={'start_station_id':'string','end_station_id':'string','station_id':'string'}))
'''
MODEL_INIT='''from time import perf_counter
import gc
from citibike.model_data import load_model_data
from citibike.features import FEATURES, CALENDAR, SavedChronologicalCV
from citibike.modeling import OperationalRegressor, SeasonalNaive, metrics
from citibike.reproduction import save_model_reproduction
training, validation, prepared=load_model_data()
X_train, y_train=training[FEATURES],training.rides
X_validation, y_validation=validation[FEATURES],validation.rides
display(pd.DataFrame(prepared['folds']))
display(pd.DataFrame({'feature':FEATURES}))
print(f'Training: {len(training):,} rijen; validatie: {len(validation):,}. Geen testdata.')
'''


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)
def pair(explanation,source): return [md(explanation),code(source)]


def start(title,description,model=False):
    result=[md('# '+title+'\n\n'+description+'\n\n'+CONTRIBUTIONS)]
    result+=pair('We bepalen de projectmap en laden de kleine tabellen. Grote bronbestanden blijven op schijf.',INIT)
    if model:
        result+=pair('## Modelinvoer en preprocessing\n\nWe laden exact de output van notebook 02: kalenderwaarden en historische tellingen. De vijf chronologische folds hebben ieder 28 latere validatiedagen. Alleen training mag preprocessing leren; de eindtest wordt niet geladen.',MODEL_INIT)
    return result


def legacy(*numbers):
    sections=json.loads((ROOT/'citibike/notebook_sections.json').read_text(encoding='utf-8'))
    return [md(c['source']) if c['cell_type']=='markdown' else code(c['source']) for n in numbers for c in sections[str(n)]]


def overview():
    cells=start('00 — Projectoverzicht','We voorspellen geregistreerde NYC-vertrekken per lokaal uurvak, één dag vooruit om 00:00 New York-tijd. Dit is geregistreerd gebruik, geen fietsvoorraad of onvervulde vraag. Jersey City wordt apart onderzocht.')
    cells+=pair('## Leesvolgorde\n\nOBTAIN → SCRUB → EXPLORE → MODEL → INTERPRET → DEPLOY. 01a onderzoekt kwaliteit; 01b patronen en statistiek; 02 maakt de modeldata; 03–06 reproduceren de modellen; 07 vergelijkt uitkomsten en fouten.',"display(pd.DataFrame({'volgorde':"+repr(NAMES)+"}))")
    cells += [md('## Reproduceerbaarheid\n\nGebruik Python 3.11 en requirements-lock.txt. Brondata, caches en modellen blijven buiten Git. Zie README voor opnieuw openen en een volledige nieuwe reproductie. Overlapcontrole, DST, streaming ingestie en leakagebescherming blijven behouden.\n\nTraining: vóór mei 2026; validatie: mei–juni; reeds bekeken eindtest: juli–augustus. Geen nieuwe keuze na de test. De oorspronkelijke rapporten blijven apart van notebook_reproduction.'),
              md('## Deployment en opdrachtvereisten\n\nDe bestaande Netlify-frontend gebruikt de FastAPI/Render-backend en dezelfde predict_day. Streamlit blijft een lokale demo. De lokale retrainingpipeline maakt een gecontroleerd artefact. Runnerregistratie en live modelupdate blijven handmatige stappen; zie DEPLOYMENT.md.\n\nDe groepsrepo bevat een Mushroom-SageMaker-notebook en tuningexports. **OPENSTAAND: AWS / SageMaker model — cloudprovenance bevestigen.** OPDRACHTCONTROLE.md vermeldt bewijs en resterende stappen.')]
    return cells


def cleaning():
    cells=start('01a — Dataonderzoek en cleaning','Een geldige start kan bruikbaar zijn voor tellingen terwijl eindstation of duur onbruikbaar is. We onderzoeken schema, ontbrekende velden, duplicaten, foutieve waarden, extreme waarden en categorieën.')
    cells+=pair('We laden het technische bronmanifest en ontwikkelingsplots. Op een nieuwe computer voer je eerst run.py prepare uit.',"from citibike.datasets import manifest\nfrom citibike.plots import show")
    cells+=legacy(2,3,4,5)
    cells+=pair('## Outliers en categorieën\n\nLange geregistreerde duur bewijst geen voortdurend fietsen. We markeren en onderzoeken deze waarden; een geldige start telt nog mee. Oude onbekende gebruikers- en fietscategorieën blijven zichtbaar als unknown.',"show('duration')\ntable('NYC_duration_statistics.csv')\nshow('user_bike')\ntable('NYC_user_bike.csv')")
    cells+=pair('## Beslissingen voor de definitieve voorbereiding\n\nDeze regels worden in notebook 02 via dezelfde modules toegepast. We verwijderen geen waarden omdat een modelscore daarmee beter lijkt.',"display(pd.DataFrame([\n {'probleem':'bronkopie','beslissing':'alleen bewezen kopieën verwijderen; historische multipliciteit bewaren'},\n {'probleem':'missing eindstation','beslissing':'geldige start behouden voor tellingen'},\n {'probleem':'negatieve/nulduur','beslissing':'uitsluiten van duurstatistiek, flag behouden'},\n {'probleem':'duur >24 uur','beslissing':'markeren en onderzoeken, geen willekeurige cutoff'},\n {'probleem':'onbekende categorie','beslissing':'unknown behouden'},\n {'probleem':'DST ambiguïteit','beslissing':'alleen onbetrouwbare fysieke duur uitsluiten'},\n {'probleem':'ontbrekende maand','beslissing':'stoppen, niet als nul invullen'}\n]))")
    return cells


def eda():
    cells=start('01b — EDA en hypothesen','Alle patronen komen uit ontwikkeling vóór mei 2026. Weekherhaling is het hoofdverhaal en motiveert het uurdoel; praktische voorspelwinst moet later uit validatie blijken.')
    cells+=pair('Na iedere figuur staat de berekende interpretatie. De aggregaties blijven gescheiden voor NYC en JC.',"from citibike.plots import show")
    cells+=legacy(6,7)
    cells+=pair('## Onderzoeksvraag → H0/H1 → methode\n\nHangt uurgebruik samen met hetzelfde uur vorige week? **H0:** de gemiddelde correlatie over tijdsblokken is nul. **H1:** deze verschilt van nul. Alpha=0,05. De vraag volgt uit EDA en is exploratief.\n\nWe berekenen weekcorrelaties en gebruiken de bestaande sign-fliptoets op vierweekblokken. Miljoenen ritten worden niet als onafhankelijk behandeld. Voldoende onafhankelijke blokken en symmetrie onder H0 blijven aannamen. Holm corrigeert voor drie vragen. De kleinste Monte-Carlo-p is 1/20001, niet nul.',"from citibike.analysis import hypothesis_family\nimport numpy as np\nhistory=pd.read_parquet(DATA/'NYC_hourly_development.parquet')\nduration_daily=pd.read_csv(REPORTS/'NYC_duration_daily.csv')\nrecalculated=hypothesis_family(history,duration_daily)\nfor old,new in zip(report('hypotheses.json'),recalculated):\n    for key in ['statistic','p_value','effect_size','holm_p_value']:\n        np.testing.assert_allclose(old[key],new[key],rtol=1e-12)\nsignal=recalculated[1]\ndisplay(pd.DataFrame([{k:signal[k] for k in ['label','alpha','statistic','effect_size','unit','ci95','p_value','holm_p_value','blocks','reject_H0']}]))")
    cells+=pair('## Resultaat → interpretatie\n\nDe effectgrootte en het interval tonen de sterkte van de samenhang. Trend en vaste patronen kunnen correlatie verhogen. Dit is geen causaliteit en geen zelfstandig bewijs dat een model de baseline verslaat.',"display(Markdown(f\"Weekcorrelatie **{signal['effect_size']:.3f}**, 95%-interval {signal['ci95']}. Holm-p {signal['holm_p_value']:.6g}. H0 {'verworpen' if signal['reject_H0'] else 'niet verworpen'}.\"))")
    cells+=pair('## Beslissing: het bestaande voorspelprobleem\n\nPas nu bespreken we uurvolume. De destijds vastgelegde gate vroeg Holm-p <0,05 en een onderste correlatiegrens >0,3. De weekherhaling en nuttige uurresolutie ondersteunen het bestaande doel. We veranderen de doelkeuze niet.',"display(pd.DataFrame(report('prediction_contracts.json')['contracts']))\ndisplay(pd.DataFrame(report('target_comparison.json')))\ndecision=report('target_decision.json')\ndisplay(Markdown(decision['chosen_prediction_problem']+'\\n\\n'+decision['reason']))")
    cells+=pair('## Aanvullend: overige hypothesen en robuustheid\n\nWerkdag/weekend en gebruikersduur zijn aanvullende analyses. Achtweekblokken controleren langere afhankelijkheid. Groepsverschillen bewijzen geen individuele voorspelbaarheid. De intervallen zijn per hypothese; Holm corrigeert de p-waarden.',"display(pd.DataFrame(recalculated)[['label','H0','H1','effect_size','unit','ci95','p_value','holm_p_value','reject_H0','sensitivity_8weeks']])\ndisplay(pd.DataFrame([report('hypothesis_protocol.json')]).drop(columns='EDA_evidence'))")
    cells+=[md('## Aanvullend: ACF en ADF\n\nAutocorrelatie toont herhaling; ADF is een diagnose. Beide zijn aanvullend en maken ARIMA of differencing niet noodzakelijk.')]
    cells+=legacy(8)
    return cells


def preparation():
    cells=[md('# 02 — Definitieve voorbereiding zonder grafieken\n\nZIPs → vaste cleaning → uuraggregatie → features → target → chronologische splits → modelinvoer. Geen trainingsrapporten of modellen nodig.\n\n'+CONTRIBUTIONS)]
    cells+=pair('We laden alleen bronverwerking, features en voorbereiding. Er wordt geen plot- of modelmodule geïmporteerd.',"from pathlib import Path\nimport sys\nimport pandas as pd\nfrom IPython.display import display\nlocations=[Path.cwd(),Path.cwd().parent,Path.cwd()/'Citybike_full']\nROOT=next(p for p in locations if (p/'citibike').exists())\nsys.path.insert(0,str(ROOT))\nfrom citibike.config import SOURCE\nfrom citibike.model_data import prepare_model_data,load_model_data,CONTRACT\nfrom citibike.features import FEATURES,leakage_audit")
    cells+=pair('## Cleaning → aggregatie → features → target\n\nDe volgende stap controleert bron- en cachechecksums, harmoniseert schema’s, verwijdert bewezen bronkopieën en bewaart kwaliteitsflags. Geldige starts worden per lokaal uur geteld. Ontbrekende maanden veroorzaken een fout.\n\nTarget: geregistreerde vertrekken per uurvak. Kalenderfeatures zijn vooraf bekend; daglags gebruiken afgesloten dagen. Rolling wordt eerst verschoven. De eerste veertien dagen missen historie en vallen uit modeldata. DST gebruikt 24 lokale labels met 0 of 2 fysieke uren waar nodig. Afwijkende bestaande modeldata worden niet overschreven.',"prepared=prepare_model_data(SOURCE)\ndisplay(pd.DataFrame([CONTRACT]))")
    cells+=pair('## Splits en voorbereide uitvoer\n\nTraining: vóór mei; validatie: mei–juni. De testpartitions juli–augustus blijven afgeschermd. Deze voorbereiding vereist geen evaluatie-lock en leest geen testlabels. De vijf folds komen alleen uit training.',"training,validation,prepared=load_model_data()\ndisplay(pd.DataFrame([{k:v for k,v in prepared.items() if k not in ['folds','contract']}]))\ndisplay(pd.DataFrame(prepared['folds']))\ndisplay(training[['timestamp','rides']+FEATURES].tail(3))\ndisplay(validation[['timestamp','rides']+FEATURES].head(3))")
    cells+=pair('## Beschikbaarheid om 00:00\n\nGeen telling op de voorspeldag beïnvloedt eerdere features. Historische tellingen moeten tijdig beschikbaar zijn; maandarchieven bewijzen geen live feed. De modelnotebooks laden exact deze output.',"display(pd.DataFrame(leakage_audit()))")
    return cells


def fitted_model(name,expression,explanation):
    cells=pair('## '+name+'\n\n'+explanation+' De wrapper gebruikt dezelfde clipping en DST-regels als deployment.',expression+'\ndisplay(model)')
    cells+=pair('Hier trainen we het concrete model op de voorbereide trainingsrijen. Preprocessing leert alleen uit training.',"started=perf_counter()\nfitted=model.fit(X_train,y_train)\nfit_seconds=perf_counter()-started")
    cells+=pair('Hier voorspellen en evalueren we. RMSE, MAE en R² worden voor training en validatie berekend en tegen de oorspronkelijke scores gecontroleerd. Het rapport wordt apart opgeslagen; test en modelkeuze blijven behouden.',"train_prediction=fitted.predict(X_train)\nvalidation_prediction=fitted.predict(X_validation)\nresult=save_model_reproduction("+repr(name)+",training,validation,train_prediction,validation_prediction,fit_seconds,prepared)\ndisplay(pd.DataFrame([result]))\ndel fitted,model\ngc.collect()")
    return cells


def cv_and_tuning(names,family=None):
    cells=pair('## Chronologische CV en train/validatie\n\nWe behouden de oorspronkelijke vijf CV-metingen naast de opnieuw uitgevoerde vaste fit. Dit is reproductie; we starten geen nieuwe modelselectie na de test.',"for row in report('model_comparison.json'):\n    if row['name'] in "+repr(names)+":\n        display(Markdown('**'+row['name']+'**'))\n        display(pd.DataFrame(row['cv']))")
    if family:
        cells+=pair('## Bestaande tuning\n\nParameterSampler onderzocht de beperkte zoekruimte met seed 42. Iedere instelling werd op dezelfde trainingsfolds beoordeeld met gemiddelde CV-RMSE. Mei–juni vergeleek daarna de vaste kandidaten. Testdata waren uitgesloten. We bewaren alle instellingen en uitkomsten; de standaarduitvoering zoekt niet opnieuw.',"display(pd.DataFrame([report('tuning_protocol.json')]))\ntable('"+family+"_tuning.csv')")
    return cells


def baseline_ridge():
    cells=start('03 — Baseline en Ridge','De weekbaseline kopieert hetzelfde uur vorige week. Ridge leert een gewogen som; regularisatie beperkt grote gewichten. StandardScaler wordt binnen iedere trainfit geleerd.',True)
    cells+=pair('We importeren de concrete estimator en preprocessing.',"from sklearn.linear_model import Ridge\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler")
    cells+=fitted_model('seasonal_naive','model=OperationalRegressor(SeasonalNaive())','Geen hyperparameters; dit is onze eenvoudige referentie.')
    cells+=fitted_model('ridge','model=OperationalRegressor(make_pipeline(StandardScaler(),Ridge(alpha=10)))','De bestaande alpha is 10; alle 14 features blijven behouden.')
    cells+=fitted_model('ridge_calendar','model=OperationalRegressor(make_pipeline(StandardScaler(),Ridge(alpha=10)),columns=CALENDAR)','Deze bestaande ablatie gebruikt alleen kalenderfeatures.')
    cells+=fitted_model('ridge_no_14d',"model=OperationalRegressor(make_pipeline(StandardScaler(),Ridge(alpha=10)),columns=[c for c in FEATURES if c!='lag_14d'])",'Deze bestaande ablatie laat de 14-dagenlag weg; het is geen nieuwe featurekeuze.')
    cells+=cv_and_tuning(['seasonal_naive','ridge','ridge_calendar','ridge_no_14d'])
    cells+=[md('## Conclusie en tuningkeuze\n\nRidge verbetert de baseline maar valideert minder goed dan de niet-lineaire modellen. Alpha=10 bleef vast. RF en LightGBM waren de twee kandidaten die op basis van trainings-CV verder getuned werden. Kalender-only mist recente intensiteit; we behouden ook deze minder geslaagde experimenten.')]
    return cells


def forest():
    cells=start('05 — Random Forest','Random Forest middelt bomen op verschillende steekproeven. Het leert niet-lineaire verbanden, maar kan training te precies passen.',True)
    cells+=pair('De volgende import levert de werkelijke estimator.',"from sklearn.ensemble import RandomForestRegressor")
    cells+=fitted_model('random_forest','model=OperationalRegressor(RandomForestRegressor(n_estimators=160,max_depth=16,min_samples_leaf=8,max_features=.8,n_jobs=2,random_state=42))','Het basismodel gebruikt 160 bomen, diepte 16 en minimaal 8 rijen per blad.')
    cells+=fitted_model('tuned_random_forest','model=OperationalRegressor(RandomForestRegressor(n_estimators=320,max_depth=None,min_samples_leaf=4,max_features=.6,n_jobs=2,random_state=42))','Dit zijn exact de oorspronkelijk gekozen instellingen: 320 bomen en minimaal 4 rijen per blad.')
    cells+=pair('De oorspronkelijke zoekruimte regelt bomen, diepte, bladgrootte en onderzochte features. Zes seeded configuraties beperkten de rekentijd.',"search_space={'n_estimators':[160,240,320],'max_depth':[10,16,22,None],'min_samples_leaf':[4,8,16],'max_features':[.6,.8,1.]}\ndisplay(pd.DataFrame({k:pd.Series(v) for k,v in search_space.items()}))")
    cells+=cv_and_tuning(['random_forest','tuned_random_forest'],'random_forest')
    cells+=[md('## Conclusie\n\nTrain-RMSE circa 536 tegenover validatie-RMSE circa 1940: een grote kloof. Overfit en veranderde systeemintensiteit zijn mogelijke oorzaken; de kloof bewijst niet uitsluitend overfit. RF-tuning bracht circa 0,27% validatiewinst en een groot artefact van circa 647 MiB. De vooraf vastgelegde keuze blijft behouden.')]
    return cells


def lightgbm():
    cells=start('06 — LightGBM','Boosting bouwt bomen na elkaar die eerdere fouten corrigeren. Learning rate, bomen, bladeren en minimale bladgrootte regelen capaciteit. Het model staat in het cursusoverzicht.',True)
    cells+=pair('LightGBM gebruikt dezelfde numerieke features, zonder een vooraf globaal geleerde scaler.',"from lightgbm import LGBMRegressor")
    cells+=fitted_model('lightgbm','model=OperationalRegressor(LGBMRegressor(n_estimators=250,num_leaves=31,learning_rate=.05,min_child_samples=40,n_jobs=2,verbosity=-1,random_state=42,deterministic=True,force_col_wise=True))','Dit is de oorspronkelijke basisconfiguratie.')
    cells+=fitted_model('tuned_lightgbm','model=OperationalRegressor(LGBMRegressor(n_estimators=200,num_leaves=31,learning_rate=.1,min_child_samples=100,n_jobs=2,verbosity=-1,random_state=42,deterministic=True,force_col_wise=True))','Deze bestaande instelling won binnen de trainings-CV-zoekprocedure.')
    cells+=pair('De oorspronkelijke zoekruimte bleef beperkt tot twintig seeded configuraties.',"search_space={'n_estimators':[200,400,650],'num_leaves':[15,31,63],'learning_rate':[.025,.05,.1],'min_child_samples':[20,50,100]}\ndisplay(pd.DataFrame(search_space))")
    cells+=cv_and_tuning(['lightgbm','tuned_lightgbm'],'lightgbm')
    cells+=[md('## Conclusie\n\nLightGBM traint snel en valideert dicht bij RF. De betere trainings-CV door tuning gaf geen betere mei–juni-score. Meer tuning is niet automatisch beter. Beide uitkomsten blijven behouden; de RF-keuze verandert niet.')]
    return cells


def pycaret():
    cells=start('04 — PyCaret-vergelijking','AutoML vergelijkt dezelfde vier operationele basismodellen. Geen random holdout; geen nieuwe finale modelkeuze.',True)
    cells+=pair('## Setup\n\nPyCaret test_data ontvangt hier mei–juni-validatie, niet de eindtest. preprocess=False voorkomt globale preprocessing. Ridge schaalt binnen zijn eigen fold-estimator.',"from pycaret.regression import RegressionExperiment\nfrom citibike.modeling import candidates\nfrom citibike.reproduction import save_pycaret_reproduction\nexperiment=RegressionExperiment()\nexperiment.setup(data=training[FEATURES+['rides']],test_data=validation[FEATURES+['rides']],target='rides',fold_strategy=SavedChronologicalCV(prepared['folds']),fold=len(prepared['folds']),data_split_shuffle=False,fold_shuffle=False,preprocess=False,session_id=42,n_jobs=2,html=False,verbose=False,system_log=False,log_experiment=False,index=False)")
    cells+=pair('## Werkelijke training\n\ncompare_models voert de vijf chronologische folds uit en fit de vier kandidaten. We controleren de werkelijk gebruikte interne rijen en bewaren deze reproductie apart.',"ranked=experiment.compare_models(include=list(candidates().values()),sort='RMSE',n_select=4,errors='raise',verbose=False)\nautoml=save_pycaret_reproduction(experiment,ranked,training,validation,prepared)\ndisplay(automl)")
    cells+=pair('## Prediction en evaluatie\n\nDe teruggegeven estimators zijn op training gefit. We voorspellen expliciet op training en validatie en tonen RMSE, MAE en R².',"rows=[]\nfor name,fitted in zip(automl.candidate,ranked):\n    train_prediction=fitted.predict(X_train.astype('float32'))\n    valid_prediction=fitted.predict(X_validation.astype('float32'))\n    rows.append({'name':name,**metrics(y_validation,valid_prediction),**{'train_'+k:v for k,v in metrics(y_train,train_prediction).items()}})\ndisplay(pd.DataFrame(rows))\nfrom citibike.config import atomic_json\natomic_json(REPORTS/'notebook_reproduction/pycaret_validation.json',rows)")
    cells+=pair('## Conclusie en tuning\n\nPyCaret verkleint numerieke dtypes; vooral LightGBM-binning kan daardoor verschillen. De oorspronkelijke handmatige validatiemetrics bepaalden de finale keuze. Tuning staat bij RF en LightGBM; AutoML krijgt geen nieuwe zoekprocedure.',"manual=pd.DataFrame(report('model_comparison.json'))[['name','cv_RMSE']]\nprecision=automl[['candidate','RMSE']].merge(manual,left_on='candidate',right_on='name')\nprecision['CV_RMSE_verschil']=precision.RMSE-precision.cv_RMSE\ndisplay(precision)\ndisplay(pd.DataFrame(prepared['folds']))")
    return cells


def comparison():
    cells=start('07 — Eindvergelijking en foutenanalyse','De finale RF-keuze lag vóór de test vast. We vergelijken originele en gecontroleerde reproducties; geen nieuwe tuning of testvoorspellingen.')
    cells+=pair('## Centrale vergelijking\n\nTestkolommen zijn alleen gevuld voor de vastgelegde winnaar en baseline. De oorspronkelijke tijd is CV + fit + opslag. PyCaret-CV staat apart van de handmatige validatiemetrics.',"from citibike.reporting import comparison_table\nfrom citibike.plots import show\ndisplay(comparison_table())\ntable('pycaret_comparison.csv')\ndisplay(pd.DataFrame(report('notebook_reproduction/pycaret_validation.json')))")
    cells+=pair('## Gecontroleerde nieuwe fits\n\nDe scores komen uit notebooks 03, 05 en 06 en moeten gelijk zijn aan de oorspronkelijke studie. Nieuwe fitduur staat apart van historische CV + fit + opslag.',"reproduced=pd.DataFrame([json.loads(p.read_text(encoding='utf-8')) for p in sorted((REPORTS/'notebook_reproduction').glob('*.json')) if p.stem not in ['pycaret_audit','pycaret_validation']])\ndisplay(reproduced[['name','train_RMSE','RMSE','MAE','R2','fit_seconds','original_metrics_equal']])\nshow('comparison')")
    cells+=pair('## Keuze en generalisatie\n\nRF valideert circa 31,34% beter dan de weekbaseline. De grote train/validatiekloof en slechts circa 0,27% tuningwinst moeten naast de score besproken worden. LightGBM is sneller en kleiner; de vastgelegde 1%-regel vergelijkt modeltypen, en binnen een familie de validatiefout. We veranderen die regel niet na de test. RMSE weegt grote fouten zwaarder dan MAE; R² kan negatief zijn.',"selection=report('model_selection.json')\ndisplay(pd.DataFrame([{k:v for k,v in selection.items() if k not in ['parameters','modeling_code_version']}]))")
    cells+=legacy(17)
    cells+=pair('## Grootste fouten\n\nResidual = actual − prediction; positief betekent onderschatting. Bekijk uur, seizoen en drukte. Een normale verdeling maakt een model niet perfect. Weer en evenementen zijn mogelijke ontbrekende factoren, geen bewezen oorzaak van één fout.',"residuals=pd.read_csv(REPORTS/'validation_residuals.csv')\ndisplay(residuals.nlargest(10,'absolute_error'))\ndisplay(pd.DataFrame([{'gemiddelde_bias':residuals.residual.mean(),'mediaan_absolute_fout':residuals.absolute_error.median()}]))")
    cells+=legacy(18,19)
    cells+=[md('## Aanvullend: oorspronkelijke vensters, tuning en ablaties\n\nWe behouden ook experimenten zonder winst. Dit zijn historische ontwikkelingsbeslissingen; we gebruiken ze nu niet om een nieuwe keuze te maken.')]
    cells+=legacy(16)
    cells+=pair('## Gedeelde deploymentfunctie\n\nDeze demonstratiedag komt uit ontwikkeling en is geen onafhankelijke prestatietest. API en Streamlit gebruiken dezelfde predict_day.',"from citibike.inference import predict_day\nhistory=pd.read_parquet(DATA/'NYC_hourly_development.parquet')\nday=history.timestamp.max().normalize()\ndisplay(predict_day(day,history[history.timestamp<day]))")
    cells+=pair('## Beperkingen en eindcontrole\n\nRestpatronen en twee zomermaanden als eindtest beperken generalisatie. Een nieuwe modelkeuze vraagt een nieuwe onafhankelijke testperiode. Zie OPDRACHTCONTROLE.md voor runnerregistratie, live update en AWS-cloudbewijs.',"display(Markdown((REPORTS/'MODEL_CARD.md').read_text(encoding='utf-8')))\ndisplay(pd.DataFrame([report('verification.json')]))\ndisplay(pd.DataFrame([report('test_results.json')]).drop(columns='test_names'))")
    return cells


def bridge(source):
    purposes=[('resource_profile','We vergelijken gemeten geheugen en rekentijd. Gesamplede RSS mist mogelijk korte pieken en is geen totale workerpiek.'),
              ('assembly_seconds','We onderscheiden gemeten schrijfstappen van checksums, globale controles en eerdere mislukte pogingen.'),
              ('month_boundary','We delen ritten in op vertrekdatum; alleen bewezen bronkopieën worden verwijderd.'),
              ('quality =','Kwaliteitsflags tonen welke velden bruikbaar zijn. Een missing veld verwijdert niet automatisch de hele rit.'),
              ('duration_time_quality','DST kan fysieke duur onbetrouwbaar maken terwijl de start nog bruikbaar is voor tellingen.'),
              ('duration_sample_audit','Concrete extreme ritten tonen waarom lange geregistreerde duur niet zomaar werkelijk fietsen betekent.'),
              ('missing_values','We tonen missing velden en de afzonderlijke duplicaatsteekproef; die steekproef bewijst geen volledige deduplicatie.'),
              ('inflow_outflow','Starts en eindes per station zijn geen actuele fietsvoorraad of bewezen herbalancering.'),
              ('training_windows','We bewaren historische vensters en alle tuninginstellingen, ook zonder validatiewinst.'),
              ('tuning_protocol','Dit protocol legt de oorspronkelijke zoekbegroting, seed en folds vast, zonder eindtestinformatie.'),
              ("show('",'Deze figuur toont het geregistreerde patroon. De berekende interpretatie onder de grafiek beschrijft de betekenis en beperking.')]
    for needle,text in purposes:
        if needle in source: return text
    raise ValueError('Specifieke uitleg nodig voor: '+source[:100])


def build():
    builders=[overview,cleaning,eda,preparation,baseline_ridge,pycaret,forest,lightgbm,comparison]
    directory=ROOT/'notebooks';directory.mkdir(exist_ok=True)
    for name,builder in zip(NAMES,builders):
        cells=[]
        for cell in builder():
            if cell.cell_type=='code' and (not cells or cells[-1].cell_type!='markdown'): cells.append(md(bridge(cell.source)))
            cells.append(cell)
        nbf.write(nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Citi Bike (.venv Python 3.11)','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}}),directory/name)
    return NAMES


def execute(names=None,rebuild=True):
    if rebuild: build()
    from jupyter_client import KernelManager
    for name in names or NAMES:
        path=ROOT/'notebooks'/name;notebook=nbf.read(path,as_version=4)
        manager=KernelManager(kernel_name='python3')
        manager.kernel_spec.argv=[sys.executable,'-X','utf8','-m','ipykernel_launcher','-f','{connection_file}']
        client=NotebookClient(notebook,timeout=1800,kernel_name='python3',km=manager,resources={'metadata':{'path':str(ROOT)}})
        try: client.execute()
        finally:
            nbf.write(notebook,path)
            if manager.has_kernel: manager.shutdown_kernel(now=True)
        print('Notebook uitgevoerd: '+name,flush=True)
