"""Didactische notebooks met korte cellen; uitvoering hergebruikt echte rapporten."""
import json
import os
import sys
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient
from .config import ROOT,REPORTS

INIT='''from pathlib import Path
import sys, json
import pandas as pd
from IPython.display import display, Markdown
locations = [Path.cwd(), Path.cwd().parent, Path.cwd() / 'Citybike_full']
ROOT = next((p for p in locations if (p / 'citibike').exists()), None)
if ROOT is None: raise RuntimeError('Start dit notebook vanuit Citybike_full of notebooks.')
sys.path.insert(0, str(ROOT))
from citibike.config import REPORTS, DATA, MODELS
from citibike.datasets import manifest
from citibike.plots import show
def report(name):
    return json.loads((REPORTS / name).read_text(encoding='utf-8'))
def table(name):
    display(pd.read_csv(REPORTS / name, dtype={'start_station_id':'string','end_station_id':'string','station_id':'string'}))
'''


def md(text):return nbf.v4.new_markdown_cell(text)
def code(text):return nbf.v4.new_code_cell(text)


def overview_cells():
    return [
      md('# NYC Citi Bike Analysis & Predictive Modeling\n\n**OBTAIN → SCRUB → EXPLORE → MODEL → INTERPRET → DEPLOY**\n\nTeamleden en echte bijdragen: **door het team in te vullen**. Codex hielp met implementatie, tests en uitvoering; controleer het opleidingsbeleid en verdedig iedere stap zelf.'),
      md('## 0. Projectdoel\n\nWat vertellen geregistreerde ritten over het systeem, en welk nuttig doel kunnen we voorspellen? Het voorspeldoel wordt pas na ontwikkelings-EDA en statistische onderzoek gekozen. NYC en Jersey City blijven gescheiden. Juli–augustus 2026 is uitsluitend eindtest; AWS-training volgt later.'),
      md('## 1. Imports & configuratie\n\nDe CLI voert de zware stappen uit. Dit notebook toont de werkelijk berekende resultaten. Ontbrekende rapporten zijn fouten, geen verzonnen vervanging. Zie README voor een volledige herbouw.'),code(INIT),
      md('## 2. Data discovery — OBTAIN\n\nDe code inventariseert bronnen en CSV-leden, inclusief maand-ZIPs in jaararchieven. SHA-256 identificeert de bronnen. Technische eindtestgegevens zoals schema en rij-aantallen mogen hier worden getoond; eindtestpatronen niet.'),
      code("m = manifest()\ndisplay(pd.DataFrame([{'bronbestanden':m['source_count'],'bron_GiB':m['source_bytes']/2**30,'ruwe_rijen':m['raw_rows'],'bewaarde_rijen':m['retained_rows'],'laatste_run_minuten_inclusief_cachecontrole':m['processing_seconds']/60}]))\ndisplay(pd.DataFrame(m['coverage']).T)"),
      md('## 3. Schemaonderzoek\n\nHistorische headers verschillen in namen, hoofdletters en inhoud. We mappen start/stop, gebruiker en coördinaten naar één schema. Oudere bronnen bevatten geen ride-ID of fietstype. Station-ID’s blijven tekst; alleen de gedocumenteerde integer-`.0`-weergave in oude data wordt genormaliseerd.'),
      code("display(pd.DataFrame([{'originele_kolommen':s['raw'],'canonieke_kolommen':s['canonical']} for s in m['schemas'].values()]))\ndisplay(Markdown(str(m['normalization'])))"),
      md('## 4. Data assembly\n\nArrow leest batches van 8 MiB. DuckDB gebruikt maximaal 4 GiB en kan naar schijf spillen. Parquet bewaart één bruikbare kolomlaag. Bij overlappende bronversies vergelijkt de code de volledige canonieke rijmultisets in beide richtingen; gelijke historische ritten behouden hun multipliciteit.'),
      code("parts = pd.DataFrame(m['partitions'])\ndisplay(parts.groupby('area')[['raw_rows','retained_rows','removed_rows','size_bytes','processing_seconds']].sum())\ndisplay(parts.loc[parts.overlap_removed > 0,['area','period','raw_rows','retained_rows','overlap_removed']])\ndisplay(Markdown(f\"Hoogste gesamplede proces-RSS: {parts.peak_rss_bytes_sampled.max()/2**30:.2f} GiB. Dit is een gemeten ondergrens van de piek; het omvat niet elk kortdurend maximum.\"))"),
      code("if (REPORTS/'resource_profile.json').exists():\n    resources=pd.DataFrame(report('resource_profile.json')['processes'])\n    resources['peak_RSS_GiB']=resources.peak_rss_bytes/2**30\n    display(resources.groupby('stage')[['peak_RSS_GiB','samples']].max())\n    display(Markdown(report('resource_profile.json')['note']))\ndisplay(Markdown(f\"Som van gemeten afzonderlijke partitionbouwtijden: {parts.processing_seconds.sum()/60:.1f} minuten. De manifest-walltime is de laatste run inclusief cachecontrole; eerdere mislukte pogingen zijn geen onderdeel van die walltime.\"))"),
      md('De resourceprofielen meten RSS van CLI-hoofdprocessen, niet de gezamenlijke RSS van eventuele PyCaret/joblib-workers. Korte pieken kunnen tussen samples vallen. De DuckDB-limiet geldt voor intern DuckDB-geheugen en is geen limiet op alle Python-processen samen.'),
      code("display(Markdown(f\"Gemeten schrijfstappen voor de maandgrenscorrectie: {parts.get('assembly_seconds',pd.Series(dtype=float)).sum()/60:.1f} minuten. Globale controle, checksumtijd en eerdere onderbroken pogingen vallen buiten deze som; de logs en hervataudits leggen die afzonderlijk vast.\"))"),
      md('## 5. Cleaning — SCRUB\n\nEen rit kan bruikbaar zijn voor vertrektellingen maar niet voor duur of geografie. We behouden flags, zodat ontbrekende eindstations geldige vertrekken niet automatisch verwijderen. Negatieve/nulduur is onbruikbaar voor duur. Ritten langer dan 24 uur worden gemarkeerd en onderzocht, niet automatisch verwijderd. Nullable onbekende historische velden zijn geen parsingfouten.'),
      md('Maandarchieven bevatten soms vertrekrecords uit een andere maand. De technische assemblylaag deelt deze in op de werkelijke lokale vertrekdatum. Gelijke moderne ID-inhoud en volledig bewezen historische bronsegmentkopieën tellen eenmaal; conflicten stoppen verwerking. Identieke historische ritten binnen een bron behouden hun multipliciteit. De eindtestgrenzen volgen vertrekdatums, niet ZIP-namen. Originele maandcaches blijven bewaard voor betrouwbare cache-invalidation.'),
      code("display(pd.DataFrame(m['month_boundary_audit']))\ndisplay(parts.loc[(parts.get('rows_moved_in',0)+parts.get('rows_moved_out',0)+parts.get('cross_partition_duplicates_removed',0))>0,['area','period','rows_moved_in','rows_moved_out','cross_partition_duplicates_removed']])"),
      code("quality = pd.DataFrame([{'area':p['area'],'period':p['period'],**p['quality_after']} for p in m['partitions']])\ndisplay(quality.groupby('area').sum(numeric_only=True).T)\ndisplay(parts[['area','period','raw_rows','retained_rows','removed_rows','removed_percent']].head(15))"),
      md('Offsetloze tijden kunnen bij het herhaalde najaarsuur geen unieke fysieke ritduur geven. De statistische duurlaag gebruikt de oude expliciet gerapporteerde duur waar beschikbaar; anders het verschil na tijdzoneconversie. Ritten met een ambigu start- of einduur worden alleen uit fysieke-duurstatistiek uitgesloten, niet uit vertrektellingen. De oorspronkelijke klokduur en flags blijven in Parquet.'),
      code("table('NYC_duration_time_quality.csv')\ntable('JC_duration_time_quality.csv')"),
      code("display(pd.DataFrame([report('NYC_duration_sample_audit.json'),report('JC_duration_sample_audit.json')]))\ntable('NYC_extreme_duration_examples.csv')"),
      code("table('NYC_missing_values.csv')\ndisplay(pd.DataFrame([report('NYC_sample_duplicates.json')]))"),
      md('## 6. Exploratory data analysis — EXPLORE\n\nIedere grafiek beantwoordt een vraag. Hieronder gebruiken we uitsluitend ontwikkelingsdata vóór mei 2026; de tijdprofielen gebruiken recente ontwikkelingsjaren. Rijtellingen tellen geregistreerd gebruik en meten geen vraag die niet kon worden bediend.'),
      code("show('temporal')"),code("show('hour_weekday')"),code("show('duration')\ntable('NYC_duration_statistics.csv')"),code("show('user_bike')\ntable('NYC_user_bike.csv')"),
      md('## 7. Aggregaties en geografie\n\nStations en routes worden vóór visualisatie geaggregeerd. Historische stationnamen kunnen wijzigen. Instroom/uitstroom telt ritten en meet geen actuele voorraad of herbalancering. Een geografisch plotvenster is geen cleaningregel. Jersey City wordt hieronder apart getoond, nooit bij NYC opgeteld.'),
      code("show('stations')"),code("show('routes')"),code("table('NYC_inflow_outflow.csv')"),code("show('spatial')"),
      code("show('temporal', area='JC')\ntable('JC_user_bike.csv')"),
      md('## 8. Statistisch onderzoek\n\nBij miljoenen ritten kan een klein effect significant zijn. We rapporteren omvang en onzekerheid. Herhaalde tijdspatronen worden onderzocht met ACF; trend en seizoen kunnen correlatie verhogen. Stationariteit is een diagnose, geen verplichte voorwaarde voor iedere regressiemethode.'),
      code("show('acf')\nif (REPORTS/'time_diagnostics.json').exists():\n    display(pd.DataFrame([report('time_diagnostics.json')]))"),
      md('## 9. Toetsbare hypothesen\n\nDe toetsfamilie bestaat uit ochtendrit-aandeel werkdag/weekend, associatie met het vorige weekuur en casual/member-ritduur. De protocolfile is geschreven vóór de berekeningen. Wekelijkse effecten worden in vierweekblokken gegroepeerd; achtweekblokken dienen als sensitiviteitscontrole. Holm-Bonferroni corrigeert de drie toetsen. Sign-flip vereist symmetrie onder H0 en voldoende onafhankelijke blokken.'),
      md('De vragen zijn geïnformeerd door de EDA: deze toetsen zijn exploratief, geen onafhankelijk bevestigend onderzoek. De 95%-intervallen gelden per hypothese; Holm corrigeert de p-waarden, niet deze intervallen. De latere tijdsvalidatie moet de praktische voorspelwaarde aantonen.'),
      code("h = report('hypotheses.json')\ndisplay(pd.DataFrame([{k:r[k] for k in ['label','H0','H1','statistic','unit','p_value','holm_p_value','ci95','blocks','reject_H0']} for r in h]))\nfor r in h:\n    display(Markdown(f\"**{r['label']}**: effect {r['effect_size']:.3f} {r['unit']}; Holm-p {r['holm_p_value']:.6g}. H0 {'verworpen' if r['reject_H0'] else 'niet verworpen'}. Achtweek-sensitiviteit: {r['sensitivity_8weeks']}. Dit is associatie, geen causaliteit.\"))"),
      md('## 10. Voorspelprobleem kiezen\n\nVóór feature engineering leggen we per kandidaat het voorspeltijdstip en de horizon vast. Groepsverschillen in duur bewijzen geen voorspelbaarheid van individuele ritten. De keuze voor uurvraag vereist een praktisch sterke wekelijkse associatie; validatie moet later aantonen of het model echt helpt.'),
      code("display(pd.DataFrame(report('prediction_contracts.json')['contracts']))\ndisplay(pd.DataFrame(report('target_comparison.json')))\ndecision = report('target_decision.json')\ndisplay(Markdown('**Chosen prediction problem:** ' + decision['chosen_prediction_problem'] + '\\n\\n' + decision['reason']))"),
      md('## 11. Feature engineering\n\nHet volledige dagprofiel wordt om 00:00 voorspeld. Daarom gebruiken we dezelfde uren op afgeronde voorgaande dagen, niet het vorige uur later op de voorspeldag. Cyclische encodings maken 23:00 en 00:00 naburige uren. Rolling gemiddelden worden eerst één dag verschoven.\n\n### Data Leakage Audit\n\nEindstation, eindtijd en duur van toekomstige ritten zijn uitgesloten. Historische tellingen zijn pas toegestaan nadat die dag voorbij is.'),
      code("display(pd.DataFrame(report('leakage_audit.json')))\nfrom citibike.features import make_features, FEATURES\nhistory = pd.read_parquet(DATA/'NYC_hourly_development.parquet')\nfeatures = make_features(history)\ndisplay(features[['timestamp','rides']+FEATURES].tail(6))"),
      md('## 12. Train / validatie / test\n\nTraining ligt vóór mei 2026; validatie is mei–juni. De vijf chronologische folds bestaan uit 28 volledige validatiedagen en voorafgaande training. PyCaret gebruikt exact dezelfde opgeslagen indexgrenzen, zonder random CV, shuffling of vooraf geleerde globale preprocessing. Juli–augustus blijft gesloten tot de selectie en het artefact zijn vergrendeld.'),
      code("split = report('chronological_splits.json')\ndisplay(pd.DataFrame(split['folds']))\ndisplay(Markdown(f\"Train: {split['train_start']} tot {split['train_end_exclusive']} exclusief ({split['train_rows']:,} rijen). Validatie: {split['validation_rows']:,} rijen. Foldfingerprint: `{split['fold_fingerprint']}`.\"))"),
      md('## 13. Baseline en Ridge\n\nDe seizoensbaseline neemt hetzelfde lokale uur van de vorige week. Ridge gebruikt een StandardScaler die telkens binnen de trainingsfold wordt gefit. Alle modellen gebruiken dezelfde niet-negatieve voorspellingen en zomertijdregels.'),
      code("comparison = pd.read_csv(REPORTS/'model_comparison.csv')\ndisplay(comparison[comparison.name.isin(['seasonal_naive','ridge'])])"),
      md('## 14. Kandidaatmodellen en AutoML\n\nRandom Forest en LightGBM kunnen niet-lineaire interacties leren. We beperken de shortlist tot schaalbare, uitlegbare modellen. PyCaret vergelijkt dezelfde operationele estimators; het bepaalt geen random holdout.'),
      code("table('pycaret_comparison.csv')\ndisplay(pd.DataFrame([report('pycaret_split_audit.json')]).drop(columns='folds'))"),
      md('## 15. Modelvergelijking\n\nRMSE is de hoofdmetriek en blijft in ritten per uur. MAE is de typische absolute fout; R² geeft aanvullende context. Het verschil tussen train en validatie helpt overfit beoordelen.'),code("show('comparison')\ntable('model_comparison.csv')"),
      md('## 16. Modelverbetering\n\nHistorische trainingsvensters van twee, vijf en alle jaren zijn eerst met trainings-CV vergeleken. De twee sterkste kandidaten krijgen maximaal twintig configuraties. Feature-ablaties onderzoeken kalender-only en het verwijderen van de 14-dagenlag. Geen zoekstap gebruikt de eindtest.'),
      code("investigation=report('validation_error_investigation.json')\ndisplay(pd.DataFrame([{k:v for k,v in investigation.items() if k not in ['hour_MAE','weekday_MAE','largest_errors']}]))\ndisplay(pd.DataFrame(investigation['largest_errors']))"),
      code("display(pd.DataFrame(report('training_windows.json')).drop(columns='folds'))\nfor path in sorted(REPORTS.glob('*_tuning.csv')):\n    display(Markdown('**'+path.stem+'**'))\n    display(pd.read_csv(path)[['configuration','cv_RMSE','parameters']])"),
      code("display(pd.DataFrame([report('tuning_protocol.json')]))"),
      md('## 17. Validatiefouten en residuen\n\nResidueel = werkelijk − voorspeld. Systematische patronen tonen wat nog niet gevangen is. De ablaties zijn ontwikkelingsvergelijkingen; ze geven geen recht om na de eindtest verder te tunen.'),code("show('residual_validation')\ntable('validation_errors_by_weekday.csv')\ntable('validation_errors_by_season.csv')"),
      md('## 18. Definitieve selectie en eenmalige eindtest\n\nDe kleinste validatie-RMSE wint; binnen 1% kiezen we het eenvoudigere model. Daarna wordt opnieuw gefit op geschikte data tot eind juni. Het vaste model voorspelt iedere testdag met alleen op dat moment bekende historie. Testdiagnostiek is uitsluitend een eindrapport.'),
      md('De vooraf vastgelegde eenvoudigheidsrang betreft modeltypen: weekbaseline → Ridge → Random Forest → LightGBM. Binnen dezelfde familie kiest de laagste validatie-RMSE. Het aantal bomen en de diepte krijgen geen afzonderlijke complexiteitsrang. De RF-tuning levert slechts circa 0,27% extra validatiewinst tegenover de oorspronkelijke RF; het grotere uiteindelijke artefact is circa 647 MiB. Die praktische kosten moeten naast de score worden beoordeeld.'),
      code("selection = report('model_selection.json')\nfinal = report('final_evaluation.json')\ndisplay(pd.DataFrame([{'model':final['selected_model'],**final['metrics']},{'model':'seasonal_naive',**final['baseline_metrics']}]))\ndisplay(Markdown(f\"Eindtest-RMSE-verbetering: **{final['RMSE_improvement_percent']:.2f}%**. MSE-verschil, weekbootstrap-95%-CI: {final['MSE_gain_week_bootstrap_ci95']}. {final['ci_limitation']}\"))"),
      code("show('test_forecast')\nshow('residual_test')"),
      md('## 19. Interpretatie — INTERPRET\n\nPermutation importance wordt op validatie berekend. Correlatie tussen features en onrealistische permutaties beperken de interpretatie. Weer, capaciteit en evenementen ontbreken; geen feature importance bewijst causaliteit.'),code("show('importance')"),
      md('## 20. Deployment — DEPLOY\n\nDe lokale Streamlit-app laadt het opgeslagen artefact zonder training. De gedeelde voorspelfunctie controleert NYC, volledige historie, tijdzoneconventie en afwezigheid van toekomstige tellingen. Het historische voorbeeld hieronder is een functiedemonstratie op ontwikkeling, geen onafhankelijke prestatietest.'),
      code("from citibike.inference import predict_day\nday = history.timestamp.max().normalize()\nprediction = predict_day(day, history[history.timestamp < day])\ndisplay(prediction)"),
      md('## 21. Conclusies, beperkingen en reproduceerbaarheid\n\nAlle getallen komen uit uitvoer. Het model voorspelt geregistreerd gebruik, geen onvervulde vraag. Twee zomermaanden testen geen volledige toekomstige seizoenscyclus. Schema’s, stations en systeemgebruik veranderen. AWS-training is nog een afzonderlijke schoolopdracht. Het model staat lokaal buiten Git; bronmanifest, scripts, dependencies en seeds maken hertraining mogelijk.'),
      code("display(Markdown((REPORTS/'MODEL_CARD.md').read_text(encoding='utf-8')))\nverification = report('verification.json')\ndisplay(pd.DataFrame([verification]))\ndisplay(pd.DataFrame([report('test_results.json')]).drop(columns='test_names'))"),
    ]


def build():
    directory=ROOT/'notebooks';directory.mkdir(parents=True,exist_ok=True)
    specs={
      'citybike_analysis.ipynb':overview_cells(),
      '01_EDA_en_hypothesen.ipynb':[
          md('# 01 — EDA en hypothesen\n\nBijdragen: door team in te vullen. Dit notebook toont uitsluitend ontwikkeling; volledige uitleg staat in citybike_analysis.'),code(INIT),
          md('NYC-vragen: tijd, ritduur, categorieën, stations, routes en geografie. Iedere interpretatie wordt uit de berekende uitvoer gemaakt.'),
          *[code(f"show('{kind}')") for kind in ['temporal','hour_weekday','duration','user_bike','stations','routes','spatial','acf']],
          md('Jersey City apart; geen optelling met NYC.'),code("show('temporal','JC')"),
          md('Geblokte hypothesen, effectgroottes, onzekerheid en Holm-correctie.'),code("display(pd.DataFrame(report('hypotheses.json'))[['label','effect_size','unit','ci95','p_value','holm_p_value','reject_H0']])"),
          code("display(pd.DataFrame(report('prediction_contracts.json')['contracts']))\ndisplay(pd.DataFrame(report('target_comparison.json')))"),
      ],
      '02_definitieve_voorbereiding.ipynb':[
          md('# 02 — Definitieve voorbereiding zonder grafieken\n\nBijdragen: door team in te vullen. `run.py prepare` verwerkt de bronnen geheel met code. Cachevalidatie gebruikt bron-, code- en partitionfingerprints. Voor een herbouw verwijder je alleen de gegenereerde data volgens README; de ruwe bronnen blijven behouden.'),code(INIT),
          md('De voorbereiding is uitgevoerd vóór dit notebook. Het manifest legt de volledige reproducerbare uitvoering vast.'),code("m=manifest()\ndisplay(pd.DataFrame(m['partitions'])[['area','period','raw_rows','retained_rows','removed_rows','removed_percent','processing_seconds']])"),
          md('Voorspeltijdstip en horizon worden vóór feature engineering getoond.'),code("display(pd.DataFrame(report('prediction_contracts.json')['contracts']))"),
          md('Het dagvooruitcontract laat uitsluitend afgeronde dagen als featurebron toe. De doelkeuze ligt al vast op ontwikkelingsdata.'),code("from citibike.features import make_features, FEATURES\nhistory=pd.read_parquet(DATA/'NYC_hourly_development.parquet')\nfeatures=make_features(history)\ndisplay(features.tail(8))"),
          code("display(pd.DataFrame(report('leakage_audit.json')))\ndisplay(pd.DataFrame(report('chronological_splits.json')['folds']))"),
      ],
      '03_baseline_en_Ridge.ipynb':model_cells('Baseline en Ridge',['seasonal_naive','ridge','ridge_calendar','ridge_no_14d']),
      '04_PyCaret_vergelijking.ipynb':[
          md('# 04 — PyCaret met dezelfde chronologische folds\n\nBijdragen: door team in te vullen. De setup gebruikt expliciete validatie, opgeslagen folds en preprocess=False. Ridge schaalt binnen de eigen fold-pipeline. Alle estimators volgen dezelfde niet-negatieve en zomertijdregels.'),code(INIT),
          md('De uitgevoerde setup in `citibike.modeling.py` gebruikt onderstaande instellingen. `compare_models` ontvangt de vier gedeelde estimators, inclusief de operationele clipping en zomertijdregel. De audit controleert de daadwerkelijke interne train- en validatierijen.\n\n```python\nexperiment.setup(data=train[FEATURES+[\"rides\"]],\n    test_data=validation[FEATURES+[\"rides\"]], target=\"rides\",\n    fold_strategy=SavedChronologicalCV(bounds), fold=len(bounds),\n    data_split_shuffle=False, fold_shuffle=False, preprocess=False,\n    session_id=42, n_jobs=2, index=False)\n```'),
          code("table('pycaret_comparison.csv')\ndisplay(pd.DataFrame(report('pycaret_split_audit.json')['folds']))"),
          code("audit=report('pycaret_split_audit.json')\ndisplay(pd.DataFrame([{k:v for k,v in audit.items() if k!='folds'}]))"),
          md('PyCaret verkleint numerieke dtypes bij de setup. Daardoor kunnen afronding en de binning van LightGBM iets verschillen, ondanks dezelfde informatie en folds. De onderstaande tabel toont het gemeten verschil. De uiteindelijke selectie gebruikt de validatiemetrics van de handmatige, gedeelde inference-pipeline; AutoML is een aanvullende vergelijking.'),
          code("manual=pd.DataFrame(report('model_comparison.json'))[['name','cv_RMSE']]\nautoml=pd.read_csv(REPORTS/'pycaret_comparison.csv')\nprecision=automl[['candidate','RMSE']].merge(manual,left_on='candidate',right_on='name')\nprecision['CV_RMSE_verschil']=precision.RMSE-precision.cv_RMSE\ndisplay(precision)"),
      ],
      '05_Random_Forest.ipynb':model_cells('Random Forest',['random_forest','tuned_random_forest']),
      '06_LightGBM.ipynb':model_cells('LightGBM',['lightgbm','tuned_lightgbm']),
      '07_eindvergelijking.ipynb':[
          md('# 07 — Definitieve modelvergelijking en eindtest\n\nBijdragen: door team in te vullen. De modelkeuze lag vóór opening van de eindtest vast. Dit notebook toont bestaande evaluatie-uitvoer en voert geen nieuwe tuning uit.'),code(INIT),
          code("show('comparison')\ndisplay(pd.DataFrame([report('model_selection.json')]).drop(columns=['parameters','modeling_code_version']))"),
          code("display(pd.DataFrame([report('final_evaluation.json')]))\nshow('test_forecast')\nshow('residual_test')"),
          code("show('importance')\ndisplay(Markdown((REPORTS/'MODEL_CARD.md').read_text(encoding='utf-8')))"),
      ],
    }
    for name,cells in specs.items():
        notebook=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Citi Bike full (.venv)','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}})
        nbf.write(notebook,directory/name)
    return list(specs)


def model_cells(title,names):
    return [md(f'# {title}\n\nBijdragen: door team in te vullen. De training gebeurt offline via `run.py train`. Dit notebook bewaart echte scores en configuraties; geen eindtest wordt voor tuning gebruikt.'),code(INIT),
            md('De kern van iedere trainingsstap is hieronder weergegeven. De CLI herhaalt dit voor dezelfde opgeslagen folds, fit preprocessing alleen op de trainingsrijen en bewaart de werkelijke scores. Dit rapportnotebook voert na de eindtest geen nieuwe selectie uit.\n\n```python\nfitted = clone(candidate).fit(training[FEATURES], training.rides)\npredicted = fitted.predict(validation[FEATURES])\nscores = metrics(validation.rides, predicted)\n```'),
            code("from citibike.modeling import candidates\nfrom citibike.features import FEATURES\nimport joblib\n"+f"for name in {names!r}:\n    if (MODELS/(name+'.joblib')).exists():\n        model=joblib.load(MODELS/(name+'.joblib'))\n        display(Markdown('**Opgeslagen estimator: '+name+'**'))\n        display(Markdown(str(model.estimator_)))"),
            md('Dezelfde chronologische validatie en operationele voorspelfunctie maken de vergelijking eerlijk.'),
            code(f"comparison=pd.read_csv(REPORTS/'model_comparison.csv')\ndisplay(comparison[comparison.name.isin({names!r})])"),
            md('Een lagere trainingsfout kan overfit betekenen. Kies op validatie en bekijk de CV-spreiding en looptijd.'),
            code(f"records=report('model_comparison.json')\nfor record in records:\n    if record['name'] in {names!r}:\n        display(Markdown('**'+record['name']+'**'))\n        display(pd.DataFrame(record['cv']))\n        display(Markdown(str(record['parameters'])))"),
            code(f"for path in sorted(REPORTS.glob('*_tuning.csv')):\n    if path.stem.split('_tuning')[0] in {sorted({name.replace('tuned_','') for name in names})!r}:\n        display(pd.read_csv(path)[['configuration','cv_RMSE','parameters']])"),
           ]


def execute():
    required=['data_manifest.json','target_decision.json','model_selection.json','final_evaluation.json','verification.json']
    for name in required:
        if not (REPORTS/name).exists():raise FileNotFoundError(name)
    names=build()
    for name in names:
        path=ROOT/'notebooks'/name;notebook=nbf.read(path,as_version=4)
        # Kernel wordt expliciet in de projectomgeving gestart; geen afhankelijkheid van globale kernelregistratie.
        from jupyter_client import KernelManager
        manager=KernelManager(kernel_name='python3')
        manager.kernel_spec.argv=[sys.executable,'-X','utf8','-m','ipykernel_launcher','-f','{connection_file}']
        client=NotebookClient(notebook,timeout=600,kernel_name='python3',km=manager,
                              resources={'metadata':{'path':str(ROOT)}})
        try:
            client.execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        nbf.write(notebook,path)
        print(f'Notebook uitgevoerd: {name}',flush=True)
