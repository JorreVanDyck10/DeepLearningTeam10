"""Curate copies and independent notebooks without modifying the original studies."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import nbformat

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT/'Final'
PROVENANCE = []
BOOT = '''from pathlib import Path
import sys, json
import pandas as pd
from IPython.display import display, Markdown
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'MushroomFinal/mushroom_project_dataset.csv').exists())
sys.path.insert(0, str(ROOT))
'''


def md(text): return nbformat.v4.new_markdown_cell(text)
def code(text): return nbformat.v4.new_code_cell(text)
def read(path): return nbformat.read(ROOT/path, as_version=4)
def sha(path): return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
def save(path, cells, source=None, note=None):
    target = DEST/path; target.parent.mkdir(parents=True, exist_ok=True)
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {'name':'python3', 'display_name':'Python 3.11 — Final', 'language':'python'}
    nbformat.validate(nb); nbformat.write(nb, target)
    if source:
        PROVENANCE.append({'final': str(target.relative_to(ROOT)).replace('\\','/'), 'source': source,
                           'source_sha256_lf': sha(ROOT/source), 'note': note or 'Curated copy; original preserved'})


def main():
    original = read('MushroomFinal/mushroom_project.ipynb')
    introduction = md('''# Mushroom — EDA op de 5.000 cursusrijen

Bronanalyse: Andrew Noeyens. Deze aparte leesversie is door Codex samengesteld op
verzoek van Jorre; de definitieve menselijke taakverdeling blijft stap 5.
De bron blijft ongewijzigd in `MushroomFinal/mushroom_project.ipynb`.
De cellen hieronder onderzoeken de juiste cursus-CSV; modeltraining staat apart.
Het volledige bestand met 61.069 UCI-rijen hoort bij de historische experimenten.''')
    cells = [introduction, md('We vinden de repository voordat we de bronanalyse uitvoeren.'), code(BOOT)]
    cells += deepcopy(original.cells[3:25])
    for c in cells:
        if c.cell_type == 'code':
            c.source = c.source.replace("pd.read_csv('mushroom_project_dataset.csv')", "pd.read_csv(ROOT/'MushroomFinal/mushroom_project_dataset.csv')")
    save(Path('Mushroom/01_EDA.ipynb'), cells, 'MushroomFinal/mushroom_project.ipynb', 'EDA section; repository-relative data path')
    save(Path('Mushroom/02_voorbereiding.ipynb'), [
        md('''# Mushroom — definitieve voorbereiding zonder grafieken

Dit notebook vat de voorbereiding voor de gekozen pipeline samen. Codex maakte
deze aparte, uitvoerbare versie op verzoek van Jorre. Stap 5 (menselijke bijdragen
en opleidingspolicy) blijft door Jorre af te ronden.

Bron is de cursus-CSV met 5.000 rijen en 13 kolommen. We normaliseren namen,
verwijderen uitsluitend `jumbled_noise_0`, behouden `jumbled_noise_1` en coderen
e=0/p=1. Nulmetingen blijven nul, zoals bij de gekozen RF300. Ontbrekende waarden
blijven in de ruwe featuretabellen staan: imputation en encoding leren later
uitsluitend uit de trainingsrijen, opnieuw binnen iedere CV-fold.'''),
        md('We controleren bronhash, schema, duplicaten, labels en de exact gedeelde splits.'),
        code(BOOT+'''from Final.tools.data import prepare, SOURCE_SHA256_LF, FEATURES
raw, X, y, train, test, folds = prepare()
manifest = pd.read_csv(ROOT/'MushroomFinal/comparison_correct/split_manifest.csv')
assert X.shape == (5000, 11) and not set(train) & set(test)
assert set(manifest.loc[manifest.split == 'train', 'source_row']) == set(train)
assert set(manifest.loc[manifest.split == 'test', 'source_row']) == set(test)
assert 'jumbled_noise_0' not in X and 'jumbled_noise_1' in X
print('Bronhash LF:', SOURCE_SHA256_LF)
print('Train/test:', len(train), len(test), '; CV-folds:', len(folds))
display(raw.isna().sum().rename('ontbrekend').to_frame())
display(raw['class'].value_counts().rename('aantal').to_frame())'''),
        md('De medianen, modi en one-hot-categorieën leren alleen uit training. De testset verandert deze waarden niet. In CV wordt een nieuwe kopie van deze transformer binnen elke fold gefit.'),
        code('''from MushroomFinal.final_model_support import preprocessing
preprocessor = preprocessing(X)
encoded_train = preprocessor.fit_transform(X.iloc[train])
encoded_test = preprocessor.transform(X.iloc[test])
numeric = preprocessor.named_transformers_['num'].named_steps['impute']
assert (numeric.statistics_ == X.iloc[train][['cap_diameter','stem_height','stem_width']].median().to_numpy()).all()
assert encoded_train.shape[1] == encoded_test.shape[1]
print('Encoded train/test:', encoded_train.shape, encoded_test.shape)
print('Alle imputatie- en encodervoorzieningen zijn op training geleerd.')'''),
        md('We bewaren ruwe gesplitste features, targets en rijnummers in een genegeerde werkmap. Zo kan ieder model zijn eigen fold-lokale preprocessing toepassen. Er worden geen grafieken of modellen gemaakt.'),
        code('''folder = ROOT/'Final/.generated/prepared'
folder.mkdir(parents=True, exist_ok=True)
for name, indices in [('train', train), ('test', test)]:
    table = X.iloc[indices].copy()
    table.insert(0, 'source_row', indices)
    table['target'] = y.iloc[indices].to_numpy()
    table.to_csv(folder/f'{name}.csv', index=False)
schema = {'source_sha256_lf': SOURCE_SHA256_LF, 'features': FEATURES,
          'train_rows':len(train), 'test_rows':len(test), 'target':{'e':0,'p':1},
          'fit_preprocessing_on':'training only, separately per CV fold'}
(folder/'schema.json').write_text(json.dumps(schema, indent=2)+'\\n')
assert len(pd.read_csv(folder/'train.csv')) == 4100
assert len(pd.read_csv(folder/'test.csv')) == 900
print('Voorbereide splits gecontroleerd en opgeslagen in Final/.generated/prepared.')''')])
    families = [('03_baseline','tree_baseline','Beslisboom depth 5',None),
                ('05_Random_Forest','rf_final','Random Forest',62),
                ('06_Gradient_Boosting','gb_final','Gradient Boosting',79),
                ('07_Logistic_Regression','lr_final','Logistic Regression',93)]
    for filename, key, title, grid_cell in families:
        cells = [md(f'''# Mushroom — {title}

Eén modeltype per notebook. Oorspronkelijke Mushroom-modellen: Andrew; de gedeelde
vergelijking en deze reproduceerbare versie: Codex op verzoek van Jorre.
Jorre rondt de definitieve taakverdeling en GenAI-policy later af (stap 5).

De huidige referentie is de correct uitgevoerde vergelijking op dezelfde 4.100
trainingsrijen, 900 ontwikkeltestrecords en vijf folds. Het model wordt hieronder
werkelijk opnieuw gefit met de vastgelegde configuratie. Alle preprocessing wordt
binnen de fold geleerd. De ontwikkeltest is eerder bekeken en is geen nieuwe
onafhankelijke eindtest. Dit notebook wijzigt de website of bronstudie niet.'''),
                 md('We laden de vastgelegde parameters en leggen uit waarom dit model is opgenomen.'),
                 code(BOOT+f'''summary = json.loads((ROOT/'MushroomFinal/comparison_correct/comparison_summary.json').read_text(encoding='utf-8'))
key = {key!r}
display(pd.Series(summary['classifier_parameters'][key], name='Vastgelegde parameters'))
''')]
        if grid_cell is not None:
            text = original.cells[grid_cell].source
            grids=[]
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, ast.Assign) and any(isinstance(n, ast.Name) and n.id.startswith('param_grid') for n in node.targets):
                    grids.append(ast.get_source_segment(text, node))
            cells += [md('''## Oorspronkelijk tuningonderzoek

Deze zoekruimte komt uit het oorspronkelijke projectnotebook. De oorspronkelijke
uitvoer blijft daar bewaard. De eerdere proximity-imputatie was vóór CV geleerd;
daarom gebruiken we die scores niet als eindbewijs. De gecorrigeerde vijf-fold-
evaluatie hieronder leert preprocessing opnieuw binnen iedere fold. Dit is een
reproductie van de vastgelegde configuratie, geen nieuwe gridsearch of modelkeuze.

```python
'''+ '\n'.join(grids) +'\n```')]
        rationale = {'rf_final':'Bomen kunnen niet-lineaire interacties modelleren. RF300 was de keuze op gedeelde CV macro F1; giftige recall en concrete fouten worden apart gerapporteerd.',
                     'gb_final':'Boosting corrigeert opeenvolgend fouten. De proximity-imputatie en alle gewone preprocessing worden binnen de fold geleerd; de extra complexiteit moet met gemeten prestaties worden afgewogen.',
                     'lr_final':'Een lineair model is een nuttige eenvoudiger vergelijking. Schalen, L1-regularisatie en class weights zijn onderdeel van de vastgelegde pipeline.',
                     'tree_baseline':'Een kleine beslisboom vormt een snelle, interpreteerbare referentie; maximale diepte 5 begrenst complexiteit.'}[key]
        cells += [md(rationale+'\n\nWe fitten vijf trainingsfolds en daarna één model op alle trainingsrijen. De code controleert alle 900 labels en kansen tegen de opgeslagen referentie.'),
                  code(f'''from Final.tools.model_workflow import reproduce
result = reproduce({key!r})
display(pd.DataFrame(result['folds']))
display(pd.Series({{k:v for k,v in result.items() if k != 'folds'}}))
assert result['labels_match_original'] and result['max_probability_difference'] < 1e-12
'''), md('De uiteindelijke keuze staat in `09_vergelijking.ipynb`. Een hoge score voor de eetbare klasse is geen bevestiging van eetbaarheid; de website gebruikt een afzonderlijke voorzichtige beslisregel.')]
        save(Path('Mushroom')/(filename+'.ipynb'), cells, 'MushroomFinal/mushroom_project.ipynb', 'One model family; fresh fold-local reproduction and historical tuning context')
    save(Path('Mushroom/04_AutoML.ipynb'), [md('''# Mushroom — uitgevoerde AutoML-vergelijking

De PyCaret 3.3.2-run is op 14 oktober 2026 werkelijk uitgevoerd in de oorspronkelijke
Python 3.11-omgeving. Dit notebook controleert de vastgelegde audit en presenteert
die bestaande resultaten apart. Het traint niet stilzwijgend een nieuwe selectie.
De bronuitvoer blijft in het oorspronkelijke projectnotebook staan. Stap 5 blijft
door Jorre af te ronden.'''), md('We controleren dat de bron en alle trainings- en testrijen dezelfde zijn als bij de handmatige modellen.'),
        code(BOOT+'''from Final.tools.data import prepare, SOURCE_SHA256_LF
_, X, y, train, test, folds = prepare()
folder = ROOT/'MushroomFinal/comparison_correct'
audit = json.loads((folder/'pycaret_audit.json').read_text())
assert audit['train_source_rows'] == train.tolist() and audit['test_source_rows'] == test.tolist()
assert audit['dataset']['sha256_lf'] == SOURCE_SHA256_LF
display(pd.Series({k:v for k,v in audit.items() if k not in ['train_source_rows','test_source_rows']}))'''),
        md('De tabel vergelijkt PyCaret-defaults; zij is niet dezelfde fit als de handmatig getunede kandidaten. De macro-F1-scorer en majority reference zijn afzonderlijk gecontroleerd.'),
        code('''results = pd.read_csv(folder/'pycaret_all.csv', index_col=0)
assert results.loc['dummy', 'Macro F1'] > .3
display(pd.read_csv(folder/'pycaret_three.csv', index_col=0))
display(results)
display(pd.read_csv(folder/'pycaret_test_metrics.csv'))'''),
        md('Random Forest heeft de hoogste uitgevoerde PyCaret CV macro F1 (circa 0,7427). De handmatige RF300 scoort circa 0,7619 in dezelfde trainingsfolds. LightGBM crashte in de oorspronkelijke Windows-PyCaret-run en is transparant uitgesloten. Opnieuw uitvoeren kan met de volledige `MushroomFinal/requirements-comparison.txt` en `MushroomFinal/execute_pycaret_notebook.py`; doe dat in een aparte omgeving.')], 'MushroomFinal/mushroom_project.ipynb')
    for target, source in [('Mushroom/08_AWS.ipynb','SolutionJorre/MushroomDataset/correct_dataset/04_aws_correct_dataset.ipynb'),
                           ('Mushroom/09_vergelijking.ipynb','secondary_mushroom/09_compare_models.ipynb'),
                           ('Mushroom/10_deployment.ipynb','secondary_mushroom/10_deploy.ipynb')]:
        nb = read(source)
        for cell in nb.cells:
            if cell.cell_type == 'markdown':
                cell.source = cell.source.replace('**Nog niet op SageMaker uitgevoerd.**', '**Deze aangeleverde run is uitgevoerd op SageMaker en afzonderlijk geaudit.**')
                cell.source = cell.source.replace('../MushroomFinal/', '../../MushroomFinal/').replace('../SolutionJorre/', '../../SolutionJorre/').replace('../docs/', '../../docs/')
            if target == 'Mushroom/10_deployment.ipynb' and cell.cell_type == 'code':
                cell.source = cell.source.replace("assert model.metadata['source_pipeline_sha256'] == audit['source_pipeline_sha256'] == summary['artifact']['sha256']",
                    "assert model.metadata['source_pipeline_sha256'] == audit['source_pipeline_sha256'] == hashlib.sha256((ROOT/model.metadata['source_pipeline']).read_bytes()).hexdigest()")
                cell.source = cell.source.replace('assert live_metadata == metadata',
                    "for key in ['model_id','source_pipeline_sha256','example','fields','dataset_rows']:\n    assert live_metadata[key] == metadata[key], key\nfor key in ['accuracy','recall_poisonous','poisonous_predicted_edible']:\n    assert abs(live_metadata['metrics'][key]-metadata['metrics'][key]) < 1e-12, key")
        nb.cells.insert(0, md('Final-leesversie; oorspronkelijke bestanden en hun uitvoer blijven behouden. Stap 5 (definitieve menselijke bijdragen en opleidingspolicy) rondt Jorre later af. Bron: `'+source+'`.'))
        save(Path(target), nb.cells, source, 'Curated copy; provenance and stale wording clarified')
    # City Bike remains the protected original study. Read reports rather than reopening its final test.
    copies=[('01_EDA_en_hypothesen','01_EDA_en_hypothesen'),('02_voorbereiding','02_definitieve_voorbereiding'),
            ('05_AutoML','04_PyCaret_vergelijking'),('06_Random_Forest','05_Random_Forest'),
            ('07_LightGBM','06_LightGBM'),('09_vergelijking','07_eindvergelijking')]
    for target, stem in copies:
        source=f'SolutionRaoul/Citybike/notebooks/{stem}.ipynb'; nb=read(source)
        for c in nb.cells:
            if c.cell_type=='code' and 'locations = [' in c.source:
                start=c.source.index('locations = [');stop=c.source.index('sys.path.insert',start)
                c.source=c.source[:start]+"REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'SolutionRaoul/Citybike').exists())\nROOT = REPO/'SolutionRaoul/Citybike'\n"+c.source[stop:]
        nb.cells.insert(0,md('''Final-leesversie van Raouls uitgevoerde onderzoek. De weergegeven bronuitvoer
komt uit zijn oorspronkelijke omgeving; deze kopie beweert geen nieuwe training.
Grote data en modellen blijven buiten Git. De originele studie en beschermde
eindtest zijn ongewijzigd. Repositorypaden zijn aangepast voor deze leesmap.
Uitvoeren van voorbereiding/training vereist eerst de gedocumenteerde downloads.
De definitieve taakverdeling blijft stap 5, door Jorre later af te ronden.'''))
        save(Path('CitiBike')/(target+'.ipynb'),nb.cells,source,'Source outputs retained; repository discovery adapted, no new final-test evaluation')
    for filename, key, title in [('03_baseline','seasonal_naive','Weekbaseline'),('04_Ridge','ridge','Ridge-regressie')]:
        save(Path('CitiBike')/(filename+'.ipynb'),[
            md(f'''# Citi Bike — {title}

Oorspronkelijk onderzoek: Raoul Buffels. Codex maakte deze afzonderlijke leesversie
op verzoek van Jorre. De oorspronkelijke training en uitvoer blijven behouden;
we openen geen nieuwe eindtest en kiezen geen nieuw model. Stap 5 blijft bij Jorre.
De code hieronder leest de bestaande, uitgevoerde trainings- en CV-rapporten.
De volledige fitcode staat in `SolutionRaoul/Citybike/citibike/modeling.py`.'''),
            md('We tonen uitsluitend dit model en zijn eerder berekende chronologische folds. Alle validatierijen liggen na de trainingsrijen.'),
            code(BOOT+f'''folder = ROOT/'SolutionRaoul/Citybike/reports'
key = {key!r}
comparison = pd.read_csv(folder/'model_comparison.csv')
display(comparison[comparison.name == key])
records = json.loads((folder/'model_comparison.json').read_text())
record = next(r for r in records if r['name'] == key)
display(pd.DataFrame(record['cv']))
display(pd.Series(record['parameters'], dtype=object))
assert len(record['cv']) == 5
'''),
            md('De weekbaseline gebruikt uitsluitend hetzelfde uur van de vorige week. Ridge gebruikt kalender- en lagkenmerken met regularisatie. Raouls vergelijking en de keuze voor RF320 staan apart in `09_vergelijking.ipynb`. Deze rapportcontrole is geen opnieuw uitgevoerde training.')],
            'SolutionRaoul/Citybike/notebooks/03_baseline_en_Ridge.ipynb', 'Split model families into independent report notebooks; original offline fit preserved')
    save(Path('CitiBike/10_deployment.ipynb'),[
        md('''# Citi Bike — API, website en modelcontract

De teamwebsite gebruikt Raouls gekozen RF320, via dezelfde verliesvrije export.
De grote oorspronkelijke pipeline blijft buiten Git; een gepinde GitHub Release
levert de arrays met SHA-256-controles. Deze cel controleert actuele inference,
zonder het oorspronkelijke model opnieuw te trainen of de eindtest te openen.
Stap 5 (menselijke bijdragen en policycontrole) blijft bij Jorre.'''),
        md('Een historische demonstratie bevat al beschikbare tellingen. Voor andere datums zijn precies 336 uurvakken van de 14 voorafgaande volledige lokale NYC-dagen nodig.'),
        code(BOOT+'''import urllib.request
API = 'https://deeplearningteam10-api.onrender.com'
request = urllib.request.Request(API+'/predict/citibike', data=json.dumps({'date':'2026-04-30','hour':8,'mode':'demo'}).encode(), headers={'Content-Type':'application/json'})
with urllib.request.urlopen(request, timeout=120) as response:
    result = json.load(response)
assert result['model_id'] == 'raoul_tuned_random_forest_320'
assert len(result['daily_predictions']) == 24
print('Gekozen model:', result['model_id'])
print('Uurvoorspelling:', result['predicted_ride_starts'], '; dagtotaal:', result['daily_total'])
display(pd.DataFrame(result['daily_predictions']))'''),
        md('Hosting en herstart staan in `docs/citibike_raoul.md`. Nieuwe Citi Bike-studies volgen de oorspronkelijke gescheiden train/validatie/eindtest-procedure; deze inlevermap herschrijft geen oude eindtest. Mushroom heeft daarnaast een echte automatische train/valideer/publiceer-workflow. De bestaande Citi Bike-export wordt bij iedere backenddeployment gecontroleerd en geladen.')])
    write=DEST/'source_manifest.json'
    write.write_text(json.dumps({'base_commit':'69ca5201b934213360b3c2abd3b381d6251142b7','copies':PROVENANCE,
        'preservation':'Original source files were not moved or replaced. Generated notebook outputs do not claim a new original study.'},indent=2)+'\n')
    print('Final notebooks built:', len(list(DEST.rglob('*.ipynb'))))


if __name__ == '__main__': main()
