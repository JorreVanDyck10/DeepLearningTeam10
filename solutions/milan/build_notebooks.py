"""Build and execute report notebooks with ordinary Python, without cloud access.

Each notebook gets a fresh namespace. Cells use print/explicit image output;
stored outputs are captured from real top-to-bottom execution. This is not a
claim of execution through a Jupyter kernel. The notebooks also work in Jupyter.
The costly experiment is run separately by mushroom_alternative.py.
"""
import base64
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import uuid

HERE = Path(__file__).resolve().parent

BOOT = '''from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'MushroomFinal/mushroom_project_dataset.csv').exists())
HERE = ROOT/'solutions/milan'
sys.path.insert(0, str(HERE))
from mushroom_alternative import development_data, pipeline, GRID, scores
X, y, ids, folds = development_data()
'''


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'id': uuid.uuid4().hex[:8], 'source': text}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'id': uuid.uuid4().hex[:8],
            'source': text, 'outputs': [], 'execution_count': None}


def save(name, cells):
    ns = {'__name__': '__notebook__'}
    count = 0
    for cell in cells:
        if cell['cell_type'] != 'code':
            continue
        count += 1
        stream = io.StringIO()
        with redirect_stdout(stream), redirect_stderr(stream):
            exec(compile(cell['source'], str(HERE / name), 'exec'), ns)
        if stream.getvalue():
            cell['outputs'].append({'output_type': 'stream', 'name': 'stdout', 'text': stream.getvalue()})
        cell['execution_count'] = count
    nb = {'nbformat': 4, 'nbformat_minor': 5, 'cells': cells,
          'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3.13 - Milan concept', 'language': 'python'},
            'language_info': {'name': 'python', 'version': '3.13.9'},
            'execution_note': 'Report cells executed top to bottom with Python in a fresh namespace. Training was executed separately; no fresh Jupyter-kernel claim.'}}
    (HERE / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print('Executed report:', name, '-', count, 'code cells')


def main():
    intro = '''**Bijdrage en AI:** Codex maakte dit concept op verzoek van de gebruiker.
De mapnaam is geen bevestiging van Milans persoonlijke auteurschap of review.
De oorspronkelijke teamoplossing blijft behouden. De kritiek en lesverwijzingen
staan in [REVIEW_FINAL.md](REVIEW_FINAL.md).

Dit is ontwikkelonderzoek. De 900 al bekende ontwikkeltestrecords worden voor
dit experiment niet gescoord. De overblijvende projectdata zijn ook eerder
gebruikt; er ontstaat dus geen nieuwe onafhankelijke eindtest.'''
    save('01_voorbereiding.ipynb', [md('# Mushroom — eenvoudige voorbereiding\n\n' + intro),
        md('## Doel en splits\nWe behouden de elf bestaande invoervelden. De oorspronkelijke gedeelde vijf folds worden buitenfolds; modelkeuze gebeurt later binnen hun trainingsdeel. Dit voorbereidingsnotebook bevat geen grafieken en fit geen model.'),
        code(BOOT + '''print('Trainingsrijen:', len(X), '; buitenfolds:', len(folds))
print('Class mapping: e=0, p=1')
print(y.value_counts().sort_index().to_string())
for number, (tr, val) in enumerate(folds, 1):
    assert not set(ids[tr]) & set(ids[val])
    print('Fold', number, ': training', len(tr), '/ validatie', len(val))
'''),
        md('## Alleen aanwezige velden\nDe tabel toont het daadwerkelijke modelinvoercontract en ontbrekende waarden binnen de 4.100 ontwikkelingsrijen. Dit is geen dictionary van de volledige UCI-bron.'),
        code('''dictionary = pd.DataFrame({'type': X.dtypes.astype(str), 'missing': X.isna().sum(), 'missing_pct': X.isna().mean()*100, 'unique_known': X.nunique()})
print(dictionary.round(2).to_string())
assert 'jumbled_noise_0' not in X and 'jumbled_noise_1' in X
'''),
        md('## Voorbereiding binnen iedere fit\nNumeriek: mediaan. Categorisch: modus of de expliciete categorie missing, gevolgd door one-hot encoding. De volledige transformer zit in de pipeline en wordt dus niet globaal vóór CV gefit. Dit experiment onderzoekt geen proximity-imputatie of automatische verwijdering van nullen.'),
        code('''print(pipeline())
print('Nog niet gefit:', not hasattr(pipeline(), 'classes_'))
assert not hasattr(pipeline(), 'classes_')
''')])
    save('02_random_forest.ipynb', [md('# Random Forest — nieuwe tuning binnen training\n\n' + intro),
        md('## De vooraf vastgelegde zoekruimte\nWe zoeken acht combinaties: max_features, min_samples_leaf en de categorische imputatie. De zoekkandidaat heeft 150 bomen om de rekentijd te begrenzen; de referentie behoudt de RF300-instellingen. Beide worden op dezelfde buitenfolds en in dezelfde omgeving opnieuw getraind. Dit is een vergelijking van complete procedures, geen geïsoleerd effect van één parameter.'),
        code(BOOT + '''from sklearn.model_selection import GridSearchCV, StratifiedKFold
import inspect
import mushroom_alternative as experiment
print(json.dumps(GRID, indent=2))
print(inspect.getsource(experiment.pipeline))
inner = StratifiedKFold(3, shuffle=True, random_state=20261016)
search_example = GridSearchCV(pipeline(), GRID, scoring='f1_macro', cv=inner, n_jobs=2, error_score='raise')
print(search_example)
'''),
        md('## Waarom twee niveaus?\nDe binnenfolds kiezen instellingen met macro F1. Alleen buitenvalidatie beoordeelt die selectieprocedure. Ook de drempel voor 90% giftige recall wordt uit binnen-OOF voorspellingen gekozen en vervolgens buiten geëvalueerd. Dit is een strengere toepassing van de geleerde scheiding tussen training, validatie en test. Buitenfolds zijn hier ontwikkelvalidatie, geen nieuwe eindtest.'),
        code('''print(inspect.getsource(experiment.choose_threshold))
selections = json.loads((HERE/'results/selections.json').read_text(encoding='utf-8'))
assert len(selections) == 5
print(pd.json_normalize(selections).to_string(index=False))
'''),
        md('## Werkelijk uitgevoerde grids\nDe tabellen zijn afkomstig van de afzonderlijk uitgevoerde trainingscode. De beste binnenscore is een selectiescore, geen onafhankelijke prestatiemeting.'),
        code('''for fold in range(1, 6):
    grid = pd.read_csv(HERE/f'results/inner_search_fold_{fold}.csv')
    assert len(grid) == 8
    print('Buitenfold:', fold)
    print(grid[['params','mean_train_score','mean_test_score','std_test_score','rank_test_score']].sort_values('rank_test_score').to_string(index=False))
run_receipt = json.loads((HERE/'results/run.json').read_text(encoding='utf-8'))
assert run_receipt['test_rows_scored'] == 0 and run_receipt['outer_evaluations'] == 20
print(json.dumps(run_receipt, indent=2))
'''),
        md('## Optioneel de training opnieuw uitvoeren\nDe rapportcellen zijn al uitgevoerd en lezen bestaande nieuwe resultaten. Zet de vlag hieronder alleen op True als je de volledige search opnieuw wilt trainen. Dat schrijft uitsluitend resultaten in deze conceptmap en duurt enkele minuten.'),
        code('''RERUN_TRAINING = False
if RERUN_TRAINING:
    import joblib
    with joblib.parallel_config(backend='threading'):
        experiment.run()
else:
    print('Bestaande nieuwe training gecontroleerd; geen tweede training gestart.')
''')])
    save('03_vergelijking.ipynb', [md('# Mushroom — vergelijking en kritische conclusie\n\n' + intro),
        md('## Buitenfoldmetrics, geen testselectie\nElke variant krijgt precies één voorspelling voor ieder van de 4.100 rijen wanneer die rij buiten de training ligt. Bij default gebruiken we de echte sklearn-labels; bij de recallpolicy geldt score >= de binnen training gekozen drempel.'),
        code(BOOT + '''summary = pd.read_csv(HERE/'results/summary.csv')
fold_metrics = pd.read_csv(HERE/'results/outer_fold_metrics.csv')
pred = pd.read_csv(HERE/'results/outer_oof_predictions.csv', float_precision='round_trip')
assert len(pred) == 4*4100
for name, group in pred.groupby('model'):
    assert group.source_row.is_unique and set(group.source_row) == set(ids)
    for fold, part in group.groupby('fold'):
        calculated = scores(part.true_class, part.probability_p, prediction=part.prediction)
        stored = fold_metrics[(fold_metrics.model == name) & (fold_metrics.fold == fold)].iloc[0]
        for metric, value in calculated.items():
            assert abs(value - stored[metric]) < 1e-12, (name, fold, metric)
print(summary[['model','macro_f1_mean','macro_f1_sd','recall_p_mean','precision_p_mean','specificity_e_mean','fn_total','fp_total']].round(4).to_string(index=False))
'''),
        md('## De afweging zichtbaar\nDe grafiek gebruikt dezelfde buitenfolds. Meer recall kan samengaan met meer valse alarmen en lagere macro F1. SD over folds is geen betrouwbaarheidsinterval. De 90%-binnenregel garandeert geen 90% buitenrecall.'),
        code('''import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
labels = ['RF300\\nstandaard','RF300\\nrecallregel','RF getuned\\nstandaard','RF getuned\\nrecallregel']
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
pos = np.arange(len(summary))
axes[0].bar(pos-.18, summary.macro_f1_mean, width=.36, label='Macro F1', color='#24688e')
axes[0].bar(pos+.18, summary.recall_p_mean, width=.36, label='Recall giftig', color='#d47b20')
axes[0].set(xticks=pos, xticklabels=labels, ylim=(0,1.05), title='Gemiddelde over vijf buitenfolds', ylabel='Score')
axes[0].legend()
axes[1].bar(pos-.18, summary.fn_total, width=.36, label='Giftig gemist (FN)', color='#b74740')
axes[1].bar(pos+.18, summary.fp_total, width=.36, label='Vals alarm (FP)', color='#8d9198')
axes[1].set(xticks=pos, xticklabels=labels, title='Fouten op dezelfde 4.100 ontwikkelrijen', ylabel='Aantal')
axes[1].legend()
fig.savefig(HERE/'results/vergelijking.png', dpi=150)
plt.close(fig)
print('Grafiek opgeslagen: results/vergelijking.png')
'''),
        md('![Buitenfoldvergelijking](results/vergelijking.png)'),
        md('## Betekenis van de uitkomst\nLees [RESULTATEN.md](RESULTATEN.md) voor het gemeten oordeel. Een kleine afwijking op deze reeds gebruikte ontwikkelingsdata bewijst geen algemene superioriteit. De sklearn 1.6.1-referentie en kandidaat zijn wel eerlijk binnen één omgeving vergeleken. Vergelijk onze getallen daarom niet alsof het een nieuwe score op de 900 ontwikkeltestrecords is. Het experiment maakt geen website-artifact en verandert geen hosting.'),
        code('''print('Onveranderde bronhash:', json.loads((HERE/'results/protocol.json').read_text())['source_sha256_lf'])
print('Nieuwe testrecords gescoord:', json.loads((HERE/'results/run.json').read_text())['test_rows_scored'])
print('Nieuwe deployment gemaakt:', json.loads((HERE/'results/run.json').read_text())['new_deployment_artifact_created'])
''')])


if __name__ == '__main__':
    main()
