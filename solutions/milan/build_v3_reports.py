"""Readable course reports from the complete third study, never partial folds."""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
import build_v2_reports as reports
import pandas as pd
from mushroom_alternative import write_json

LABELS = {'rf300_reference_default': 'Oude RF300',
    'rf_compatible_f1_threshold': 'Kandidaat v2',
    'rf_selection': 'RF + featureselectie', 'lightgbm': 'LightGBM',
    'catboost': 'CatBoost', 'voting': 'Gewogen voting'}


def label(name):
    if name in LABELS:
        return LABELS[name]
    for suffix, text in [('_fpr_budget', ' + FPR-budget'), ('_default', ' + standaardgrens'),
                         ('_f1', ' + F1-drempel')]:
        if name.endswith(suffix):
            return LABELS[name[:-len(suffix)]] + text
    raise ValueError(name)


BOOT = '''from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
HERE = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'results_v3/run.json').exists())
sys.path.insert(0, str(HERE))
from mushroom_alternative import development_data, scores
from improve_v3 import experiments, InputFeatures, WEIGHTS
from build_v3_reports import label
X, y, ids, folds = development_data()
summary = pd.read_csv(HERE/'results_v3/summary.csv').set_index('model')
details = pd.read_csv(HERE/'results_v3/outer_fold_metrics.csv')
predictions = pd.read_csv(HERE/'results_v3/outer_oof_predictions.csv', float_precision='round_trip')
selections = json.loads((HERE/'results_v3/selections.json').read_text())
assert len(summary) == 14 and summary.folds.eq(5).all()
'''


def run(inline=False):
    complete = json.loads((HERE/'results_v3/run.json').read_text())
    assert complete['complete']
    table = pd.read_csv(HERE/'results_v3/summary.csv').set_index('model')
    assert len(table) == 14 and table.folds.eq(5).all()
    reference = table.loc['rf_compatible_f1_threshold']
    candidates = table.drop(index=['rf300_reference_default','rf_compatible_f1_threshold'])
    eligible = candidates[(candidates.macro_f1_mean > reference.macro_f1_mean)
        & (candidates.fn_total < reference.fn_total) & (candidates.fp_total <= reference.fp_total)]
    chosen = eligible.macro_f1_mean.idxmax() if len(eligible) else None
    best = candidates.macro_f1_mean.idxmax()
    write_json(HERE/'results_v3/decision.json', {
        'performance_goal_met': chosen is not None, 'goal_qualifying_procedures': eligible.index.tolist(),
        'highest_macro_f1': best, 'best_goal_qualifying_procedure': chosen,
        'selection_scope': 'Exploratory comparison after completing all prespecified folds; no independent test.',
        'production_changed': False, 'test_rows_scored': 0})
    lines = ['# Derde studie: wat verbetert echt?', '',
        'Vijf gelijke buitenfolds op de oorspronkelijke 4.100 trainingsrijen; alle',
        'voorbereiding, configuraties, drempels en votinggewichten worden binnen de',
        'training gekozen. De bekende 900 ontwikkeltestrijen zijn niet gescoord.', '',
        'De lessen over featureselectie, boosting en ensembles vormen de basis.',
        'Deze herhaalde ontwikkelstudie is geen nieuwe onafhankelijke eindtest.', '',
        '| Procedure | Macro F1 | Recall giftig | FN | FP | Accuracy |',
        '|---|---:|---:|---:|---:|---:|']
    for name,row in table.iterrows():
        lines.append(f'| {label(name)} | {row.macro_f1_mean:.4f} | {row.recall_p_mean:.2%} | {int(row.fn_total)} | {int(row.fp_total)} | {row.accuracy_mean:.2%} |')
    lines += ['', 'Macro F1, recall en accuracy zijn gemiddelden over vijf folds.',
        'FN en FP zijn de opgetelde aantallen over de 4.100 buitenfoldvoorspellingen.', '',
        f'De hoogste macro F1 komt van **{label(best)}**: **{candidates.loc[best].macro_f1_mean:.4f}**.', '']
    if chosen:
        row = candidates.loc[chosen]
        lines += [f'Het vooraf beschreven doel wordt gehaald door **{label(chosen)}**:',
            f'macro F1 stijgt tegenover v2 met **{row.macro_f1_mean-reference.macro_f1_mean:.4f}**,',
            f'er zijn **{int(reference.fn_total-row.fn_total)} minder gemiste giftige gevallen**',
            f'en **{int(row.fp_total)} valse alarmen** tegenover {int(reference.fp_total)} bij v2.',
            'De omvang en consistentie staan in het vergelijkingsnotebook; een kleine winst blijft klein.']
    else:
        lines += ['**Geen nieuwe procedure haalt het volledige verbeterdoel tegenover v2.**',
            'Een hogere score alleen rechtvaardigt dus geen overstap. Een booster of',
            'ensemble toevoegen is geen verbetering op zichzelf. De bestaande v2-kandidaat',
            'blijft beschikbaar met zijn gecontroleerde invoer, export en preview.']
    details=pd.read_csv(HERE/'results_v3/outer_fold_metrics.csv')
    paired=details.pivot(index='fold',columns='model',values='macro_f1')
    if chosen:
        difference=paired[chosen]-paired['rf_compatible_f1_threshold']
        lines += ['', f'De gekozen procedure verbetert macro F1 in **{int((difference>0).sum())} van 5 folds**.',
            'De vijf verschillen staan in het notebook; dit wordt niet als een',
            'significantietoets of veiligheidsbewijs voorgesteld.']
    rf_receipt=HERE/'results_v3/candidate_package.json'
    if rf_receipt.exists() and json.loads(rf_receipt.read_text())['packaged']:
        rf=json.loads(rf_receipt.read_text()); row=table.loc['rf_selection_'+rf['policy']]
        lines += ['', '## Een eenvoudiger alternatief voor aansluiting', '',
            f'RF met featureselectie en standaardgrens haalt **{row.macro_f1_mean:.4f} macro F1**',
            f'en **{row.accuracy_mean:.2%} accuracy**. Het pakket houdt de bestaande',
            'modellezer en elf invoervelden. De fulltrain-selectie negeert intern',
            f"`{', '.join(rf['ignored_features'])}`. De folds kiezen dit niet allemaal hetzelfde.",
            f'Tegenover v2 mist deze RF **{rf["extra_fn_against_v2"]} meer giftige gevallen**,',
            f'met **{-rf["fp_change_against_v2"]} minder valse alarmen**. Hij haalt het volledige v2-doel dus niet.',
            'Tegenover de oorspronkelijke RF300 verlaagt hij wel beide foutaantallen.',
            'Kies hem voor de eenvoudiger aansluiting en deze specifieke foutafweging;',
            'noem hem niet beter dan voting in giftige recall. Zie [integratie v3](INTEGRATIE_V3.md).']
    lines += ['', '## Wat is geleerd?', '',
        '- Featureselectie is onderdeel van de pipeline en wordt via de binnenfolds gekozen.',
        '- LightGBM krijgt expliciete categorieën; onbekende categorieën worden ontbrekend.',
        '- CatBoost krijgt categorische ontbrekende waarden als aparte tekstcategorie.',
        '- De regularisatie en complexiteit van beide boosters worden vergeleken.',
        '- Gewogen soft voting combineert kansen; binnen-OOF kiest uit zeven vaste gewichten.',
        '- Het FPR-budget van 14% geldt voor binnen-OOF. De buitenfold toetst of dit overgaat.',
        '- Alle twaalf nieuwe procedures worden gerapporteerd, inclusief tegenvallende uitkomsten.', '',
        'De 360 gridfits betreffen 24 configuraties × 3 binnenfolds × 5 buitenfolds,',
        'naast de benodigde refits en binnen-OOF-berekeningen. Meer onderzoek maakt',
        'eerder bekeken data niet weer onaangeraakt. Kies de laatste kandidaat op',
        'ontwikkelbewijs en gebruik later echt nieuwe gelabelde data voor bevestiging.', '',
        'Zie [onderzoeksafspraken](STUDIE_V3.md), [protocol](results_v3/protocol.json)',
        'en de uitgevoerde notebooks 08–10 voor code, controleerbare uitvoer en foutanalyse.']
    (HERE/'RESULTATEN_V3.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    reports.INLINE = inline
    reports.configured_kernel()
    md = reports.nbf.v4.new_markdown_cell
    code = reports.nbf.v4.new_code_cell
    receipts = []
    receipts.append(reports.notebook('08_featureselectie_en_boosting.ipynb', [
        md('# Featureselectie en boosting\n\nLesbasis: **4 — Models**, dia 42 en 49 voor geoptimaliseerde boosting en regularisatie. Alleen 4.100 trainingsrijen; dit is verdere ontwikkeling op eerder gebruikte data.'),
        code(BOOT),
        md('## Eerst de datavoorbereiding\n\nEen veld intern weglaten verandert het publieke formulier met elf velden niet. De keuze gebeurt binnen training. Het label komt nooit in de invoer terecht. Voor LightGBM leren we de categorievocabulaire uitsluitend van training; CatBoost verwerkt categorische tekstwaarden.'),
        code("display(X.isna().mean().sort_values(ascending=False).rename('fractie_ontbrekend').to_frame())\nfor family,(pipeline,grid) in experiments().items():\n    print(family, pipeline)\n    print('zoekruimte:', grid)"),
        md('## De trainingsstappen zoals in de oefeningen\n\nOnderstaande optionele cel herhaalt één RF-buitenfold en schrijft geen bestaand resultaat over. De volledige echte run staat in `improve_v3.py`; alle vijf folds en alle families zijn al uitgevoerd voor de tabellen hier. Zet de schakelaar alleen aan als je deze ene fit opnieuw wilt doen.'),
        code("from sklearn.base import clone\nfrom sklearn.model_selection import GridSearchCV, StratifiedKFold\n\nRERUN_ONE_FOLD = False\nif RERUN_ONE_FOLD:\n    train, validation = folds[0]\n    pipeline, grid = experiments()['rf_selection']\n    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=9411)\n    search = GridSearchCV(clone(pipeline), grid, scoring='f1_macro', cv=inner_cv, n_jobs=1, error_score='raise')\n    search.fit(X.iloc[train], y.iloc[train])\n    probability = search.best_estimator_.predict_proba(X.iloc[validation])[:,1]\n    raw_label = search.best_estimator_.predict(X.iloc[validation])\n    display(search.best_params_)\n    display(scores(y.iloc[validation], probability, prediction=raw_label))\nelse:\n    print('Rapportmodus: hieronder staan de bewaarde resultaten van de complete training.')"),
        md('## Bekijk de werkelijke selectie per buitenfold\n\nDeze tabel toont gekozen hyperparameters en binnen-CV-scores, geen onafhankelijke eindprestaties. De buitenfold wordt pas na de selectie voorspeld.'),
        code("display(pd.DataFrame([s for s in selections if s['family']!='voting'])[['family','fold','parameters','inner_cv_macro_f1','f1_threshold','fpr_threshold']])"),
        md('## Vergelijk de uiteindelijke procedures\n\nDe standaardgrens 0,5 en de binnen training afgestemde drempels staan afzonderlijk. Gebruik de cijfers van de procedure die je werkelijk toepast.'),
        code("view = summary[['macro_f1_mean','recall_p_mean','fn_total','fp_total','accuracy_mean']].copy()\nview.index = [label(n) for n in view.index]\ndisplay(view.round(4))"),
        md('Voor opnieuw trainen: voer `reproduce_v3.py` uit met de vastgelegde dependencies. De rapportcellen lezen de complete echte zoekuitvoer; ze herhalen niet stilzwijgend honderden fits.')]))
    receipts.append(reports.notebook('09_voting_en_vergelijking.ipynb', [
        md('# Gewogen voting en eerlijke vergelijking\n\nLesbasis: **4 — Models**, dia 36 en 38 over ensembles en bagging. Soft voting middelt kansen; de vaste kandidaatgewichten worden op binnen-OOF gekozen.'),
        code(BOOT),
        code("print('Vaste gewichten RF / LightGBM / CatBoost:', WEIGHTS)\ndisplay(pd.DataFrame([s for s in selections if s['family']=='voting']))"),
        md('Een gewicht van één voor RF betekent dat de andere modellen in die fold niet helpen volgens de trainingsselectie. Een ingewikkelder ensemble verdient geen voorkeur zonder voldoende winst.'),
        code("paired = details.pivot(index='fold', columns='model', values='macro_f1')\ndelta = paired.subtract(paired['rf_compatible_f1_threshold'], axis=0)\ndisplay(delta.round(4))\ndisplay(pd.DataFrame({'gemiddeld_verschil': delta.mean(), 'folds_beter_dan_v2': (delta>0).sum()}))"),
        md('## Fouten vergelijken\n\nIedere punt gebruikt dezelfde 4.100 buitenfoldrijen. Lager en meer naar links betekent minder fouten van beide typen. Dit is een beschrijvende vergelijking; geen gecorrigeerde significantietoets na meerdere studies.'),
        code("import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nfrom IPython.display import Image, display\nfig, ax = plt.subplots(figsize=(12,7))\nfor name, row in summary.iterrows():\n    ax.scatter(row.fp_total, row.fn_total)\n    right = row.fp_total > 390\n    ax.annotate(label(name), (row.fp_total, row.fn_total), xytext=(-5 if right else 5,4), ha='right' if right else 'left', textcoords='offset points', fontsize=8)\nax.set(xlabel='Vals alarm: eetbaar als giftig', ylabel='Giftig gemist', title='Alle vijf buitenfolds samen — ontwikkelonderzoek')\nax.grid(alpha=.25)\nfig.tight_layout()\nfig.savefig(HERE/'results_v3/foutenvergelijking.png', dpi=150)\nplt.close(fig)\ndisplay(Image(filename=str(HERE/'results_v3/foutenvergelijking.png')))"),
        code("display(json.loads((HERE/'results_v3/decision.json').read_text()))"),
        md('De uiteindelijke kandidaat wordt op ontwikkelbewijs gekozen. Een betrouwbaar eindcijfer vereist later nieuwe data; de al bekeken 900 records worden daarvoor niet hergebruikt.')]))
    receipts.append(reports.notebook('10_foutanalyse_v3.ipynb', [
        md('# Foutanalyse en toepassingsgrenzen\n\nGebruik buitenfoldvoorspellingen voor de foutanalyse. Subgroepen beschrijven samenhang; ze bewijzen geen oorzaak en hebben soms weinig records.'),
        code(BOOT),
        code("decision = json.loads((HERE/'results_v3/decision.json').read_text())\nname = decision['best_goal_qualifying_procedure'] or 'rf_compatible_f1_threshold'\np = predictions[predictions.model==name].set_index('source_row').loc[ids].copy()\np['missing_count'] = X.isna().sum(axis=1).to_numpy()\np['missing_group'] = pd.cut(p.missing_count, [-1,2,4,11], labels=['0–2','3–4','5–11'])\nrows = []\nfor group, part in p.groupby('missing_group', observed=True):\n    row = {'groep': str(group), 'rijen':len(part), 'giftig':int(part.true_class.sum())}\n    if part.true_class.nunique()==2:\n        row.update(scores(part.true_class,part.probability_p,prediction=part.prediction))\n    rows.append(row)\nprint('Procedure:', label(name))\ndisplay(pd.DataFrame(rows))"),
        md('## Hoe verschillend zijn de fouten?\n\nDe kruistabel laat zien welke fouten een nieuwe procedure herstelt en welke ze toevoegt. Betere gemiddelde prestaties maken een individueel geval niet betrouwbaar.'),
        code("baseline=predictions[predictions.model=='rf_compatible_f1_threshold'].set_index('source_row').loc[ids]\ndisplay(pd.crosstab(baseline.prediction.ne(baseline.true_class),p.prediction.ne(p.true_class),rownames=['v2 fout'],colnames=['gekozen procedure fout']))\nprint('Nog gemiste giftige gevallen:', int(((p.true_class==1)&(p.prediction==0)).sum()))"),
        md('## Controle van de onderzoeksketen\n\nDe tests reconstrueren metrics vanuit opgeslagen individuele voorspellingen, controleren originele rijlidmaatschappen en reconstrueren votingkansen uit de gekozen gewichten. Export- en API-controles staan afzonderlijk bij de kandidaat.'),
        code("for name in ['test_results.json','package_test_results.json','frontend_test_results.json']:\n    receipt=json.loads((HERE/'results_v3'/name).read_text())\n    assert receipt['success']\n    print(name)\n    display({k:v for k,v in receipt.items() if k not in ['output','hashes']})"),
        md('Geen eindtest of eetbaarheidsbevestiging. Controleer bij latere integratie artifact, voorbereiding, drempel, metrics, versiecontract en automatische training samen. De gezamenlijke applicatie is hier niet aangepast.')]))
    write_json(HERE/'results_v3/notebook_execution.json', {'notebooks':receipts,
        'execution_engine':'in-process IPython, fresh namespaces' if inline else 'Jupyter kernel',
        'all_executed':True, 'validated_with':'nbformat.validate'})
    print('Three complete study reports executed and validated.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inline',action='store_true')
    run(parser.parse_args().inline)
