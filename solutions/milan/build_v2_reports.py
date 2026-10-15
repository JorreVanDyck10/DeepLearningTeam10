"""Generate reports; use a Jupyter kernel or an explicit in-process IPython run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PACKAGES = HERE / '.generated/preview_packages'
if PACKAGES.exists():
    sys.path.insert(0, str(PACKAGES))
import nbformat as nbf
from nbclient import NotebookClient
import numpy as np
import pandas as pd

from mushroom_alternative import ROOT, development_data, scores, write_json
from package_candidate import acceptance
from improve import OUT

INLINE = False
LABELS = {'rf300_reference_default': 'RF300 huidig',
    'rf_compatible_default': 'RF300 nieuwe voorbereiding',
    'rf_compatible_f1_threshold': 'RF300 + beslisregel',
    'gradient_boosting_default': 'Gradient Boosting',
    'gradient_boosting_f1_threshold': 'GB + beslisregel',
    'rf_features_default': 'RF + extra kenmerken',
    'rf_features_f1_threshold': 'RF extra + beslisregel',
    'svm_default': 'SVM', 'svm_f1_threshold': 'SVM + beslisregel'}

BOOT = '''from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
HERE = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'results_v2/summary.csv').exists())
sys.path.insert(0, str(HERE))
from mushroom_alternative import development_data, scores
from improve import experiments, make_pipeline, Measurements
X, y, ids, folds = development_data()
summary = pd.read_csv(HERE/'results_v2/summary.csv')
details = pd.read_csv(HERE/'results_v2/outer_fold_metrics.csv')
selections = json.loads((HERE/'results_v2/selections.json').read_text())
'''


def configured_kernel():
    jupyter = HERE / '.generated/jupyter'
    kernel = jupyter / 'kernels/milan'
    kernel.mkdir(parents=True, exist_ok=True)
    write_json(kernel / 'kernel.json', {'argv': [sys.executable, '-B', '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Milan concept — Python 3.13', 'language': 'python'})
    os.environ['JUPYTER_PATH'] = str(jupyter)
    os.environ['JUPYTER_RUNTIME_DIR'] = str(HERE / '.generated/runtime')
    os.environ['IPYTHONDIR'] = str(HERE / '.generated/ipython')
    os.environ['JUPYTER_CONFIG_DIR'] = str(HERE / '.generated/jupyter_config')
    os.environ['PYTHONPATH'] = str(PACKAGES) + os.pathsep + str(HERE)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def notebook(name, cells):
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {'name': 'milan', 'display_name': 'Milan concept — Python 3.13', 'language': 'python'}
    if INLINE:
        from IPython.core.interactiveshell import InteractiveShell
        from IPython.utils.capture import capture_output
        shell = InteractiveShell.instance()
        shell.reset(new_session=True)
        for count, cell in enumerate([c for c in nb.cells if c.cell_type == 'code'], 1):
            with capture_output() as captured:
                result = shell.run_cell(cell.source, store_history=False)
            if result.error_before_exec or result.error_in_exec:
                raise RuntimeError(str(result.error_before_exec or result.error_in_exec) + '\n' + captured.stdout)
            cell.execution_count = count
            cell.outputs = []
            for channel in ['stdout', 'stderr']:
                if getattr(captured, channel):
                    cell.outputs.append(nbf.v4.new_output('stream', name=channel, text=getattr(captured, channel)))
            for item in captured.outputs:
                cell.outputs.append(nbf.v4.new_output('display_data', data=item.data, metadata=item.metadata))
        nb.metadata.execution_note = 'Executed with in-process IPython, fresh notebook namespace; not a separate Jupyter-kernel run.'
    else:
        NotebookClient(nb, kernel_name='milan', timeout=600,
            resources={'metadata': {'path': str(HERE)}}).execute()
    nbf.validate(nb)
    path = HERE / name
    nbf.write(nb, path)
    codes = [c for c in nb.cells if c.cell_type == 'code']
    assert all(c.execution_count is not None for c in codes)
    assert not any(o.output_type == 'error' for c in codes for o in c.outputs)
    return {'file': name, 'code_cells': len(codes), 'executed_in_kernel': not INLINE,
        'execution_engine': 'in-process IPython, fresh namespace' if INLINE else 'Jupyter kernel',
        'sha256_lf': hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()}


def results_text():
    table = pd.read_csv(OUT / 'summary.csv').set_index('model')
    assert table['folds'].eq(5).all() and len(table) == 9
    reference = table.loc['rf300_reference_default']
    lines = ['# Verdere verbetering: de gemeten uitkomst', '',
        'Alle cijfers hieronder komen uit vijf buitenfolds op de 4.100 historische',
        'trainingsrijen. De 900 bekende ontwikkeltestrecords zijn niet gescoord.',
        'Dit blijft ontwikkelonderzoek op eerder gebruikte projectdata.', '',
        '| Procedure | Macro F1 | Recall giftig | Precision giftig | FN | FP |',
        '|---|---:|---:|---:|---:|---:|']
    for name, row in table.iterrows():
        lines.append(f'| {LABELS[name]} | {row.macro_f1_mean:.4f} | {100*row.recall_p_mean:.2f}% | {100*row.precision_p_mean:.2f}% | {int(row.fn_total)} | {int(row.fp_total)} |')
    ranked = table.drop(index='rf300_reference_default').sort_values('macro_f1_mean', ascending=False)
    best_name = str(ranked.index[0]); best = ranked.iloc[0]
    compatible = table.loc['rf_compatible_f1_threshold']
    gates = acceptance(policy='f1_threshold')
    write_json(OUT / 'development_decision.json', {'best_exploratory_procedure': best_name,
        'best_macro_f1': float(best.macro_f1_mean), 'best_macro_f1_gain': float(best.macro_f1_mean-reference.macro_f1_mean),
        'compatible_procedure': gates, 'prospective_test': False, 'automatic_promotion': False})
    lines += ['', '## Wat verbetert daadwerkelijk?', '',
        f'De hoogste ontwikkelscore komt van **{LABELS[best_name]}**: macro F1',
        f'**{best.macro_f1_mean:.4f}**, tegenover **{reference.macro_f1_mean:.4f}** voor RF300.',
        f'De beperkte-integratievariant RF300 met de binnen training gekozen drempel haalt',
        f'**{compatible.macro_f1_mean:.4f}** en **{100*compatible.recall_p_mean:.2f}% giftige recall**.',
        f'Dat zijn **{int(reference.fn_total-compatible.fn_total)} minder FN** en',
        f'**{int(compatible.fp_total-reference.fp_total)} extra FP** op dezelfde 4.100 rijen.', '',
        'De drempel hoort bij de **beslispolicy**. De oorspronkelijke ruwe classifierlabels',
        'en modelscores blijven in de preview beschikbaar. Daarom worden raw-modelmetrics',
        'en policy-metrics in het kandidaatmanifest strikt apart opgeslagen.', '',
        f'De beperkte-integratievariant haalt de vooraf beschreven ontwikkelcriteria: **{"ja" if gates["passes_performance_gates"] else "nee"}**.',
        f'De vijf gepaarde macro-F1-verschillen zijn `{[round(v, 6) for v in gates["paired_fold_macro_f1_differences"]]}`.',
        'Deze criteria zijn geen significantietoets of onafhankelijke eindvalidatie.', '',
        'Meer modelcomplexiteit wordt alleen aanbevolen als de daadwerkelijke winst de',
        'extra voorbereiding en aangepaste inference rechtvaardigt. De totale accuracy',
        'blijft hier ongeveer 78,8%; de winst zit in de andere afweging van fouttypen.',
        'De uitgebreidere RF-voorbereiding en SVM leveren in deze studie geen reden',
        'om hun extra complexiteit over te nemen. Een hoger maximum',
        'na meer zoeken is op zichzelf onvoldoende bewijs.', '',
        '## Wat staat klaar voor later?', '',
        '- Een gewone RF300-pipeline zonder eigen serialisatieklassen en een verliesvrije',
        '  array-export, gecontroleerd met de bestaande `PortableMushroom`.',
        '- Een lokale preview die het huidige invoerschema en de echte API-controller',
        '  hergebruikt. Alleen de voorgestelde beslisregel en bijpassende metadata worden',
        '  in de conceptmap toegepast. De website bevestigt nog steeds geen eetbaarheid.',
        '- [Integratie-instructies](INTEGRATIE.md) met de kleine wijzigingen die later',
        '  samen moeten worden overgenomen, plus concrete uitvoeringscontroles.', '',
        '## Grenzen', '',
        'Er is geen algemene veiligheid of onaangeraakte eindtest aangetoond. De gridkeuzes',
        'zijn binnen iedere training gemaakt; de totale tweede studie is wel geïnformeerd',
        'door eerder projectonderzoek. Dat onderscheid blijft expliciet.', '',
        'Verschillen tussen de families zijn verschillen tussen complete procedures.',
        'Daarom schrijven we een verschil niet zonder aparte ablatietest volledig toe aan',
        'één ratio, één imputatiemethode of één classifierinstelling.', '',
        '[Protocol](results_v2/protocol.json), [foldmetrics](results_v2/outer_fold_metrics.csv),',
        '[samenvatting](results_v2/summary.csv) en [besluit](results_v2/development_decision.json).', '',
        '![Vergelijking van de tweede studie](results_v2/vergelijking.png)', '']
    (HERE / 'RESULTATEN_V2.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    receipt = json.loads((OUT / 'run.json').read_text())
    assert receipt['complete']
    configured_kernel()
    results_text()
    md = nbf.v4.new_markdown_cell; code = nbf.v4.new_code_cell
    intro = '''**Concept op verzoek van de gebruiker, uitgewerkt met Codex.** Milan moet
de keuzes en eigen bijdrage zelf beoordelen. Alle nieuwe bestanden blijven onder
`solutions/milan`. We gebruiken de cursusmethoden en de oorspronkelijke elf
invoervelden. Geen nieuwe onafhankelijke eindtest of menselijke review geclaimd.
De dure training is afzonderlijk werkelijk uitgevoerd; onderstaande rapportcellen
    worden van boven naar beneden uitgevoerd; de precieze uitvoeringsmethode staat
in de notebookmetadata en uitvoeringsreceipt.'''
    receipts = []
    for filename, title, names in [
        ('04_RF_verbeterd.ipynb', 'Random Forest — voorbereiding en beslisregel', ['rf_compatible','rf_features']),
        ('05_Gradient_Boosting.ipynb', 'Gradient Boosting — eenvoudige fold-lokale voorbereiding', ['gradient_boosting']),
        ('06_SVM.ipynb', 'SVM — scaling, C en gamma', ['svm'])]:
        receipts.append(notebook(filename, [md('# ' + title + '\n\n' + intro),
            md('## Model en zoekruimte\nWe gebruiken exact dezelfde buitenfolds. Per familie selecteren drie binnenfolds de instellingen. Scaling wordt bij SVM binnen de pipeline geleerd; boommodellen krijgen geen onnodige standaardisatie.'),
            code(BOOT + f'''families = {names!r}
for family in families:
    estimator, grid = experiments()[family]
    display(estimator)
    print('Zoekruimte:', grid)
'''),
            md('## De werkelijk uitgevoerde selectie\nDeze tabel geeft per buitenfold de binnen training gekozen instellingen en drempel. Een binnenscore is een selectiescore, niet de buitenmeting.'),
            code('''display(pd.DataFrame([s for s in selections if s['family'] in families]))
for family in families:
    display(details[details.model.str.startswith(family+'_')])
'''),
            md('## Resultaten tegenover RF300\nStandaardlabels en een afgestemde macro-F1-beslisregel staan apart. Die drempel is niet op buitenvalidatie gekozen. Grotere recall heeft alleen betekenis samen met precision, FN en FP.'),
            code('''display(summary[summary.model.eq('rf300_reference_default') | summary.model.str.startswith(tuple(families))][['model','macro_f1_mean','macro_f1_sd','recall_p_mean','precision_p_mean','fn_total','fp_total']])
'''),
            md('## Opnieuw trainen\nDe opgeslagen training herhalen kan met onderstaande vlag. Standaard worden alleen de echte resultaten gelezen, zonder onnodig opnieuw te fitten.'),
            code('''RERUN = False
if RERUN:
    import joblib
    from improve import run
    with joblib.parallel_config(backend='threading'):
        run(families)
else:
    print('Rapportcontrole; geen nieuwe training gestart.')
''')]))
    receipts.append(notebook('07_foutanalyse_en_integratie.ipynb', [
        md('# Fouten, afweging en beperkte integratie\n\n' + intro),
        md('## Alle procedures op dezelfde buitenfolds\nGeen testset is opnieuw gebruikt om een winnaar te kiezen. Deze grafiek toont ontwikkelbewijs, met spreiding over folds en foutaantallen.'),
        code(BOOT + '''import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import Image, display
display(summary[['model','macro_f1_mean','recall_p_mean','fn_total','fp_total']])
from build_v2_reports import LABELS
labels = summary.model.map(LABELS)
fig, axes = plt.subplots(1,2,figsize=(13,6),layout='constrained')
pos = np.arange(len(summary))
axes[0].barh(pos-.17,summary.macro_f1_mean,height=.34,label='Macro F1',color='#24688e')
axes[0].barh(pos+.17,summary.recall_p_mean,height=.34,label='Recall giftig',color='#d47b20')
axes[0].set(yticks=pos,yticklabels=labels,xlim=(0,1),title='Vijf buitenfolds: gemiddelde scores')
axes[0].legend();axes[0].invert_yaxis()
axes[1].barh(pos-.17,summary.fn_total,height=.34,label='Giftig gemist',color='#b74740')
axes[1].barh(pos+.17,summary.fp_total,height=.34,label='Vals alarm',color='#8d9198')
axes[1].set(yticks=pos,yticklabels=['']*len(pos),title='Fouten op dezelfde 4.100 rijen')
axes[1].legend();axes[1].invert_yaxis()
fig.savefig(HERE/'results_v2/vergelijking.png',dpi=150)
plt.close(fig)
display(Image(filename=str(HERE/'results_v2/vergelijking.png')))
'''),
        md('## Waar blijft de kandidaat fouten maken?\nDe rijen hieronder zijn buitenfoldvoorspellingen, niet de finale fit op zichzelf. De groepering op ontbrekende waarden is achteraf diagnostisch en bewijst geen oorzaak.'),
        code('''pred = pd.read_csv(HERE/'results_v2/outer_oof_predictions.csv',float_precision='round_trip')
chosen = pred[pred.model=='rf_compatible_f1_threshold'].copy()
assert chosen.source_row.is_unique and set(chosen.source_row)==set(ids)
missing = pd.Series(X.isna().sum(axis=1).to_numpy(),index=ids)
chosen['missing_count'] = chosen.source_row.map(missing)
chosen['missing_group'] = pd.cut(chosen.missing_count,[-1,2,4,11],labels=['0-2','3-4','5+'])
chosen['FN'] = (chosen.true_class==1)&(chosen.prediction==0)
chosen['FP'] = (chosen.true_class==0)&(chosen.prediction==1)
errors = chosen.groupby('missing_group',observed=True).agg(rows=('source_row','size'),poisonous=('true_class','sum'),FN=('FN','sum'),FP=('FP','sum'))
errors['recall_p'] = 1-errors.FN/errors.poisonous
display(errors)
errors.to_csv(HERE/'results_v2/errors_by_missing.csv')
display(chosen[chosen.FN].sort_values('probability_p').head(10))
'''),
        md('## Ruwe classifier en beslispolicy niet verwarren\nDe bestaande modellezer blijft labels bij de oorspronkelijke grens geven. De preview kiest de waarschuwing met de apart vastgelegde drempel. De twee metrieksets worden apart opgeslagen; geen oude testcijfers bij een nieuw model.'),
        code('''package = json.loads((HERE/'results_v2/candidate_package.json').read_text())
manifest = json.loads((HERE/'artifacts/mushroom_rf300.json').read_text())
display(pd.Series(package))
print('Ruwe classificatiegrens:',manifest['raw_classifier_threshold'])
print('Voorgestelde beslisgrens:',manifest['proposed_decision_threshold'])
print('Raw metric scope:',manifest['metrics']['metrics_scope'])
print('Policy metric scope: buitenfolds, afzonderlijke beslispolicy')
assert package['labels_disagree']==0 and package['development_test_rows_scored']==0
'''),
        md('Lees [RESULTATEN_V2.md](RESULTATEN_V2.md) en [INTEGRATIE.md](INTEGRATIE.md). De notebookuitvoering is controleerbaar in `results_v2/notebook_execution.json`. Een later nieuw eindonderzoek vereist nieuwe evaluatiedata; opnieuw shuffelen lost de oude testbesmetting niet op.')]))
    write_json(OUT / 'notebook_execution.json', receipts)
    print('Executed and validated', len(receipts), 'notebooks;', 'in-process IPython' if INLINE else 'Jupyter kernels')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inline', action='store_true', help='Use IPython in this process when child-kernel startup is unavailable.')
    args = parser.parse_args()
    INLINE = args.inline
    before = Path.cwd()
    try:
        if INLINE:
            os.chdir(HERE)
        main()
    finally:
        os.chdir(before)
