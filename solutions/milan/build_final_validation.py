"""Execute the actual full-check notebook in a real isolated Jupyter kernel."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from final_validation import configure,EVIDENCE,KERNEL,WORK


def save_html(path, html):
    # DataFrame.info() pads its text output; trim only HTML line endings.
    path.write_text('\n'.join(line.rstrip() for line in html.splitlines())+'\n',encoding='utf-8')


def make_notebook():
    md=nbf.v4.new_markdown_cell; code=nbf.v4.new_code_cell
    cells=[md('''# Final — opnieuw uitgevoerde eindcontrole

Dit notebook voert de controles echt uit en bewaart de resultaten. De bestaande
Final-notebooks worden in **aparte Jupyter-kernels met Python 3.11** herhaald.
De API-tests gebruiken een schone Python 3.13-omgeving met de exacte
`backend/requirements.txt`-versies. Actieve modellen worden niet vervangen.
Uitgevoerd door Codex op verzoek van de gebruiker; geen menselijke eindgoedkeuring.

De zeven oorspronkelijke AWS-/Citi Bike-bronnotebooks worden op schema en opgeslagen
fouten gecontroleerd. Nieuwe AWS-training, volledige PyCaret-zoektocht en de
oorspronkelijke grote Citi Bike-training worden hier niet opnieuw uitgevoerd.
De twaalf notebooks uit de bestaande Final-uitvoeringslijst worden wel echt herhaald.'''),
        code('''from pathlib import Path
import sys,json
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'Final/tools/verify_final.py').exists())
sys.path.insert(0,str(ROOT/'solutions/milan'))
import pandas as pd
from IPython.display import display
import final_validation as validation
checks=[]
def run_check(name, action):
    print(name)
    try:
        result=action()
        checks.append({'controle':name,'geslaagd':True})
        return result
    except Exception as error:
        checks.append({'controle':name,'geslaagd':False,'fout':str(error)})
        print('MISLUKT:',str(error))
        return None
environment=run_check('Exacte trainingsomgeving',validation.environment)
display(environment)
'''),
        md('## 1. Final: bronnen, notebookstructuur en links'),
        code("structure=run_check('Final-structuur en alle lokale links',validation.final_structure)\nif structure: print(structure['output'])"),
        md('## 2. De volledige bestaande Python-testset\n\nInclusief beide echte modelbestanden, foutieve invoer, exportreferenties, modelpublicatie, CORS en Citi Bike-historie/zomertijd.'),
        code("python_tests=run_check('Volledige Python API-/modeltests',validation.python_suite)\nif python_tests: print(python_tests['output'])"),
        md('## 3. JavaScript: syntaxis en bestaande frontendtests'),
        code("frontend=run_check('Frontendtests en syntaxis',validation.frontend_suite)\nif frontend: print(frontend['output'])"),
        md('## 4. Twaalf Final-notebooks opnieuw van boven naar beneden\n\nIedere notebook krijgt een eigen kernel. De vier modelreproducties fitten hun vaste configuraties opnieuw en vergelijken met bestaande referenties. De oorspronkelijke Final-uitvoer blijft behouden; verse kopieën worden afzonderlijk opgeslagen.'),
        code("notebook_runs=run_check('Twaalf verse notebookuitvoeringen',validation.notebooks)\nif notebook_runs: display(pd.DataFrame(notebook_runs)[['notebook','code_cells','executed','errors','embedded_plots','seconds','passed']])"),
        md('## 5. Het actieve Mushroom-model vergelijken met zijn export\n\nAlle 5.000 invoerrijen worden op uitvoergelijkheid gecontroleerd. Dit is een integriteitscontrole, geen nieuwe onafhankelijke prestatietest.'),
        code("export=run_check('Pipeline en serving-export identiek',validation.export_equivalence)\ndisplay(export)"),
        md('## 6. De publieke website en API op dit moment\n\nControle van modelrevisie, echte Mushroom- en Citi Bike-voorspellingen en CORS. Dit controleert de gepubliceerde hoofdversie. De conceptbranch bevat rapportverbeteringen en is geen nieuwe publieke deployment.'),
        code("live=run_check('Live website en beide API-modellen',validation.live_checks)\nif live:\n    display(live['health'])\n    display(live['mushroom_prediction'])\n    display(pd.DataFrame(live['city_prediction']['daily_predictions']))\n    print('CORS:',live['cors_status'],'; modelrevisie lokaal en live gelijk:',live['same_local_and_live_revision'])"),
        md('## 7. Bewaken dat de controle niets aan de werkende keten verandert'),
        code("preserved=run_check('Final en actieve modelketen behouden',validation.unchanged)\ndisplay(preserved)"),
        md('## Resultaat en grenzen\n\nEen geslaagde technische controle betekent dat de geteste keten werkt; het bewijst geen foutloze voorspellingen op nieuwe gevallen. Zie de afzonderlijke uitvoer en receipts voor scope en tijdstip.'),
        code("display(pd.DataFrame(checks))\nsummary={'finished_at_utc':validation.datetime.now(validation.timezone.utc).isoformat(),'all_passed':len(checks)==8 and all(r['geslaagd'] for r in checks),'checks':checks,'training_repeated':'Four fixed Mushroom model reproductions through the twelve Final notebooks; no automatic publication.'}\nvalidation.write('summary.json',summary)\nassert summary['all_passed'], 'Er zijn mislukte controles; zie de uitvoer hierboven.'\nprint('ALLE ACHT CONTROLEGROEPEN GESLAAGD.')")]
    nb=nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec={'name':KERNEL,'display_name':'Final-validatie Python 3.11','language':'python'}
    return nb


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-only',action='store_true')
    args=parser.parse_args()
    configure(); EVIDENCE.mkdir(exist_ok=True)
    path=HERE/'11_Final_eindcontrole.ipynb'
    if args.render_only:
        nb=nbf.read(path,as_version=4)
    else:
        nb=make_notebook()
        def start(cell,cell_index,**kwargs):
            if cell.cell_type=='code': print('Execute proof cell',cell_index,flush=True)
        try:
            NotebookClient(nb,timeout=2400,kernel_name=KERNEL,allow_errors=True,
                on_cell_start=start,resources={'metadata':{'path':str(ROOT)}}).execute()
        finally:
            nbf.validate(nb); nbf.write(nb,path)
    exporter=HTMLExporter(template_name='lab',exclude_input=True)
    html,_=exporter.from_notebook_node(nb)
    save_html(EVIDENCE/'Final_eindcontrole.html',html)
    for stem in ['01_EDA_heruitgevoerd','09_vergelijking_heruitgevoerd']:
        if (EVIDENCE/(stem+'.ipynb')).exists():
            fresh=nbf.read(EVIDENCE/(stem+'.ipynb'),as_version=4)
            html,_=exporter.from_notebook_node(fresh)
            save_html(EVIDENCE/(stem+'.html'),html)
    errors=[o for c in nb.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error']
    summary=json.loads((EVIDENCE/'summary.json').read_text()) if (EVIDENCE/'summary.json').exists() else {}
    print('Proof saved:',path,'; errors:',len(errors),'; all passed:',summary.get('all_passed'),flush=True)
    return 0 if not errors and summary.get('all_passed') else 1


if __name__=='__main__':
    raise SystemExit(main())
