"""Keep confirmed Milan contributions in notebook Markdown, preserving execution."""
from copy import deepcopy
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[2]
CELL_ID = 'milan-bijdrage-20261015'
FINAL_PARTS = {
    'Mushroom/01_EDA.ipynb': (
        'Meegewerkt aan de data cleaning en EDA van de Mushroom-data. Mijn '
        'oorspronkelijke cleaning-/EDA-uitwerking staat in '
        '[SolutionMilan](../../SolutionMilan/MushroomDataset/01_clean_en_explore.ipynb). '
        'Deze Final-leesversie gebruikt de 5.000 cursusrijen en behoudt Andrews '
        'bronvermelding; het oorspronkelijke onderzoek is gezamenlijk projectwerk.'),
    'Mushroom/02_voorbereiding.ipynb': (
        'Meegewerkt aan de data cleaning en voorbereiding van de Mushroom-data. '
        'Mijn [oorspronkelijke voorbereiding](../../SolutionMilan/MushroomDataset/02_data_preparation.ipynb) '
        'hoort bij de historische databron. Deze afzonderlijke Final-versie gebruikt '
        'de definitieve 5.000 cursusrijen en de bestaande gezamenlijke pipeline.'),
    'Mushroom/09_vergelijking.ipynb': (
        'De recente gerichte Final-verbeteringen met Codex laten uitwerken: '
        'de verkeerde notebookverwijzing hersteld en de aanvullende '
        'ontwikkelvergelijking met bevroren resultaten toegevoegd. Ik vroeg om '
        'lesconforme verbeteringen en controleerbaar uitvoeringsbewijs. Het actieve '
        'model is niet vervangen door een experimenteel alternatief.'),
    'Mushroom/10_deployment.ipynb': (
        'De frontend van het gezamenlijke project herbouwd. De eerdere UI van '
        'Jorre en de oorspronkelijke Citi Bike-applicatieonderdelen van Raoul '
        'blijven als bronbijdragen vermeld. De notebookverwijzing is met Codex '
        'hersteld en ik liet de werkende model-/API-/websiteketen opnieuw '
        'controleren en de echte notebookuitvoer bewaren.'),
    'CitiBike/10_deployment.ipynb': (
        'De gezamenlijke frontend, waaronder de Citi Bike-weergave, herbouwd. '
        'Raouls oorspronkelijke Citi Bike-inference en applicatieonderdelen en '
        'Jorres gedeelde backend/deployment behouden hun bronvermelding. '
        'Ik liet met Codex de bestaande Citi Bike-route en het publieke '
        'voorspelformulier opnieuw controleren; dit is geen nieuwe modeltraining.'),
}
ORIGINAL_PARTS = {
    '01_clean_en_explore.ipynb': 'meegewerkt aan de data cleaning en EDA van de Mushroom-data',
    '02_data_preparation.ipynb': 'meegewerkt aan de cleaning en voorbereiding van de Mushroom-data',
}


def apply_milan_contribution(notebook, target, *, original=False):
    before = deepcopy(notebook)
    name = Path(target).as_posix()
    for cell in notebook.cells:
        if cell.cell_type != 'markdown':
            continue
        cell.source = cell.source.replace(
            'De overige teamleden bevestigen hun eigen aandeel en zelfstandige beheersing.',
            'De individuele bijdragen en AI-inzet worden per teamlid vastgelegd; '
            'persoonlijke beheersing blijft een aparte controle.')
        if original:
            lines = cell.source.splitlines()
            cell.source = '\n'.join(
                '- Milan Wouters: '+ORIGINAL_PARTS[Path(name).name]+'.'
                if line.startswith('- Student(en): vul hier') else line for line in lines)
    if original:
        original_part = ORIGINAL_PARTS[Path(name).name]
        part = original_part[0].upper()+original_part[1:]+'.'
        links = ('Zie [bijdragen](../../Final/BIJDRAGEN.md) en '
                 '[GenAI Disclosure](../../Final/GENAI_DISCLOSURE.md). '
                 'Dit is de oorspronkelijke studie; de definitieve Final-vergelijking '
                 'gebruikt een andere, voorgeschreven bron met 5.000 cursusrijen.')
    elif name in FINAL_PARTS:
        part = FINAL_PARTS[name]
        links = ('Zie [bijdragen](../BIJDRAGEN.md), '
                 '[GenAI Disclosure](../GENAI_DISCLOSURE.md) en de '
                 '[uitgevoerde eindcontrole](../../solutions/milan/11_Final_eindcontrole.ipynb).')
    else:
        return notebook != before
    text = ('## Milan Wouters — mijn bijdrage\n\n'
            '**Eigen aandeel, bevestigd op 15 oktober 2026:** '+part+'\n\n'
            '**GenAI:** de code, structuur en documentatie zijn met hulp van '
            'OpenAI Codex uitgewerkt. Bij de recente review, Final-aanpassingen '
            'en technische eindcontrole was die hulp substantieel; ook de '
            'uitvoering van die controles gebeurde met Codex. Dit wordt niet '
            'als volledig zelfstandig geschreven code voorgesteld. Ik blijf '
            'verantwoordelijk voor het kunnen toelichten van mijn onderdeel.\n\n'+links)
    cell = nbformat.v4.new_markdown_cell(text,id=CELL_ID,
        metadata={'contribution':{'person':'Milan Wouters','confirmed_on':'2026-10-15',
                                  'basis':'User-confirmed contribution; substantial Codex assistance disclosed'}})
    matches = [i for i,c in enumerate(notebook.cells) if c.get('id')==CELL_ID]
    assert len(matches)<=1, 'Duplicate Milan contribution blocks'
    if matches:
        notebook.cells[matches[0]]=cell
    else:
        notebook.cells.insert(0,cell)
    assert [c for c in before.cells if c.cell_type=='code']==[
        c for c in notebook.cells if c.cell_type=='code'], 'Execution must be preserved'
    return notebook != before


def main():
    changed=[]
    for path in sorted((ROOT/'Final').rglob('*.ipynb')):
        notebook=nbformat.read(path,as_version=4)
        if apply_milan_contribution(notebook,path.relative_to(ROOT/'Final')):
            nbformat.validate(notebook)
            write_markdown_only(path,notebook)
            changed.append(path.relative_to(ROOT).as_posix())
    for name in ORIGINAL_PARTS:
        path=ROOT/'SolutionMilan/MushroomDataset'/name
        notebook=nbformat.read(path,as_version=4)
        if apply_milan_contribution(notebook,name,original=True):
            nbformat.validate(notebook)
            write_markdown_only(path,notebook)
            changed.append(path.relative_to(ROOT).as_posix())
    print('Contribution Markdown updated:',len(changed),'notebooks')


def write_markdown_only(path,notebook):
    # Preserve even the original JSON representation of every executed cell.
    original=json.loads(path.read_text(encoding='utf-8'))
    code_cells=iter(c for c in original['cells'] if c['cell_type']=='code')
    serialized=json.loads(nbformat.writes(notebook))
    original['cells']=[next(code_cells) if c['cell_type']=='code' else c
                       for c in serialized['cells']]
    path.write_text(json.dumps(original,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
