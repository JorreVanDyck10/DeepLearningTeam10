"""Apply the approved report-only additions, preserving code and saved outputs."""
from copy import deepcopy
import csv
import hashlib
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'Final/Mushroom/reports/milan_development_summary_v3.csv'
SOURCE_SHA256_LF = '6e6e3f089a24554029fcd2b804e5527bba094a3951788365e68e5ebc50a8d5b4'
STUDY_COMMIT = 'd8fe8dc5516372316e7d6d09e5e2ecdab457ce1b'
STUDY_URL = 'https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/' + STUDY_COMMIT + '/solutions/milan/'
CELL_ID = 'milan-development-review-v3'
MODELS = {
    'rf300_reference_default': 'Huidige Final-RF300',
    'rf_selection_default': 'RF300 met interne featureselectie',
    'voting_f1': 'Voting met binnen training gekozen F1-drempel',
}


def development_section():
    digest = hashlib.sha256(SOURCE.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    assert digest == SOURCE_SHA256_LF, 'Development evidence changed: review before updating Final'
    with SOURCE.open(encoding='utf-8', newline='') as handle:
        rows = {row['model']: row for row in csv.DictReader(handle)}
    for name in MODELS:
        row = rows[name]
        assert int(row['folds']) == 5, f'Incomplete development comparison: {name}'
        assert sum(int(row[k + '_total']) for k in ['tn', 'fp', 'fn', 'tp']) == 4100

    lines = [
        '## Aanvullend ontwikkelonderzoek — huidige deployment behouden', '',
        'Deze aanvullende vergelijking beoordeelt drie procedures op dezelfde vijf',
        'buitenfolds van de oorspronkelijke **4.100 trainingsrijen**. Configuraties,',
        'voorbereiding en eventuele votinggewichten/drempels worden binnen training',
        'gekozen. De bekende **900 ontwikkeltestrecords zijn hiervoor niet opnieuw',
        'gescoord**. Dit blijft verder onderzoek op eerder gebruikte data, geen',
        'nieuwe onafhankelijke eindtest.', '',
        '| Procedure | Macro F1 | Recall giftig | Giftig gemist (FN) | Vals alarm (FP) |',
        '|---|---:|---:|---:|---:|',
    ]
    for name, label in MODELS.items():
        row = rows[name]
        lines.append(f"| {label} | {float(row['macro_f1_mean']):.4f} | "
                     f"{float(row['recall_p_mean']):.2%} | {int(row['fn_total'])} | {int(row['fp_total'])} |")
    lines += [
        '', 'Macro F1 en recall zijn foldgemiddelden. FN en FP zijn opgetelde',
        'buitenfoldfouten over 4.100 rijen. **Vergelijk deze aantallen niet met de',
        '121 FN op de 900 ontwikkeltestrecords uit het eerdere rapport:** dat is',
        'een andere evaluatieset.', '',
        'De featureselectievariant behoudt de elf publieke invoervelden en negeert',
        'in de uiteindelijke trainingsfit intern sporenkleur en het extra ruisveld.',
        'Hij behandelt categorische missings expliciet als een eigen categorie.',
        'De afzonderlijke buitenfolds kiezen niet allemaal dezelfde veldselectie.',
        'Voting combineert kansen van modellen; de binnenfolds kiezen uit vooraf',
        'vastgelegde gewichten. Dit sluit aan op de lessen over voorbereiding,',
        'featureselectie, boosting, ensembles en modelkwaliteit.', '',
        'De RF-variant verlaagt hier beide foutaantallen tegenover de huidige RF300.',
        'Voting haalt de hogere macro F1 en mist minder giftige gevallen, maar geeft',
        'meer valse alarmen dan de huidige RF300. Dit zijn ontwikkelresultaten;',
        'ze bewijzen geen algemene superioriteit of veiligheid.', '',
        '**Deze rapportaanvulling vervangt het actieve model niet.** De bestaande',
        'pipeline, API, website, beslisgrens en automatische training blijven de',
        'huidige Final-RF300 gebruiken. Een latere modelwissel vereist een gezamenlijke',
        'aanpassing en controle van training, export, metadata en publicatie.', '',
        'De studie en deze rapportaanvulling zijn met substantiële hulp van Codex',
        'gemaakt op verzoek van de gebruiker. Ze schrijven Milan geen onbevestigde',
        'zelfstandige bijdrage toe en vervangen geen menselijke inhoudelijke review.', '',
        'De [bevroren resultatensamenvatting](reports/milan_development_summary_v3.csv)',
        'is bij Final opgenomen; voor het tonen of herbouwen van deze aanvulling',
        'is de lokale Milan-map niet nodig. De bronhash wordt bij herbouwen gecontroleerd.', '',
        f'Bronnen op de vastgelegde onderzoekscommit: [volledige resultaten en beperkingen]({STUDY_URL}RESULTATEN_V3.md),',
        f'[onderzoeksafspraken]({STUDY_URL}STUDIE_V3.md),',
        f'[featureselectie en boosting]({STUDY_URL}08_featureselectie_en_boosting.ipynb),',
        f'[voting en vergelijking]({STUDY_URL}09_voting_en_vergelijking.ipynb) en',
        f'[foutanalyse]({STUDY_URL}10_foutanalyse_v3.ipynb).',
    ]
    return nbformat.v4.new_markdown_cell(
        '\n'.join(lines), id=CELL_ID,
        metadata={'review_addition': {'source': SOURCE.relative_to(ROOT).as_posix(),
                  'source_sha256_lf': digest, 'original_study_commit': STUDY_COMMIT,
                  'original_source': 'solutions/milan/results_v3/summary.csv',
                  'scope': 'Report only; no training or deployment'}})


def apply_review_additions(notebook, target):
    """Idempotently repair the two links and append one marked report section."""
    name = Path(target).as_posix()
    if name not in ['Mushroom/09_vergelijking.ipynb', 'Mushroom/10_deployment.ipynb']:
        return False
    before = deepcopy(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'markdown':
            cell.source = cell.source.replace('](09_compare_models.ipynb)', '](09_vergelijking.ipynb)')
            cell.source = cell.source.replace('[09_compare_models.ipynb](09_vergelijking.ipynb)',
                                              '[09_vergelijking.ipynb](09_vergelijking.ipynb)')
    if name == 'Mushroom/09_vergelijking.ipynb':
        section = development_section()
        existing = [i for i, cell in enumerate(notebook.cells) if cell.get('id') == CELL_ID]
        assert len(existing) <= 1, 'Duplicate review sections'
        if existing:
            notebook.cells[existing[0]] = section
        else:
            notebook.cells.append(section)
    return notebook != before


def main():
    for target in ['Mushroom/09_vergelijking.ipynb', 'Mushroom/10_deployment.ipynb']:
        path = ROOT / 'Final' / target
        notebook = nbformat.read(path, as_version=4)
        if apply_review_additions(notebook, target):
            nbformat.validate(notebook)
            nbformat.write(notebook, path)
            print('Updated report:', target)
        else:
            print('Already current:', target)


if __name__ == '__main__':
    main()
