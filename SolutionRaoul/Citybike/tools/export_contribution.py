"""Exporteer de deelbare bijdrage zonder data, modellen of Python-omgeving."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
import nbformat


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(source, destination):
    source, destination = source.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError(f'Doel bestaat al: {destination}')
    destination.mkdir(parents=True)
    files = [source / name for name in ['README.md', '.gitignore', 'app.py', 'run.py',
                                      'requirements.txt', 'requirements-lock.txt']]
    for directory in ['citibike', 'notebooks', 'reports', 'tests', 'tools', '.streamlit']:
        files.extend(path for path in (source/directory).rglob('*')
                     if path.is_file() and '__pycache__' not in path.parts
                     and path.name != 'secrets.toml'
                     and path.suffix not in ['.pyc', '.tmp', '.log', '.joblib', '.pkl', '.parquet'])
    records = []
    for path in sorted(files):
        relative = path.relative_to(source)
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        records.append({'path': relative.as_posix(), 'source_sha256': sha(path),
                        'source_bytes': path.stat().st_size})
    disclosure = (
        '## Bijdrage Raoul Buffels — DeepLearningTeam10\n\n'
        'Deze Citi Bike-uitwerking is aangeleverd als de bijdrage van **Raoul Buffels**. '
        'Teamleden: Jorre Van Dyck, Milan Wouters, Raoul Buffels en Andrew Noeyens. '
        'De concrete menselijke taakverdeling moet het team zelf aanvullen. '
        '**AI-gebruik:** Codex hielp met implementatie, uitleg, tests, uitvoering en verpakking.\n\n'
        'De codecellen zijn uitgevoerd in de oorspronkelijke lokale `Citybike_full`-omgeving '
        'op 9 oktober 2026. Deze export behoudt die echte uitvoer; alleen deze Markdown-cel '
        'is toegevoegd. Lokale paden, timings en fingerprints in de rapporten beschrijven '
        'de oorspronkelijke uitvoering. Start voor heruitvoering vanuit `SolutionRaoul/Citybike` '
        'of de map `notebooks` met de bijbehorende Python 3.11-omgeving. '
        'AWS is een afzonderlijke latere taak.'
    )
    for path in (destination/'notebooks').glob('*.ipynb'):
        notebook = nbformat.read(path, as_version=4)
        notebook.cells.insert(0, nbformat.v4.new_markdown_cell(disclosure))
        nbformat.write(notebook, path)
    readme = (destination/'README.md').read_text(encoding='utf-8')
    readme = readme.replace(str(source), str(destination)).replace('Citybike_full/', 'SolutionRaoul/Citybike/')
    readme = readme.replace('Teamnaam, teamleden en echte bijdragen: **zelf invullen**, ook in de notebooks.',
                            'Team: **DeepLearningTeam10** — Jorre Van Dyck, Milan Wouters, Raoul Buffels en Andrew Noeyens.\n'
                            'Deze map bevat de Citi Bike-bijdrage van **Raoul Buffels**. De concrete menselijke\n'
                            'taakverdeling moet het team zelf aanvullen; er worden geen bijdragen verzonnen.')
    notice = (
        '# Raoul Buffels — Citi Bike\n\n'
        'Dit is de deelbare export van de volledig uitgevoerde Citi Bike-uitwerking. '
        'Begin bij [`notebooks/citybike_analysis.ipynb`](notebooks/citybike_analysis.ipynb). '
        'De code, acht uitgevoerde notebooks, figuren, tests, requirements, manifesten, '
        'modelkaart en lokale trainingsmetingen staan in deze map.\n\n'
        f'Oorspronkelijke uitvoering: `{source}`. Rapporten behouden de oorspronkelijke '
        'paden en fingerprints als provenance; ze zijn geen bewijs van een nieuwe run in '
        'deze repository. `EXPORT_MANIFEST.json` legt de gekopieerde bestanden en eventuele '
        'documentatieaanpassingen vast. Kerncode, voorspelfunctie en modelregels blijven identiek.\n\n'
        'Op deze computer zijn `data`, `models` en `.venv` lokaal via directory-junctions '
        'gekoppeld aan de originele uitwerking. Die koppelingen en hun inhoud worden niet '
        'gecommit. Op een andere computer maak je een eigen omgeving en download/verwerk '
        'je de bronnen volgens de commando\'s hieronder. Het modelbestand van circa 647 MiB '
        'blijft lokaal; training is reproduceerbaar met de vastgelegde code en configuratie.\n\n'
        'De bestaande teamwebsite, Citi Bike-baseline, API en bijdragen van teamgenoten '
        'worden door deze export niet vervangen. AWS-training blijft een latere taak.\n\n'
        '## Lokale koppelingen opnieuw maken op Raouls computer\n\n'
        'Alleen uitvoeren als de oorspronkelijke mappen aanwezig zijn en de doelmappen '
        'nog niet bestaan:\n\n```powershell\n'
        f"Set-Location '{destination}'\n"
        f"New-Item -ItemType Junction -Path 'data' -Target '{source / 'data'}'\n"
        f"New-Item -ItemType Junction -Path 'models' -Target '{source / 'models'}'\n"
        f"New-Item -ItemType Junction -Path '.venv' -Target '{source / '.venv'}'\n"
        '```\n\n'
    )
    (destination/'README.md').write_text(notice+readme, encoding='utf-8')
    for record in records:
        target = destination/record['path']
        record.update(export_sha256=sha(target), export_bytes=target.stat().st_size,
                      documentation_changed=sha(target) != record['source_sha256'])
    attributes = destination/'.gitattributes'
    attributes.write_text('# Preserve exact bytes for recorded SHA-256 fingerprints.\n* -text\n', encoding='utf-8')
    records.append({'path': '.gitattributes', 'generated': True,
                    'export_sha256': sha(attributes), 'export_bytes': attributes.stat().st_size})
    manifest = {'exported_at': datetime.now(timezone.utc).isoformat(),
                'source': str(source), 'destination': str(destination),
                'author': 'Raoul Buffels', 'repository': 'JorreVanDyck10/DeepLearningTeam10',
                'excluded': ['data', 'models', '.venv', 'logs', '__pycache__', 'secrets'],
                'executed_notebook_outputs_preserved': True,
                'files': records}
    (destination/'EXPORT_MANIFEST.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'{len(records)} bestanden geëxporteerd naar {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.destination)
