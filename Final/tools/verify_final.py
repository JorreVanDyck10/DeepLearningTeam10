"""Check portable source preservation, final notebooks and required entry points."""
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
import nbformat

ROOT = Path(__file__).resolve().parents[2]


def verify_local_links(text, path):
    for url in re.findall(r'\]\(([^)]+)\)', text):
        url = url.strip().strip('<>')
        if urlsplit(url).scheme or url.startswith('#'):
            continue
        target = unquote(urlsplit(url).path)
        assert (path.parent / target).resolve().exists(), (path, url)


def verify():
    manifest = json.loads((ROOT/'Final/source_manifest.json').read_text())
    for row in manifest['copies']:
        digest = hashlib.sha256((ROOT/row['source']).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        assert digest == row['source_sha256_lf'], row['source']
        assert (ROOT/row['final']).is_file(), row['final']
    paths = sorted((ROOT/'Final').rglob('*.ipynb'))
    assert len(paths) == 19, len(paths)
    for path in paths:
        nb = nbformat.read(path, as_version=4); nbformat.validate(nb)
        assert not any(output.output_type == 'error' for cell in nb.cells if cell.cell_type == 'code' for output in cell.outputs), path
        assert any(c.cell_type == 'code' for c in nb.cells), path
        for cell in nb.cells:
            if cell.cell_type == 'markdown':
                verify_local_links(cell.source, path)
    prep = nbformat.read(ROOT/'Final/Mushroom/02_voorbereiding.ipynb', as_version=4)
    assert all(not re.search(r'plt\.|sns\.|\.plot\(', c.source) for c in prep.cells if c.cell_type == 'code')
    for name in ['README.md','INLEVERCHECK.md','UPLOADTEKST.md','AUTOMATISERING.md','Presentatie/Team10_presentatie.pptx','Presentatie/SPREEKPLAN.md']:
        assert (ROOT/'Final'/name).is_file(), name
    runs = json.loads((ROOT/'Final/execution_checks.json').read_text())
    assert len(runs) == 12 and all(r['executed'] == r['code_cells'] and r['errors'] == 0 for r in runs)
    for path in (ROOT/'Final').rglob('*.md'):
        verify_local_links(path.read_text(encoding='utf-8'), path)
    print('Final verified: 19 notebooks, stored execution receipts, original source hashes and local links.')


if __name__ == '__main__': verify()
