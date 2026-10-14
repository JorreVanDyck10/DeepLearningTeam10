"""Execute final checks/reproductions; preserve source studies and SageMaker output."""
import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ['Mushroom/01_EDA.ipynb', 'Mushroom/02_voorbereiding.ipynb', 'Mushroom/03_baseline.ipynb',
           'Mushroom/04_AutoML.ipynb', 'Mushroom/05_Random_Forest.ipynb',
           'Mushroom/06_Gradient_Boosting.ipynb', 'Mushroom/07_Logistic_Regression.ipynb',
           'Mushroom/09_vergelijking.ipynb', 'Mushroom/10_deployment.ipynb',
           'CitiBike/03_baseline.ipynb', 'CitiBike/04_Ridge.ipynb', 'CitiBike/10_deployment.ipynb']


def execute(names):
    receipt_path = ROOT/'Final/execution_checks.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else []
    env = dict(os.environ); env['PATH'] = str(Path(sys.executable).parent)+os.pathsep+env['PATH']
    for name in names:
        path = ROOT/'Final'/name
        nb = nbformat.read(path, as_version=4)
        print('Execute', name, flush=True); started = perf_counter()
        NotebookClient(nb, timeout=1200, kernel_name='python3', resources={'metadata':{'path':str(path.parent)}}).execute(env=env)
        nbformat.validate(nb); nbformat.write(nb,path)
        rows = [c for c in nb.cells if c.cell_type == 'code']
        receipt = [r for r in receipt if r['notebook'] != name]
        receipt.append({'notebook':name, 'code_cells':len(rows), 'executed':sum(c.execution_count is not None for c in rows),
                        'errors':sum(o.output_type=='error' for c in rows for o in c.outputs),
                        'seconds':perf_counter()-started})
        (ROOT/'Final/execution_checks.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print('Final checks executed:', len(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebooks', nargs='*'); args = parser.parse_args()
    execute(args.notebooks or DEFAULT)
