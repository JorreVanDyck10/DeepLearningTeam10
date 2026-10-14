"""Execute the repaired PyCaret section, preserving Andrew's earlier research outputs."""
import copy
import os
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
import pandas as pd

FOLDER=Path(__file__).resolve().parent


def run():
    path=FOLDER/'mushroom_project.ipynb'
    nb=nbformat.read(path,as_version=4)
    # Locate the original section by source rather than fragile numeric cell positions.
    start=next(i for i,c in enumerate(nb.cells) if c.cell_type=='markdown' and '## 3 - Pycaret comparison' in c.source)
    code_indices=[i for i in range(start,len(nb.cells)) if nb.cells[i].cell_type=='code']
    assert len(code_indices)==3
    sources=['''from pathlib import Path
import sys
import pandas as pd
from IPython.display import display
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'MushroomFinal/mushroom_project_dataset.csv').exists())
sys.path.insert(0, str(ROOT))
import pycaret
from pycaret.classification import ClassificationExperiment
from MushroomFinal.compare_correct_dataset import load_data, pycaret_run, OUT
print('PyCaret', pycaret.__version__, '| Python', sys.version.split()[0])''',
'''raw, X, y, train, test, folds = load_data()
assert len(X) == 5000 and X.shape[1] == 11
assert 'jumbled_noise_0' not in X and 'jumbled_noise_1' in X
print('Bron: MushroomFinal/mushroom_project_dataset.csv')
print('Train:', len(train), '| Test:', len(test), '| Doel: 0=e, 1=p')
display(X.head())''',
'''# Actual training: compare the three requested families, then all available fast families.
# The helper supplies the SAME 4,100/900 rows and five shuffled folds to PyCaret.
# PyCaret 3.3.2 has no setup(positive_label=...); labels are explicitly encoded 0=e, 1=p.
pycaret_run()
comparison_results = pd.read_csv(OUT / 'pycaret_three.csv', index_col=0)
all_comparison_results = pd.read_csv(OUT / 'pycaret_all.csv', index_col=0)
display(comparison_results)
display(all_comparison_results)
display(pd.read_csv(OUT / 'pycaret_test_metrics.csv'))''']
    section=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(s) for s in sources])
    section.metadata.kernelspec={'name':'python3','display_name':'Python 3.11 (PyCaret 3.3.2)','language':'python'}
    env=dict(os.environ);env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env['PATH']
    NotebookClient(section,timeout=1200,kernel_name='python3',resources={'metadata':{'path':str(FOLDER)}}).execute(env=env)
    for index,cell in zip(code_indices,section.cells): nb.cells[index]=copy.deepcopy(cell)
    three=pd.read_csv(FOLDER/'comparison_correct/pycaret_three.csv',index_col=0)
    all_models=pd.read_csv(FOLDER/'comparison_correct/pycaret_all.csv',index_col=0)
    conclusion=next(c for c in nb.cells if c.cell_type=='markdown' and
        (c.source.strip()=='RESULTS AND CONCLUSION' or c.source.startswith('### Results and conclusion — executed')))
    conclusion.source=f'''### Results and conclusion — executed 14 October 2026

PyCaret 3.3.2 was executed in an isolated Python 3.11 environment on the provided
5,000-row CSV. `jumbled_noise_0` is excluded; `jumbled_noise_1` is retained.
Exactly the same 4,100 training rows, 900 test rows and five shuffled stratified
training folds as the shared comparison are used. Positive class is encoded as `1=p`.

Among the three default algorithms, **{three.iloc[0]['Model']}** ranks first on
mean CV Macro F1 (**{three.iloc[0]['Macro F1']:.4f}**). Across the available fast
families, **{all_models.iloc[0]['Model']}** ranks first (**{all_models.iloc[0]['Macro F1']:.4f}**).
These PyCaret defaults are not the manually tuned pipelines from section 2.
They must therefore not be presented as new scores for those tuned artifacts.
LightGBM did not complete in this Windows environment (a separately verified native
access-violation error). The ranking covers the successful families; the shared report
records this exclusion rather than assigning a misleading score.

The earlier section 2 outputs remain the original research record. Their proximity
imputation was learned before CV; its tuned RF was subsequently refitted on raw inputs.
The new shared comparison learns all imputations inside each training fold and keeps
complete preprocessing in the exported pipeline. Its selection and error analysis
are documented in [the corrected report](../secondary_mushroom/09_vergelijkingsrapport.md).
The test set has already been inspected in the original notebook; these are development
results, not an untouched final test. The former 61,069-row UCI experiment is superseded.

Contribution: Andrew Noeyens wrote the original notebook; Codex, at Jorre's request,
repaired and executed this PyCaret section and documented the validation correction.
'''
    # Correct misleading setup description, while retaining the author's original EDA.
    nb.cells[code_indices[2]-1].source='''The helper configures PyCaret with explicit training and test data from the shared
split. Median/mode imputation and encoding are fitted within training folds. Macro F1
is calculated inside a custom callable (rather than relying on add_metric kwargs),
and checked against a majority reference. `positive_label` was removed because it is not a
PyCaret 3.3.2 setup argument; explicit binary encoding makes class p positive.
Use `requirements-comparison.txt` and Python 3.11. The full setup and training code is
in `compare_correct_dataset.py`; no pre-imputed matrix or test labels enter training.'''
    banner='''**Data correction — 14 October 2026.** The course source is the uploaded
`MushroomFinal/mushroom_project_dataset.csv` (5,000 rows), not the 61,069-row UCI download.
Sections 1–2 preserve Andrew's original outputs; section 3 has now actually been
executed using PyCaret 3.3.2 / Python 3.11. For the common comparison, corrected CV and
final development choice, see [09_compare_models.ipynb](../secondary_mushroom/09_compare_models.ipynb).
'''
    if not any(c.cell_type=='markdown' and c.source.startswith('**Data correction') for c in nb.cells):
        nb.cells.insert(1,nbformat.v4.new_markdown_cell(banner))
    nb.metadata.kernelspec=section.metadata.kernelspec
    nb.metadata.language_info={'name':'python','version':'3.11.0'}
    nbformat.validate(nb);nbformat.write(nb,path)
    print('Repaired PyCaret cells actually executed; original research outputs preserved.')


if __name__=='__main__': run()
