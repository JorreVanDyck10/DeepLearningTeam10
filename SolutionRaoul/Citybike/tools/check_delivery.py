"""Controleer de geleverde notebooks en echte uitvoerrapporten zonder herselectie."""
from pathlib import Path
import json
import sys
import nbformat

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from citibike.config import REPORTS,atomic_json,now,sha256_file
from citibike.notebooks import NAMES
from citibike.reproduction import check_protected_files


if __name__=='__main__':
    notebooks=sorted((ROOT/'notebooks').glob('*.ipynb'))
    assert [p.name for p in notebooks]==NAMES,'Verwacht precies de negen genummerde notebooks'
    results=[]
    for path in notebooks:
        notebook=nbformat.read(path,as_version=4)
        nbformat.validate(notebook)
        cells=[c for c in notebook.cells if c.cell_type=='code']
        assert cells and all(c.execution_count is not None for c in cells),path.name
        errors=[o for c in cells for o in c.outputs if o.output_type=='error']
        assert not errors,(path.name,errors)
        assert 'Messi10' in notebook.cells[0].source, path.name
        assert 'Raoul Buffels' in notebook.cells[0].source and 'Jorre Van Dyck' in notebook.cells[0].source, path.name
        assert all(i > 0 and notebook.cells[i-1].cell_type == 'markdown'
                   for i,c in enumerate(notebook.cells) if c.cell_type == 'code'), path.name
        if path.name.startswith('02_'):
            assert any('prepare_model_data(SOURCE)' in c.source for c in cells)
            assert not any('report(' in c.source or 'MODELS' in c.source for c in cells)
            assert not any('image/png' in o.get('data',{}) for c in cells for o in c.outputs)
        results.append({'name':path.name,'executed_code_cells':len(cells),
                        'outputs':sum(len(c.outputs) for c in cells),
                        'sha256':sha256_file(path),'error_cells':0})
    tests=json.loads((REPORTS/'test_results.json').read_text(encoding='utf-8'))
    assert tests['failed']==0 and tests['passed']==tests['tests']
    verification=json.loads((REPORTS/'verification.json').read_text(encoding='utf-8'))
    assert all(v is True for v in verification.values() if isinstance(v,bool))
    assert verification['offline_app_predictions_equal']
    final=json.loads((REPORTS/'final_evaluation.json').read_text(encoding='utf-8'))
    receipt=json.loads((REPORTS/'model_artifact.json').read_text(encoding='utf-8'))
    assert receipt['sha256']==final['model_sha256']==sha256_file(Path(receipt['path']))
    correction_path=REPORTS/'frontend_correction.json'
    correction=json.loads(correction_path.read_text(encoding='utf-8')) if correction_path.exists() else {}
    report={'completed_at':now(),'notebooks':results,'tests_passed':tests['passed'],
            'model_sha256':receipt['sha256'],'selected_model':final['selected_model'],
            'verification_complete':True,
            'protected_files':check_protected_files(ROOT),
            'preparation_independent':json.loads((REPORTS/'preparation_independence.json').read_text(encoding='utf-8')),
            'AWS_scope':'Lokaal Mushroom-SageMaker-notebook en tuningexports gevonden; cloudprovenance nog bevestigen.',
            'live_hosting_verified':bool(correction.get('netlify_browser_prediction_verified')),
            'frontend':'DeepLearningTeam10/frontend',
            'automatic_model_rollout_connected':False,
            'local_retraining_pipeline':json.loads((REPORTS/'retraining_pipeline_latest.json').read_text(encoding='utf-8')),
            'github_runner_registered':False,'github_retraining_run_verified':False}
    atomic_json(REPORTS/'delivery_check.json',report)
    print(json.dumps(report,indent=2),flush=True)
