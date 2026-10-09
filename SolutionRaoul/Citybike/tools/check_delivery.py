"""Controleer de geleverde notebooks en echte uitvoerrapporten zonder herselectie."""
from pathlib import Path
import json
import sys
import nbformat

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from citibike.config import REPORTS,atomic_json,now,sha256_file


if __name__=='__main__':
    notebooks=sorted((ROOT/'notebooks').glob('*.ipynb'))
    assert len(notebooks)==8,'Verwacht overzicht plus zeven ondersteunende notebooks'
    results=[]
    for path in notebooks:
        notebook=nbformat.read(path,as_version=4)
        cells=[c for c in notebook.cells if c.cell_type=='code']
        assert cells and all(c.execution_count is not None for c in cells),path.name
        errors=[o for c in cells for o in c.outputs if o.output_type=='error']
        assert not errors,(path.name,errors)
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
    report={'completed_at':now(),'notebooks':results,'tests_passed':tests['passed'],
            'model_sha256':receipt['sha256'],'selected_model':final['selected_model'],
            'verification_complete':True,'AWS_scope':'afzonderlijke latere taak'}
    atomic_json(REPORTS/'delivery_check.json',report)
    print(json.dumps(report,indent=2),flush=True)
