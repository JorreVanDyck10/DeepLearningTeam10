"""Real Final checks used by the executed evidence notebook; no publication."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter, sleep
import urllib.error
import urllib.request

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = HERE/'.generated/final_validation'
EVIDENCE = HERE/'eindcontrole'
KERNEL = 'final-validation-311'
SERVING = Path(os.environ.get('FINAL_SERVING_PYTHON', HERE/'.generated/validation_serving313/Scripts/python.exe'))
NODE = Path(os.environ.get('FINAL_NODE', Path('C:/Users/steve/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe')))
API = 'https://deeplearningteam10-api.onrender.com'
WEBSITE = 'https://10messi10.netlify.app'


def write(name, data):
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE/name).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def fingerprint(path):
    content=Path(path).read_bytes()
    if Path(path).suffix in ['.py','.json','.csv','.md','.ipynb','.yml','.txt','.js','.html','.css']:
        content=content.replace(b'\r\n',b'\n')
    return hashlib.sha256(content).hexdigest()


def configure():
    WORK.mkdir(parents=True,exist_ok=True)
    for name in ['runtime','ipython','config','tmp']:
        (WORK/name).mkdir(exist_ok=True)
    os.environ.update(JUPYTER_PATH=str(WORK/'jupyter'), JUPYTER_RUNTIME_DIR=str(WORK/'runtime'),
        IPYTHONDIR=str(WORK/'ipython'),JUPYTER_CONFIG_DIR=str(WORK/'config'),
        PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='module://matplotlib_inline.backend_inline',
        RAOUL_CITIBIKE_MODEL_DIR=str(HERE/'.generated/validation_city'),
        TMP=str(WORK/'tmp'),TEMP=str(WORK/'tmp'),TMPDIR=str(WORK/'tmp'))
    assert (WORK/'tmp').resolve().is_relative_to(HERE.resolve())
    kernel=WORK/'jupyter/kernels'/KERNEL
    kernel.mkdir(parents=True,exist_ok=True)
    (kernel/'kernel.json').write_text(json.dumps({
        'argv':[sys.executable,'-B','-m','ipykernel_launcher','-f','{connection_file}'],
        'display_name':'Final-validatie Python 3.11','language':'python'}),encoding='utf-8')
    sys.path.insert(0,str(ROOT))


def environment():
    configure()
    expected={'numpy':'1.26.4','pandas':'2.1.4','scipy':'1.11.4','scikit-learn':'1.4.2',
              'joblib':'1.3.2','matplotlib':'3.7.5','seaborn':'0.13.2','nbformat':'5.11.1','nbclient':'0.11.0','ipykernel':'6.29.5',
              'ipython':'8.37.0','matplotlib-inline':'0.1.7'}
    versions={name:importlib.metadata.version(name) for name in expected}
    assert sys.version_info[:2]==(3,11), sys.version
    assert versions==expected,versions
    serving_probe=command([SERVING,'-B','-c',
        "import json,sys,importlib.metadata as m; print(json.dumps({'python':sys.version.split()[0], 'versions':{k:m.version(k) for k in ['fastapi','uvicorn','pandas','scikit-learn','joblib','numpy','scipy','httpx']}}))"])
    assert serving_probe['passed'],serving_probe['output']
    serving=json.loads(serving_probe['output'])
    for name,version in {'fastapi':'0.115.12','uvicorn':'0.20.0','pandas':'3.0.2',
                         'scikit-learn':'1.8.0','joblib':'1.5.3'}.items():
        assert serving['versions'][name]==version,serving
    files=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    protected=[name for name in files if name.startswith(('Final/','backend/','frontend/','.github/'))]
    snapshot={name:fingerprint(ROOT/name) for name in protected}
    write('protected_before.json',snapshot)
    result={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version.split()[0],
            'versions':versions,'serving_environment':serving,
            'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'tracked_changes':subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=ROOT,text=True).splitlines(),
            'protected_files':len(snapshot),'passed':True}
    write('environment.json',result)
    return result


def command(args, *, serving=False, timeout=1200):
    env=dict(os.environ)
    env['PYTHONPATH']=str(ROOT)
    started=perf_counter()
    done=subprocess.run([str(a) for a in args],cwd=ROOT,env=env,capture_output=True,text=True,
                        encoding='utf-8',errors='replace',timeout=timeout)
    return {'exit_code':done.returncode,'seconds':perf_counter()-started,'output':done.stdout+done.stderr,
            'passed':done.returncode==0}


def final_structure():
    result=command([sys.executable,'-B','Final/tools/verify_final.py'])
    write('final_structure.json',result)
    assert result['passed'], result['output']
    return result


def python_suite():
    result=command([SERVING,'-B','-m','unittest','discover','-s','tests','-v'],serving=True)
    count=re.search(r'Ran (\d+) tests?',result['output'])
    result['tests_run']=int(count[1]) if count else 0
    result['passed']=result['passed'] and result['tests_run']>=28 and not re.search(r'\bskipped=',result['output'])
    write('python_tests.json',result)
    assert result['passed'],result['output']
    return result


def frontend_suite():
    runs=[]
    for name in ['config.js','api.js','app.js','citibike.js','mushroom.js']:
        runs.append(command([NODE,'--check','frontend/'+name]))
    tests=command([NODE,'--test','--test-reporter=tap','tests/test_api_client.cjs','tests/test_api_config.cjs',
                   'tests/test_citibike_frontend.cjs','tests/test_mushroom_frontend.cjs'])
    count=re.search(r'^# tests (\d+)$',tests['output'],re.M)
    result={'syntax_checks':len(runs),'tests_run':int(count[1]) if count else 0,
            'passed':all(r['passed'] for r in runs) and tests['passed'] and bool(count),
            'output':tests['output']}
    write('frontend_tests.json',result)
    assert result['passed'],result['output']
    return result


def notebooks():
    import nbformat
    from nbclient import NotebookClient
    from Final.tools.execute_notebooks import DEFAULT
    rows=[]
    for name in DEFAULT:
        source=ROOT/'Final'/name
        notebook=nbformat.read(source,as_version=4)
        for cell in notebook.cells:
            if cell.cell_type=='code':
                cell.outputs=[]; cell.execution_count=None
        notebook.metadata.kernelspec={'name':KERNEL,'display_name':'Final-validatie Python 3.11','language':'python'}
        started=perf_counter(); problem=None
        try:
            NotebookClient(notebook,timeout=1200,kernel_name=KERNEL,
                resources={'metadata':{'path':str(source.parent)}}).execute()
        except Exception as error:
            problem=str(error)
        target=WORK/'notebooks'/name; target.parent.mkdir(parents=True,exist_ok=True)
        nbformat.validate(notebook); nbformat.write(notebook,target)
        code=[c for c in notebook.cells if c.cell_type=='code']
        errors=sum(o.output_type=='error' for c in code for o in c.outputs)
        plots=sum('image/png' in o.get('data',{}) for c in code for o in c.outputs)
        if name=='Mushroom/01_EDA.ipynb' and plots<4:
            problem=problem or 'Expected the EDA figures to be embedded in the executed notebook.'
        rows.append({'notebook':name,'code_cells':len(code),'executed':sum(c.execution_count is not None for c in code),
            'errors':errors,'embedded_plots':plots,'seconds':round(perf_counter()-started,2),'passed':problem is None and errors==0,
            'source_sha256_lf':fingerprint(source),'executed_copy_sha256_lf':fingerprint(target),
            'error':problem})
        write('notebook_runs.json',rows)
    assert len(rows)==12 and all(r['passed'] and r['executed']==r['code_cells'] for r in rows),rows
    # Publish the fresh comparison as the requested directly inspectable notebook.
    source=ROOT/'Final/Mushroom/09_vergelijking.ipynb'
    fresh=nbformat.read(WORK/'notebooks/Mushroom/09_vergelijking.ipynb',as_version=4)
    destination=EVIDENCE/'09_vergelijking_heruitgevoerd.ipynb'
    def rewrite(match):
        url=match[1]
        if '://' in url or url.startswith('#'):
            return match[0]
        path,separator,fragment=url.partition('#')
        actual=(source.parent/path).resolve()
        relative=os.path.relpath(actual,destination.parent).replace('\\','/')
        return ']('+relative+(separator+fragment if separator else '')+')'
    for cell in fresh.cells:
        if cell.cell_type=='markdown':
            cell.source=re.sub(r'\]\(([^)]+)\)',rewrite,cell.source)
    fresh.metadata.execution_note='Fresh separate Jupyter kernel, Python 3.11; code unchanged. Markdown links adapted for this evidence copy.'
    nbformat.write(fresh,destination)
    eda=nbformat.read(WORK/'notebooks/Mushroom/01_EDA.ipynb',as_version=4)
    for cell in eda.cells:
        if cell.cell_type=='markdown':
            cell.source=re.sub(r'\]\(([^)]+)\)',rewrite,cell.source)
    eda.metadata.execution_note='Fresh Jupyter execution with embedded charts; original Final code unchanged.'
    nbformat.write(eda,EVIDENCE/'01_EDA_heruitgevoerd.ipynb')
    return rows


def export_equivalence():
    import joblib
    import numpy as np
    import pandas as pd
    from backend.mushroom_model import PortableMushroom,FEATURES
    model=PortableMushroom()
    source=ROOT/model.metadata['source_pipeline']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==model.metadata['source_pipeline_sha256']
    pipeline=joblib.load(source)
    raw=pd.read_csv(ROOT/'MushroomFinal/mushroom_project_dataset.csv')
    raw.columns=raw.columns.str.replace('-','_',regex=False)
    expected=pipeline.predict_proba(raw[FEATURES]); actual=model.predict_proba(raw[FEATURES])
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
    np.testing.assert_array_equal(model.predict(raw[FEATURES]),pipeline.predict(raw[FEATURES]))
    saved=pd.read_csv(ROOT/'backend/artifacts/mushroom_reference_predictions.csv',float_precision='round_trip')
    np.testing.assert_allclose(actual[saved.source_row,1],saved.probability_p,rtol=0,atol=1e-12)
    result={'passed':True,'input_rows':len(raw),'saved_reference_rows':len(saved),
        'max_probability_difference':float(np.max(abs(actual-expected))),
        'label_disagreements':0,'model_revision':model.metadata['source_pipeline_sha256'],
        'scope':'Export equivalence and stored-reference regression; no new independent accuracy test.'}
    write('export_equivalence.json',result)
    return result


def request_json(url,payload=None,method=None,headers=None):
    body=None if payload is None else json.dumps(payload).encode()
    for attempt in range(3):
        try:
            request=urllib.request.Request(url,data=body,method=method,
                headers={'User-Agent':'Team10-Final-Validation','Content-Type':'application/json',**(headers or {})})
            with urllib.request.urlopen(request,timeout=90) as response:
                content=response.read()
                try:
                    parsed=json.loads(content) if content else None
                except json.JSONDecodeError:
                    parsed=content.decode('utf-8')
                return {'status':response.status,'headers':{k.lower():v for k,v in response.headers.items()},
                        'body':parsed}
        except (urllib.error.URLError,TimeoutError):
            if attempt==2: raise
            sleep(2*(attempt+1))


def live_checks():
    import numpy as np
    import pandas as pd
    from backend.mushroom_model import PortableMushroom,FEATURES
    metadata=request_json(WEBSITE+'/metadata.json')['body']
    health=request_json(API+'/health')['body']
    local=PortableMushroom()
    assert health['mushroom_model_loaded'] and health['citibike_model_loaded']
    assert health['mushroom_model_revision']==metadata['source_pipeline_sha256']==local.metadata['source_pipeline_sha256']
    prediction=request_json(API+'/predict/mushroom',metadata['example'])['body']
    expected=local.predict_record(metadata['example'])
    assert abs(prediction['probability_poisonous']-expected['probability_poisonous'])<=1e-12
    assert prediction['model_revision']==health['mushroom_model_revision']
    city=request_json(API+'/predict/citibike',{'date':'2026-04-30','hour':8,'mode':'demo'})['body']
    assert len(city['daily_predictions'])==24
    values=[r['predicted_ride_starts'] for r in city['daily_predictions']]
    assert np.isfinite(values).all() and min(values)>=0
    preflight=request_json(API+'/predict/citibike',method='OPTIONS',headers={
        'Origin':WEBSITE,'Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'content-type'})
    assert preflight['status']==200
    assert preflight['headers'].get('access-control-allow-origin')==WEBSITE
    result={'passed':True,'checked_at_utc':datetime.now(timezone.utc).isoformat(),'website':WEBSITE,'api':API,
        'health':health,'mushroom_prediction':prediction,'city_prediction':city,'cors_status':preflight['status'],
        'same_local_and_live_revision':True}
    write('live_checks.json',result)
    return result


def unchanged():
    before=json.loads((EVIDENCE/'protected_before.json').read_text())
    changes=[name for name,digest in before.items() if fingerprint(ROOT/name)!=digest]
    result={'passed':not changes,'protected_files_checked':len(before),'changed':changes,
            'scope':'Final source notebooks, active models, API, frontend and automation preserved during the checks.'}
    write('unchanged.json',result)
    assert not changes,changes
    return result
