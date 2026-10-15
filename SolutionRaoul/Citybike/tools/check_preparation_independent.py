"""Voer het echte notebook 02 uit in een project zonder model- of onderzoeksrapporten."""
from pathlib import Path
import csv
import io
import os
import shutil
import sys
import tempfile
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from citibike.config import atomic_json, now, sha256_file


def main():
    notebook_path=ROOT/'notebooks/02_definitieve_voorbereiding.ipynb'
    with tempfile.TemporaryDirectory(prefix='citibike-independent-') as temporary:
        isolated=Path(temporary)/'project';isolated.mkdir()
        shutil.copytree(ROOT/'citibike',isolated/'citibike',ignore=shutil.ignore_patterns('__pycache__'))
        source=Path(temporary)/'sources';source.mkdir()
        for area in ['NYC','JC']:
            for ym in ['202511','202512','202601','202602','202603','202604','202605','202606']:
                prefix='' if area=='NYC' else 'JC-'
                name=prefix+ym+'-citibike-tripdata'
                content=io.StringIO();writer=csv.writer(content)
                writer.writerow(['ride_id','rideable_type','started_at','ended_at','start_station_id','start_station_name','end_station_id','end_station_name','start_lat','start_lng','end_lat','end_lng','member_casual'])
                day=ym[:4]+'-'+ym[4:]+'-15'
                writer.writerow([area+ym,'classic_bike',day+' 08:00:00',day+' 08:10:00','001','Start','002','End',40.7,-74.0,40.71,-74.01,'member'])
                with ZipFile(source/(name+'.zip'),'w') as archive: archive.writestr(name+'.csv',content.getvalue())
        assert not (isolated/'reports').exists() and not (isolated/'models').exists()
        notebook=nbformat.read(notebook_path,as_version=4)
        for cell in notebook.cells:
            if cell.cell_type=='code': cell.outputs=[];cell.execution_count=None
        manager=KernelManager(kernel_name='python3')
        manager.kernel_spec.argv=[sys.executable,'-X','utf8','-m','ipykernel_launcher','-f','{connection_file}']
        client=NotebookClient(notebook,timeout=300,km=manager,resources={'metadata':{'path':str(isolated)}})
        try: client.execute(env={**os.environ,'CITIBIKE_SOURCE':str(source)})
        finally:
            if manager.has_kernel:manager.shutdown_kernel(now=True)
        forbidden=['model_selection.json','target_decision.json','chronological_splits.json','evaluation_lock.json','final_evaluation.json','model_comparison.json','training_windows.json']
        assert not any((isolated/'reports'/name).exists() for name in forbidden)
        assert not (isolated/'models').exists()
        assert all(c.execution_count is not None for c in notebook.cells if c.cell_type=='code')
        assert not any(o.output_type=='error' or 'image/png' in o.get('data',{}) for c in notebook.cells if c.cell_type=='code' for o in c.outputs)
        import json
        prepared=json.loads((isolated/'reports/prepared_model_data.json').read_text(encoding='utf-8'))
        result={'checked_at':now(),'notebook_sha256':sha256_file(notebook_path),
                'actual_notebook_executed':True,'initial_reports_empty':True,'models_absent':True,
                'model_or_research_reports_required':False,'figures':0,'test_labels_read':False,
                'source':'16 synthetic ZIPs, NYC and JC, November 2025 through June 2026',
                'train_rows':prepared['train_rows'],'validation_rows':prepared['validation_rows'],
                'limitation':'Synthetic fixture proves dependencies; full real-data notebook execution separately verifies exact existing dataset equality.'}
        atomic_json(ROOT/'reports/preparation_independence.json',result)
        print(result)


if __name__=='__main__':main()
