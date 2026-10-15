"""Werk uitsluitend de gecontroleerde Citi Bike-bijdrage bij; raak junctions niet aan."""
from pathlib import Path
import argparse
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from citibike.config import now,sha256_file
from citibike.notebooks import NAMES


def sync(destination):
    destination=Path(destination).absolute()
    if destination.name!='Citybike' or destination.parent.name!='SolutionRaoul':
        raise ValueError('Doel moet de bestaande SolutionRaoul/Citybike-bijdrage zijn')
    if not destination.is_dir():raise FileNotFoundError(destination)
    # Houd nieuwe en gewijzigde bronbestanden expliciet; overige teamuitwerkingen blijven behouden.
    paths=['README.md','DEPLOYMENT.md','AUDIT.md','OPDRACHTCONTROLE.md','run.py','.gitignore',
           'citibike/notebooks.py','citibike/notebook_builder.py','citibike/notebook_sections.json',
           'citibike/model_data.py','citibike/reproduction.py','citibike/modeling.py',
           'tools/local_training.py','tools/run_workflow.py','tools/check_delivery.py','tools/export_contribution.py',
           'tools/check_preparation_independent.py','tools/retrain_pipeline.py',
           'tools/check_inference_parity.py','tools/sync_group_copy.py','tests/test_model_data.py',
           'tests/test_retrain_pipeline.py']
    paths += ['notebooks/'+name for name in NAMES]
    reports=['prepared_model_data.json','preparation_independence.json','improvement_protection.json',
             'test_results.json','data_manifest.json','delivery_check.json','inference_parity.json',
             'retraining_pipeline_latest.json','aws_evidence_review.json','implementation_checks.json',
             'notebook_content_migration.json']
    paths += ['reports/'+name for name in reports]
    paths += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'reports/notebook_reproduction').glob('*')) if p.is_file()]
    archive=ROOT/'.local_archive/group_before_improvements_20261014'
    records=[]
    for relative in paths:
        source=ROOT/relative;target=destination/relative
        if not source.is_file():raise FileNotFoundError(source)
        if target.exists() and sha256_file(target)==sha256_file(source):
            changed=False
        else:
            if target.exists():
                backup=archive/relative;backup.parent.mkdir(parents=True,exist_ok=True)
                if not backup.exists():shutil.copy2(target,backup)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target);changed=True
        records.append({'path':relative,'source_sha256':sha256_file(source),
                        'export_sha256':sha256_file(target),'source_bytes':source.stat().st_size,
                        'export_bytes':target.stat().st_size,'documentation_changed':False,'updated':changed})
    obsolete=['01_EDA_en_hypothesen.ipynb','citybike_analysis.ipynb']
    for name in obsolete:
        target=destination/'notebooks'/name
        # Exacte bestandsnaam binnen het geverifieerde notebookdoel, geen directorydelete.
        if target.is_file():
            backup=archive/'notebooks'/name;backup.parent.mkdir(parents=True,exist_ok=True)
            if not backup.exists():shutil.copy2(target,backup)
            target.unlink()
    old=destination/'EXPORT_MANIFEST.json'
    previous=json.loads(old.read_text(encoding='utf-8')) if old.exists() else {}
    indexed={r['path']:r for r in previous.get('files',[])}
    indexed.update({r['path']:r for r in records})
    for name in obsolete:indexed.pop('notebooks/'+name,None)
    receipt={**previous,'exported_at':now(),'source':str(ROOT),'destination':str(destination),
             'source_of_truth':'Citybike_full','executed_notebook_outputs_preserved':True,
             'controlled_update':True,'junctions_untouched':['data','models','.venv'],
             'removed_obsolete_notebooks':obsolete,'files':list(indexed.values())}
    old.write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding='utf-8')
    for record in records:
        assert record['source_sha256']==record['export_sha256']
    print(f'{len(records)} gecontroleerde bestanden; {sum(r["updated"] for r in records)} bijgewerkt.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,default=Path(r'C:\Deeplearning\DeepLearningTeam10\SolutionRaoul\Citybike'))
    sync(parser.parse_args().destination)
