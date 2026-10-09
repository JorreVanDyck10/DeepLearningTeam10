"""Eenmalige, gecontroleerde NYC-cacheovergang na uitsluitend JC-precisiefix."""
from pathlib import Path
import sys
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from citibike.config import ROOT,DATA,REPORTS,sha256_file,fingerprint,atomic_json,now

OLD='8bae9a7f0402b0afea183b77b0c8f5867f7f9e5d557040d5eb585b0b70e45121'
NEW='f75971f7f9e60d0ba4a934cae1cab789ff1e681d401325eaa34ec07878f2e9a2'
before=DATA/'assembly_before_jc_precision.py';after=ROOT/'citibike'/'assembly.py'
assert sha256_file(before)==OLD and sha256_file(after)==NEW,'Andere codeovergang: geen migratie'
def writer(text):return text[text.index('            for part in current:'):text.index('            audit.append(area_audit)')]
assert writer(before.read_text(encoding='utf-8'))==writer(after.read_text(encoding='utf-8'))
parts=json.loads((REPORTS/'processing_progress.json').read_text(encoding='utf-8'))['partitions']
inputs=[(p['area'],p['period'],p['sha256']) for p in parts]
old_signature=fingerprint({'inputs':inputs,'code':OLD});new_signature=fingerprint({'inputs':inputs,'code':NEW})
lookup={(p['area'],p['period']):p for p in parts};changes=[]
for receipt in sorted((DATA/'canonical_receipts'/'NYC').glob('*.json')):
    saved=json.loads(receipt.read_text(encoding='utf-8'))
    if saved['assembly_signature']==new_signature:continue
    assert saved['assembly_signature']==old_signature
    original=lookup[('NYC',saved['period'])]
    assert saved['source_cache_sha256']==original['sha256']==sha256_file(Path(original['path']))
    assert saved['sha256']==sha256_file(Path(saved['path']))
    changes.append((receipt,saved))
audit={'started_at':now(),'old_code_sha256':OLD,'new_code_sha256':NEW,
       'old_signature':old_signature,'new_signature':new_signature,
       'proof':'Canonical writer identical byte-for-byte. NYC time_alias_ids is forced empty; added sort prefix is constant and retains existing order. Existing NYC station proof and source-removal policy unchanged. JC-only precision correction cannot affect NYC rows.',
       'partitions':[]}
for receipt,saved in changes:
    saved['assembly_signature']=new_signature
    saved['cache_transition']='reports/nyc_cache_transition.json'
    atomic_json(receipt,saved)
    audit['partitions'].append({'period':saved['period'],'unchanged_sha256':saved['sha256'],'source_cache_sha256':saved['source_cache_sha256']})
audit['finished_at']=now();atomic_json(REPORTS/'nyc_cache_transition.json',audit)
print(f'{len(changes)} NYC-caches: bron- en outputchecksum gecontroleerd; uitsluitend receiptversie aangepast.',flush=True)
