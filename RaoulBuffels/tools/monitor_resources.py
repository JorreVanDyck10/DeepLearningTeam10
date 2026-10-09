"""Meet RSS van deze project's CLI-processen, inclusief hash/tuning-stappen."""
from pathlib import Path
import json
import time
from datetime import datetime,timezone
import psutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from citibike.config import atomic_json
records={};last_seen=time.monotonic();started=datetime.now(timezone.utc).isoformat()
existing=ROOT/'reports'/'resource_profile.json'
if existing.exists():
    previous=json.loads(existing.read_text(encoding='utf-8'))
    records={f"{row['pid']}:{row['stage']}":row for row in previous['processes']}
    started=previous['started_at']
while time.monotonic()-last_seen<120:
    for process in psutil.process_iter(['pid','cmdline','cwd','name']):
        try:
            args=process.info['cmdline'] or []
            if process.info['cwd']!=str(ROOT) or not args or not any(Path(str(a)).name=='run.py' for a in args):continue
            if not process.info['name'].lower().startswith('python'):continue
            key=f"{process.pid}:{args[-1]}";memory=process.memory_info().rss
            row=records.setdefault(key,{'stage':args[-1],'pid':process.pid,'samples':0,'peak_rss_bytes':0})
            row['samples']+=1;row['peak_rss_bytes']=max(row['peak_rss_bytes'],memory)
            row['last_sample_at']=datetime.now(timezone.utc).isoformat();last_seen=time.monotonic()
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    atomic_json(ROOT/'reports'/'resource_profile.json',{
         'started_at':started,'processes':list(records.values()),
         'note':'1-second sampled RSS of CLI main processes, not combined worker memory. Preparation monitoring started after early partitions; restarts can leave gaps. Not an exact continuous peak.'})
    time.sleep(1)
