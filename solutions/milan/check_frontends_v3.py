"""Execute Node's real frontend tests and retain their receipt."""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.dont_write_bytecode=True
from mushroom_alternative import HERE, write_json


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node',default=shutil.which('node'))
    args=parser.parse_args()
    if not args.node:
        parser.error('Install Node.js or pass --node with its executable path.')
    completed=subprocess.run([args.node,'--test','--test-reporter=tap',str(HERE/'test_v3_frontends.cjs')],
        cwd=HERE,capture_output=True,text=True)
    output=completed.stdout+completed.stderr
    print(output)
    match=re.search(r'^# tests (\d+)$',output,re.M)
    write_json(HERE/'results_v3/frontend_test_results.json',{
        'finished_at_utc':datetime.now(timezone.utc).isoformat(),'success':completed.returncode==0,
        'tests_run':int(match[1]) if match else None,'exit_code':completed.returncode,
        'test_sha256_lf':hashlib.sha256((HERE/'test_v3_frontends.cjs').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        'scope':'Actual generated v2/v3 frontend decision functions, boundaries and form metadata; not a browser/server run.',
        'output':output})
    raise SystemExit(completed.returncode)
