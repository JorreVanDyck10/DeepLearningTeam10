"""Voer de resterende lokale workflow uit nadat de bronpipeline klaar is."""
from pathlib import Path
import argparse
import subprocess
import sys
import time
import psutil

ROOT=Path(__file__).resolve().parents[1]


def preparing():
    for process in psutil.process_iter(['cmdline','name']):
        args=process.info['cmdline'] or []
        if process.pid==psutil.Process().pid:continue
        if args and args[-1]=='prepare' and any('run.py' in str(a) for a in args):
            return True
    return False


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait-for-prepare',action='store_true')
    args=parser.parse_args()
    if args.wait_for_prepare:
        while preparing():time.sleep(10)
    if not (ROOT/'reports'/'data_manifest.json').exists():
        raise RuntimeError('Voorbereiding niet voltooid; zie processing.log')
    subprocess.run([sys.executable,'-X','utf8',str(ROOT/'tools'/'run_tests.py')],cwd=ROOT,check=True)
    for stage,receipt in [('research','target_decision.json'),('train','model_artifact.json'),
                          ('evaluate','final_evaluation.json'),('verify',None),('notebooks',None)]:
        if receipt and (ROOT/'reports'/receipt).exists():
            print(f'Reeds voltooid: {stage}',flush=True);continue
        print(f'STAGE {stage}',flush=True)
        subprocess.run([sys.executable,'-X','utf8',str(ROOT/'run.py'),stage],cwd=ROOT,check=True)
