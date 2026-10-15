"""Re-run study 3 from a clean output folder without changing published evidence.

The normal improve_v3 entry point resumes an existing local run and uses its
local checkpoints. This entry point also works in a fresh Git clone where those
ignored checkpoints do not exist. Pass a new --name for each independent rerun.
"""
import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode=True
import joblib
import improve_v3


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',default='reproduction_v3')
    args=parser.parse_args()
    if Path(args.name).name!=args.name or args.name in ['.','..']:
        raise ValueError('Use a simple folder name')
    improve_v3.OUT=Path(__file__).resolve().parent/'.generated'/args.name
    if improve_v3.OUT.exists():
        raise FileExistsError('Choose a new --name to keep previous evidence intact.')
    with joblib.parallel_config(backend='threading'):
        improve_v3.run()
