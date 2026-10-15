"""Run real third-study tests and save their output and evidence fingerprints."""
from datetime import datetime, timezone
import hashlib
import argparse
import io
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
from mushroom_alternative import write_json


def main(packages=False):
    stream = io.StringIO()
    if packages:
        suite=unittest.TestSuite([
            unittest.defaultTestLoader.discover(str(HERE),pattern='test_v3_packages.py'),
            unittest.defaultTestLoader.discover(str(HERE),pattern='test_improvements.py')])
    else:
        suite = unittest.defaultTestLoader.discover(str(HERE),pattern='test_v3.py')
    result = unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    output = stream.getvalue()
    print(output)
    files = [HERE/'improve_v3.py',HERE/'test_v3.py',*sorted((HERE/'results_v3').glob('*.csv')),
             HERE/'results_v3/protocol.json',HERE/'results_v3/selections.json',HERE/'results_v3/run.json']
    if packages:
        files.extend(HERE/n for n in ['preview.py','preview_v3.py','preview_voting_v3.py',
            'voting_model_v3.py','test_v3_packages.py','results_v3/candidate_package.json','results_v3/voting_package.json'])
        files.extend(p for folder in ['artifacts','artifacts_v3','voting_artifacts_v3']
            for p in (HERE/folder).rglob('*') if p.is_file())
    filename='package_test_results.json' if packages else 'test_results.json'
    write_json(HERE/'results_v3'/filename, {
        'finished_at_utc':datetime.now(timezone.utc).isoformat(),
        'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'skipped':len(result.skipped),'success':result.wasSuccessful(),
        'scope':('RF/voting export and actual API/form contract, plus all seven v2 regression checks; no production/Citi Bike integration.' if packages else
            'Feature preparation, original row membership, recomputed metrics and voting equivalence. No production integration.'),
        'hash_mode':'LF-normalized SHA256 for text; raw binary hashes',
        'hashes':{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n') if p.suffix in ['.json','.csv','.py','.txt'] else p.read_bytes()).hexdigest() for p in files},
        'output':output})
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packages',action='store_true')
    raise SystemExit(main(parser.parse_args().packages))
