"""Run the concept's actual tests and preserve a machine-readable receipt."""
from datetime import datetime, timezone
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main(learning_only=False):
    study = json.loads((HERE/'results_v2/run.json').read_text())
    assert study['complete'], 'Complete all declared model families before final verification.'
    stream = io.StringIO()
    suite = (unittest.defaultTestLoader.loadTestsFromName('test_improvements.LearningChecks') if learning_only else
        unittest.defaultTestLoader.discover(str(HERE), pattern='test_improvements.py'))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    output = stream.getvalue()
    print(output)
    receipt = {'finished_at_utc': datetime.now(timezone.utc).isoformat(),
        'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'skipped': len(result.skipped), 'success': result.wasSuccessful(),
        'scope': 'Milan ML checks and local Mushroom controller/form metadata. No full production/Citi Bike integration claim.',
        'artifact_hash_mode': 'LF-normalized SHA256 for JSON/CSV; raw SHA256 for binary models',
        'artifacts': {p.name: hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n') if p.suffix in ('.json', '.csv') else p.read_bytes()).hexdigest()
            for p in sorted((HERE/'artifacts').iterdir()) if p.is_file()},
        'output': output}
    filename = 'test_results_learning.json' if learning_only else 'test_results.json'
    (HERE/'results_v2'/filename).write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--learning-only', action='store_true', help='Run the ML subset when HTTP test threads are unavailable.')
    args = parser.parse_args()
    raise SystemExit(main(args.learning_only))
