"""Bewaar echte unittest-uitvoer als reproduceerbaar notebookrapport."""
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from citibike.config import REPORTS,atomic_json,code_version,now


def names(suite):
    for item in suite:
        if isinstance(item,unittest.TestSuite):
            yield from names(item)
        else:
            yield item.id()


if __name__=='__main__':
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    test_names=list(names(suite))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    atomic_json(REPORTS/'test_results.json',{
        'executed_at':now(),'command':'.venv/Scripts/python.exe -X utf8 tools/run_tests.py',
        'tests':result.testsRun,'passed':result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        'failed':len(result.failures)+len(result.errors),'skipped':len(result.skipped),
        'test_names':test_names,'code_version':code_version()})
    sys.exit(0 if result.wasSuccessful() else 1)
