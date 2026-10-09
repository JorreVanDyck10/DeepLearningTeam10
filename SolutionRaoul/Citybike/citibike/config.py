"""Vastgelegde onderzoeksafspraken; geen keuzes op basis van eindtestuitkomsten."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import importlib.metadata
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("CITIBIKE_SOURCE", ROOT.parents[1] / "citibike_data"))
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
TRAIN_END = "2026-05-01"
TEST_START = "2026-07-01"
TEST_END = "2026-09-01"
SCHEMA_VERSION = "1.0"
SEED = 42
TIMEZONE = "America/New_York"
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "duckdb", "pyarrow",
            "pycaret", "lightgbm", "joblib", "streamlit", "matplotlib", "psutil"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def code_version(paths=None) -> dict:
    files = paths or sorted((ROOT / "citibike").glob("*.py"))
    hashes = {str(Path(p).relative_to(ROOT)): sha256_file(p) for p in files}
    def git(*args):
        run = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return run.stdout.strip() if run.returncode == 0 else None
    return {"fingerprint": fingerprint(hashes), "files": hashes,
            "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
            "packages": {p: importlib.metadata.version(p) for p in PACKAGES}}


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    # Windows-readers/antivirus kunnen een zeer korte delete-sharing-lock houden.
    for attempt in range(15):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt==14:
                raise
            time.sleep(min(.05*2**attempt,.5))


def connect(path=":memory:"):
    import duckdb
    connection = duckdb.connect(str(path))
    connection.execute("SET memory_limit='4GiB'")
    connection.execute("SET threads=4")
    connection.execute("SET preserve_insertion_order=false")
    temporary = DATA / "temporary"
    temporary.mkdir(parents=True, exist_ok=True)
    connection.execute("SET temp_directory=?", [str(temporary)])
    return connection


def sql_path(path: Path) -> str:
    return "'" + str(path).replace("\\", "/").replace("'", "''") + "'"
