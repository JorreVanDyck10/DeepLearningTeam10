"""Streaming ZIP → Arrow → DuckDB → Parquet, met bronoverlapcontrole.

Slechts één genest maandarchief wordt op schijf gezet. Elke maand is een
atomische cache-eenheid. Geen volledige CSV/ZIP wordt in geheugen geladen.
"""
from collections import defaultdict
from contextlib import contextmanager
import csv
import io
import json
import re
import shutil
import inspect
import time
from pathlib import Path
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.csv as pacsv
import psutil
import requests

from .config import (ROOT, SOURCE, DATA, PROCESSED, REPORTS, SCHEMA_VERSION,
                     atomic_json, code_version, connect, fingerprint, now,
                     sha256_file, sql_path)

ALIASES = {
    "tripduration": "reported_duration_seconds", "trip_duration": "reported_duration_seconds",
    "starttime": "started_at", "start_time": "started_at",
    "stoptime": "ended_at", "stop_time": "ended_at",
    "start_station_latitude": "start_lat", "start_station_longitude": "start_lng",
    "end_station_latitude": "end_lat", "end_station_longitude": "end_lng",
    "bikeid": "bike_id", "usertype": "member_casual", "user_type": "member_casual",
}
COLUMNS = ["ride_id", "rideable_type", "started_at", "ended_at", "start_station_id",
           "start_station_name", "end_station_id", "end_station_name", "start_lat",
           "start_lng", "end_lat", "end_lng", "member_casual", "bike_id",
           "reported_duration_seconds"]
TIME_FORMATS = "['%Y-%m-%d %H:%M:%S.%f','%Y-%m-%d %H:%M:%S','%Y-%m-%d %H:%M','%m/%d/%Y %H:%M:%S','%m/%d/%Y %H:%M','%m/%d/%Y %H:%M:%S.%f']"


def canonical_name(name: str) -> str:
    normalized = re.sub(r"\s+", "_", name.strip().lower())
    return ALIASES.get(normalized, normalized)


def useful_member(name: str) -> bool:
    return not name.startswith("__MACOSX/") and not Path(name).name.startswith(("._", "."))


def source_area(name: str) -> str:
    if re.match(r"^JC[- ]", name):
        return "JC"
    if re.match(r"^20\d{2}(?:\d{2})?-citibike-tripdata\.zip$", name):
        return "NYC"
    raise ValueError(f"Onbekend brongebied: {name}")


def member_period(name: str, fallback: str | None = None) -> str:
    matches = re.findall(r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])(?!\d)", name)
    if not matches:
        if fallback:
            return fallback
        raise ValueError(f"Geen betrouwbare maand in bronpad: {name}")
    periods = {f"{year}-{month}" for year, month in matches}
    if len(periods) != 1:
        raise ValueError(f"Tegenstrijdige perioden: {name}")
    return periods.pop()


def logical_groups(members, fallback=None):
    groups = defaultdict(list)
    for name in members:
        period = member_period(name, fallback)
        # Parallelle representaties (root-CSV tegenover submap met CSV-delen).
        parent = str(Path(name).parent)
        groups[(period, parent)].append(name)
    result={}
    for (period,parent),names in groups.items():
        split=[n for n in names if re.search(r'(?:_\d+|-part\d+)\.csv$',n,re.IGNORECASE)]
        whole=[n for n in names if n not in split]
        if split and whole:
            result[(period,parent+'::parts')]=split
            result[(period,parent+'::whole')]=whole
        else:
            result[(period,parent)]=names
    return result


@contextmanager
def open_container(path: Path, nested: str | None):
    if nested is None:
        with ZipFile(path) as archive:
            yield archive
    else:
        temporary = DATA / "temporary" / "nested_month.zip"
        temporary.parent.mkdir(parents=True, exist_ok=True)
        try:
            with ZipFile(path) as outer, outer.open(nested) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            with ZipFile(temporary) as archive:
                yield archive
        finally:
            temporary.unlink(missing_ok=True)


def discover(source=SOURCE):
    source = Path(source)
    paths = sorted(source.rglob("*.zip"))
    if not paths:
        raise FileNotFoundError(f"Geen ZIP-bronnen in {source}")
    start = time.perf_counter()
    records, tasks, schemas = [], defaultdict(list), {}
    for index, path in enumerate(paths, 1):
        area = source_area(path.name)
        record = {"path": str(path), "area": area, "size_bytes": path.stat().st_size,
                  "sha256": sha256_file(path), "members": [], "groups": [], "schemas": {}}
        discovery_receipt = DATA / 'discovery' / (path.name+'.json')
        if discovery_receipt.exists():
            cached=json.loads(discovery_receipt.read_text(encoding='utf-8'))
            if cached['sha256']==record['sha256'] and cached['path']==str(path):
                cached['groups']=[]
                for nested in sorted(set(m['nested'] for m in cached['members']),key=lambda x:x or ''):
                    members=[m['name'] for m in cached['members'] if m['nested']==nested]
                    fallback = member_period(nested or path.name, None) if (nested or len(path.name.split('-')[0])!=4) else None
                    for (period,parent),names in logical_groups(members,fallback).items():
                        descriptor={'source':str(path),'source_sha256':cached['sha256'],'nested':nested,
                                    'members':sorted(names),'parent':parent}
                        tasks[(area,period)].append(descriptor)
                        cached['groups'].append({'period':period,'descriptor':descriptor})
                schemas.update(cached['schemas']);records.append(cached)
                print(f'Inventariscache {index}/{len(paths)}: {path.name}',flush=True)
                continue
        with ZipFile(path) as outer:
            nested_names = [i.filename for i in outer.infolist()
                            if useful_member(i.filename) and i.filename.lower().endswith(".zip")]
        for nested in ([None] if not nested_names else sorted(nested_names)):
            with open_container(path, nested) as archive:
                members = [i.filename for i in archive.infolist()
                           if useful_member(i.filename) and i.filename.lower().endswith(".csv")]
                if not members:
                    raise ValueError(f"Geen CSV in {path.name}::{nested}")
                fallback = member_period(nested or path.name, None) if (nested or len(path.name.split('-')[0])!=4) else None
                for (period, parent), group in logical_groups(members,fallback).items():
                    descriptor = {"source": str(path), "source_sha256": record["sha256"],
                                  "nested": nested, "members": sorted(group), "parent": parent}
                    tasks[(area, period)].append(descriptor)
                    record['groups'].append({'period':period,'descriptor':descriptor})
                for member in members:
                    with archive.open(member) as stream:
                        header = next(csv.reader([stream.readline().decode("utf-8-sig")]))
                    mapped = [canonical_name(c) for c in header]
                    if len(set(mapped)) != len(mapped) or not {"started_at", "ended_at"}.issubset(mapped):
                        raise ValueError(f"Onbekend of dubbel schema: {path.name}::{member}: {header}")
                    schemas[fingerprint(header)] = {"raw": header, "canonical": mapped}
                    record['schemas'][fingerprint(header)] = schemas[fingerprint(header)]
                    info = archive.getinfo(member)
                    record["members"].append({"nested": nested, "name": member, "period": member_period(member,fallback),
                                              "uncompressed_bytes": info.file_size, "crc": info.CRC,
                                              "schema": fingerprint(header)})
        records.append(record)
        atomic_json(discovery_receipt,record)
        print(f"Inventaris {index}/{len(paths)}: {path.name}", flush=True)
    inventory = {"source": str(source), "source_count": len(records),
                 "source_bytes": sum(r["size_bytes"] for r in records), "sources": records,
                 "schemas": schemas, "discovery_seconds": time.perf_counter() - start,
                 "source_fingerprint": fingerprint([(r["path"], r["sha256"]) for r in records])}
    atomic_json(REPORTS / "source_inventory.json", inventory)
    return inventory, tasks


def batches(archive, member):
    with archive.open(member) as stream:
        reader = pacsv.open_csv(stream, read_options=pacsv.ReadOptions(block_size=8 * 1024 * 1024),
                                convert_options=pacsv.ConvertOptions(default_column_type=pa.string(),
                                                                     strings_can_be_null=True,
                                                                     null_values=["", "NULL", "null", "\\N"]))
        names = [canonical_name(name) for name in reader.schema.names]
        for batch in reader:
            table = pa.Table.from_batches([batch]).rename_columns(names)
            arrays = [table[c] if c in table.column_names else pa.nulls(table.num_rows, type=pa.string())
                      for c in COLUMNS]
            yield pa.Table.from_arrays(arrays, names=COLUMNS)


def transform_sql(area, period):
    year, month = map(int, period.split("-"))
    return f"""
    WITH parsed AS (
      SELECT nullif(trim(ride_id),'') AS ride_id,
        coalesce(nullif(lower(trim(rideable_type)),''),'unknown') AS rideable_type,
        try_strptime(trim(started_at), {TIME_FORMATS}) AS started_at,
        try_strptime(trim(ended_at), {TIME_FORMATS}) AS ended_at,
        CASE WHEN bike_id IS NOT NULL THEN nullif(regexp_replace(trim(start_station_id),'\\.0+$',''),'')
          ELSE nullif(trim(start_station_id),'') END AS start_station_id,
        nullif(trim(start_station_name),'') AS start_station_name,
        CASE WHEN bike_id IS NOT NULL THEN nullif(regexp_replace(trim(end_station_id),'\\.0+$',''),'')
          ELSE nullif(trim(end_station_id),'') END AS end_station_id,
        nullif(trim(end_station_name),'') AS end_station_name,
        round(try_cast(start_lat AS DOUBLE),7) AS start_lat, round(try_cast(start_lng AS DOUBLE),7) AS start_lng,
        round(try_cast(end_lat AS DOUBLE),7) AS end_lat, round(try_cast(end_lng AS DOUBLE),7) AS end_lng,
        CASE lower(trim(member_casual)) WHEN 'subscriber' THEN 'member' WHEN 'customer' THEN 'casual'
          WHEN 'member' THEN 'member' WHEN 'casual' THEN 'casual' ELSE 'unknown' END AS member_casual,
        nullif(trim(bike_id),'') AS bike_id,
        try_cast(reported_duration_seconds AS DOUBLE) AS reported_duration_seconds,
        started_at IS NULL OR trim(started_at)='' AS missing_started_at,
        ended_at IS NULL OR trim(ended_at)='' AS missing_ended_at
      FROM batch
    ), derived AS (
      SELECT *, epoch(ended_at-started_at)/60.0 AS duration_minutes,
        '{area}'::VARCHAR AS area,
        coalesce(year(started_at)={year} AND month(started_at)={month},false) AS valid_demand,
        coalesce(start_lat BETWEEN -90 AND 90 AND start_lng BETWEEN -180 AND 180,false) AS valid_start_coordinates,
        coalesce(end_lat BETWEEN -90 AND 90 AND end_lng BETWEEN -180 AND 180,false) AS valid_end_coordinates
      FROM parsed
    )
    SELECT *, coalesce(valid_demand AND duration_minutes>0,false) AS valid_duration,
      coalesce(duration_minutes>1440,false) AS long_duration_flag
    FROM derived
    """


def rows_signature(connection, table):
    # HASH is used only as a fast diagnostic. Exact multiset comparison decides equivalence.
    return connection.execute(f"SELECT count(*), min(started_at), max(started_at) FROM {table}").fetchone()


def equivalent(connection, left, right):
    columns = ','.join(COLUMNS + ['duration_minutes', 'area'])
    mismatch = connection.execute(f"""SELECT EXISTS(
      (SELECT {columns} FROM {left} EXCEPT ALL SELECT {columns} FROM {right})
      UNION ALL
      (SELECT {columns} FROM {right} EXCEPT ALL SELECT {columns} FROM {left})
    )""").fetchone()[0]
    return not mismatch


def quality(connection, table):
    expressions = {
        "rows": "count(*)", "valid_demand": "count(*) FILTER (WHERE valid_demand)",
        "invalid_start": "count(*) FILTER (WHERE started_at IS NULL)",
        "invalid_end": "count(*) FILTER (WHERE ended_at IS NULL)",
        "outside_source_month": "count(*) FILTER (WHERE started_at IS NOT NULL AND NOT valid_demand)",
        "missing_start_station": "count(*) FILTER (WHERE start_station_id IS NULL)",
        "missing_end_station": "count(*) FILTER (WHERE end_station_id IS NULL)",
        "bad_start_coordinates": "count(*) FILTER (WHERE NOT valid_start_coordinates)",
        "bad_end_coordinates": "count(*) FILTER (WHERE NOT valid_end_coordinates)",
        "nonpositive_duration": "count(*) FILTER (WHERE duration_minutes<=0)",
        "long_duration": "count(*) FILTER (WHERE long_duration_flag)",
        "valid_duration": "count(*) FILTER (WHERE valid_duration)",
        "unknown_member": "count(*) FILTER (WHERE member_casual='unknown')",
        "unknown_bike": "count(*) FILTER (WHERE rideable_type='unknown')",
        "missing_ride_id": "count(*) FILTER (WHERE ride_id IS NULL)",
        "missing_started_at": "count(*) FILTER (WHERE missing_started_at)",
        "missing_ended_at": "count(*) FILTER (WHERE missing_ended_at)",
        "unparseable_start": "count(*) FILTER (WHERE started_at IS NULL AND NOT missing_started_at)",
        "unparseable_end": "count(*) FILTER (WHERE ended_at IS NULL AND NOT missing_ended_at)",
        "reported_duration_mismatch": "count(*) FILTER (WHERE abs(reported_duration_seconds-duration_minutes*60)>1)",
        "negative_duration": "count(*) FILTER (WHERE duration_minutes<0)",
        "unknown_bike_category": "count(*) FILTER (WHERE rideable_type NOT IN ('unknown','classic_bike','electric_bike','docked_bike'))",
    }
    result = connection.execute(f"SELECT {','.join(expressions.values())} FROM {table}").fetchone()
    return dict(zip(expressions, result))


def prepare_partition(area, period, descriptors, signature):
    destination = PROCESSED / area / f"year={period[:4]}" / f"month={period[5:]}" / "rides.parquet"
    receipt = DATA / "receipts" / area / f"{period}.json"
    if receipt.exists() and destination.exists():
        cached = json.loads(receipt.read_text(encoding="utf-8"))
        if cached["signature"] == signature and cached["sha256"] == sha256_file(destination):
            print(f"Cache {area} {period}", flush=True)
            return cached
    start = time.perf_counter()
    staging = DATA / "temporary" / "staging.duckdb"
    staging.unlink(missing_ok=True)
    connection = connect(staging)
    raw_rows, variants, peak_rss = 0, [], 0
    try:
        for index, descriptor in enumerate(descriptors):
            table_name = f"variant_{index}"
            created = False
            with open_container(Path(descriptor["source"]), descriptor["nested"]) as archive:
                for member in descriptor["members"]:
                    for batch in batches(archive, member):
                        raw_rows += batch.num_rows
                        connection.register("batch", batch)
                        query = transform_sql(area, period)
                        if not created:
                            connection.execute(f"CREATE TABLE {table_name} AS {query}")
                            created = True
                        else:
                            connection.execute(f"INSERT INTO {table_name} {query}")
                        connection.unregister("batch")
                        peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
            if not created:
                raise ValueError(f"Lege bronrepresentatie {area} {period}")
            variants.append({"table": table_name, "source": descriptor, "rows": rows_signature(connection, table_name)[0]})
        chosen = variants[0]["table"]
        overlap = []
        for candidate in variants[1:]:
            if not equivalent(connection, chosen, candidate["table"]):
                raise ValueError(f"Bronrepresentaties zijn niet equivalent: {area} {period}. Onderzoek verplicht.")
            overlap.append({"discarded_representation": candidate["source"], "rows": candidate["rows"],
                            "proof": "exact canonical multiset equality (EXCEPT ALL both directions)"})
        before = quality(connection, chosen)
        # IDs: alleen exacte herhalingen worden verwijderd; conflicterende inhoud stopt verwerking.
        compare_columns = [c for c in COLUMNS if c != 'ride_id']
        grouped = ','.join(compare_columns)
        conflicts = connection.execute(f"""SELECT count(*) FROM (
          SELECT ride_id FROM (SELECT DISTINCT ride_id,{grouped} FROM {chosen} WHERE ride_id IS NOT NULL)
          GROUP BY ride_id HAVING count(*)>1)""").fetchone()[0]
        if conflicts:
            raise ValueError(f"{conflicts} conflicterende ride-ID's in {area} {period}")
        connection.execute(f"""CREATE TABLE clean AS
          SELECT * FROM {chosen} WHERE ride_id IS NULL
          UNION ALL SELECT * EXCLUDE(rn) FROM (
            SELECT *,row_number() OVER(PARTITION BY ride_id ORDER BY started_at) rn
            FROM {chosen} WHERE ride_id IS NOT NULL) WHERE rn=1""")
        after = quality(connection, "clean")
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(".parquet.part")
        connection.execute(f"COPY clean TO {sql_path(partial)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)")
        partial.replace(destination)
        schema = connection.execute("DESCRIBE clean").fetchall()
        bounds = connection.execute("SELECT min(started_at),max(started_at) FROM clean WHERE valid_demand").fetchone()
        result = {"area": area, "period": period, "path": str(destination), "signature": signature,
                  "sha256": sha256_file(destination), "size_bytes": destination.stat().st_size,
                  "schema": schema, "raw_rows": raw_rows, "retained_rows": after["rows"],
                  "removed_rows": raw_rows-after["rows"], "overlap_removed": sum(v["rows"] for v in overlap),
                  "duplicate_ids_removed": before["rows"]-after["rows"],
                  "removed_percent": 100*(raw_rows-after["rows"])/raw_rows,
                  "quality_before": before, "quality_after": after, "overlap_proofs": overlap,
                  "minimum_timestamp": bounds[0], "maximum_timestamp": bounds[1],
                  "processing_seconds": time.perf_counter()-start, "peak_rss_bytes_sampled": peak_rss,
                  "test_protection": "technical_only" if period>='2026-07' else "development_or_validation"}
        atomic_json(receipt, result)
        print(f"Gereed {area} {period}: {raw_rows:,} -> {after['rows']:,}; {result['processing_seconds']:.1f}s", flush=True)
        return result
    finally:
        connection.close()
        staging.unlink(missing_ok=True)
        staging.with_suffix('.duckdb.wal').unlink(missing_ok=True)


def prepare(source=SOURCE):
    start, started = time.perf_counter(), now()
    inventory, tasks = discover(source)
    pipeline_code = code_version([Path(__file__), ROOT / "citibike" / "config.py", ROOT / "citibike" / "assembly.py"])
    semantics=processing_fingerprint()
    partitions = []
    for (area, period), descriptors in sorted(tasks.items()):
        signature = fingerprint({"sources": descriptors, "code": semantics, "schema": SCHEMA_VERSION})
        partitions.append(prepare_partition(area, period, descriptors, signature))
        atomic_json(REPORTS / "processing_progress.json", {"updated_at": now(), "partitions": partitions})
    from .assembly import assemble
    partitions,boundary_audit=assemble(partitions)
    manifest = {**inventory, "schema_version": SCHEMA_VERSION, "code_version": pipeline_code,
                "schema_mapping": ALIASES, "normalization": {
                    "legacy_station_ids": "remove integer .0 suffix only where legacy bike_id field exists; leading zeros preserved",
                    "coordinate_precision": "7 decimal places (~centimeter); equivalent source versions differ at floating-point serialization precision",
                    "overlap_decision": "exact multiset equality after documented normalization; multiplicities preserved"},
                "configuration": {"duckdb_memory": "4GiB", "threads": 4,
                "arrow_block_bytes": 8*1024*1024, "train_end": "2026-05-01", "test_start": "2026-07-01"},
                "started_at": started, "finished_at": now(), "processing_seconds": time.perf_counter()-start,
                "raw_rows": sum(p["raw_rows"] for p in partitions),
                "retained_rows": sum(p["retained_rows"] for p in partitions),
                "removed_rows": sum(p["removed_rows"] for p in partitions), "partitions": partitions,
                "month_boundary_audit": boundary_audit,
                "dataset_fingerprint": fingerprint([(p['area'],p['period'],p['sha256'],SCHEMA_VERSION) for p in partitions])}
    import pandas as pd
    manifest['coverage'] = {}
    for area in ('NYC','JC'):
        present = sorted(p['period'] for p in partitions if p['area']==area)
        expected = [str(p) for p in pd.period_range(present[0],present[-1],freq='M')]
        manifest['coverage'][area] = {'first_month':present[0], 'last_month':present[-1],
                                      'missing_months':sorted(set(expected)-set(present))}
    atomic_json(REPORTS / "data_manifest.json", manifest)
    return manifest


def processing_fingerprint():
    """Cachekey van relevante verwerking; volledige codeversie staat apart in manifest."""
    functions=[canonical_name,batches,transform_sql,equivalent,quality,prepare_partition]
    return fingerprint({'functions':{f.__name__:inspect.getsource(f) for f in functions},
                        'aliases':ALIASES,'columns':COLUMNS,'time_formats':TIME_FORMATS,
                        'configuration':inspect.getsource(connect),'schema':SCHEMA_VERSION})


def download(source=SOURCE):
    """Herstel exact de geïnventariseerde bronnen; controleer elke download met SHA-256."""
    manifest = json.loads((REPORTS / "source_inventory.json").read_text(encoding="utf-8"))
    source = Path(source)
    source.mkdir(parents=True, exist_ok=True)
    for record in manifest['sources']:
        path = source / Path(record['path']).name
        if path.exists() and sha256_file(path)==record['sha256']:
            continue
        url = f"https://s3.amazonaws.com/tripdata/{path.name}"
        partial = path.with_suffix('.zip.part')
        with requests.get(url, stream=True, timeout=(30,120)) as response:
            response.raise_for_status()
            with partial.open('wb') as target:
                for block in response.iter_content(8*1024*1024):
                    target.write(block)
        if sha256_file(partial)!=record['sha256']:
            raise ValueError(f"Bron gewijzigd: {url}; herinventarisatie nodig")
        partial.replace(path)
