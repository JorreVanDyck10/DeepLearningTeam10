"""Technische maandgrenscorrectie op immutable maandcaches, vóór onderzoek."""
import json
import time
from pathlib import Path
from .config import DATA,REPORTS,connect,sql_path,sha256_file,fingerprint,atomic_json
from .prepare import COLUMNS,quality


def assemble(partitions):
    signature=fingerprint({'inputs':[(p['area'],p['period'],p['sha256']) for p in partitions],
                           'code':sha256_file(Path(__file__))})
    completion=DATA/'canonical_completion.json'
    if completion.exists():
        cached=json.loads(completion.read_text(encoding='utf-8'))
        if cached['signature']==signature and all(Path(p['path']).exists() and
                sha256_file(Path(p['path']))==p['sha256'] for p in cached['partitions']):
            return cached['partitions'],cached['audit']
    results=[];audit=[]
    columns=','.join(COLUMNS+['duration_minutes','area'])
    for area in ('NYC','JC'):
        current=[p for p in partitions if p['area']==area]
        if not current:continue
        con=connect()
        try:
            paths='['+','.join(sql_path(Path(p['path'])) for p in current)+']'
            con.execute(f'CREATE VIEW source_rows AS SELECT * FROM read_parquet({paths},hive_partitioning=false,filename=true)')
            assert con.execute('SELECT DISTINCT area FROM source_rows').fetchall()==[(area,)]
            con.execute('CREATE TEMP TABLE spill_rows AS SELECT * FROM source_rows WHERE started_at IS NOT NULL AND NOT valid_demand')
            con.execute('CREATE TEMP TABLE duplicate_ids AS SELECT ride_id FROM source_rows WHERE ride_id IS NOT NULL GROUP BY ride_id HAVING count(*)>1')
            con.execute('CREATE TEMP TABLE duplicate_rows AS SELECT * FROM source_rows WHERE ride_id IN (SELECT ride_id FROM duplicate_ids)')
            raw_aliases=con.execute(f'SELECT ride_id FROM (SELECT DISTINCT {columns} FROM duplicate_rows) GROUP BY ride_id HAVING count(*)>1').fetchall()
            if area=='JC':
                con.execute("""CREATE TEMP TABLE time_alias_ids AS SELECT ride_id FROM duplicate_rows GROUP BY ride_id
                    HAVING (count(DISTINCT started_at)>1 OR count(DISTINCT ended_at)>1)
                    AND count(DISTINCT date_trunc('second',started_at))=1
                    AND count(DISTINCT date_trunc('second',ended_at))=1
                    AND count(*) FILTER(WHERE started_at=date_trunc('second',started_at)
                                       AND ended_at=date_trunc('second',ended_at))>0""")
            else:
                con.execute('CREATE TEMP TABLE time_alias_ids AS SELECT ride_id FROM duplicate_rows WHERE false')
            time_aliases=[row[0] for row in con.execute('SELECT ride_id FROM time_alias_ids ORDER BY ride_id').fetchall()]
            # Alleen bron-equivalentie: ID's blijven in output tekst zoals aangeleverd.
            # Fractionele nulserialisatie is alleen toegestaan bij bekende gelijke naam/locatie.
            expressions=[]
            for endpoint in ('start','end'):
                field=endpoint+'_station_id'
                expressions.append(f"CASE WHEN regexp_full_match({field},'[0-9]+\\.[0-9]+') AND {endpoint}_station_name IS NOT NULL AND {endpoint}_lat IS NOT NULL AND {endpoint}_lng IS NOT NULL THEN rtrim(rtrim({field},'0'),'.') ELSE {field} END AS {field}")
            if area=='JC':
                for field in ('started_at','ended_at'):
                    expressions.append(f"CASE WHEN ride_id IN(SELECT ride_id FROM time_alias_ids) THEN date_trunc('second',{field}) ELSE {field} END AS {field}")
                expressions.append("CASE WHEN ride_id IN(SELECT ride_id FROM time_alias_ids) THEN epoch(date_trunc('second',ended_at)-date_trunc('second',started_at))/60.0 ELSE duration_minutes END AS duration_minutes")
            con.execute('CREATE VIEW proof_rows AS SELECT * REPLACE('+','.join(expressions)+') FROM duplicate_rows')
            conflict=con.execute(f'SELECT ride_id FROM (SELECT DISTINCT {columns} FROM proof_rows) GROUP BY ride_id HAVING count(*)>1 LIMIT 1').fetchone()
            if conflict:raise ValueError(f'Conflicterend ride-ID over bronmaanden ({area}): {conflict[0]}')
            con.execute('''CREATE TEMP TABLE modern_remove AS SELECT filename,ride_id FROM (
                SELECT filename,ride_id,row_number() OVER(PARTITION BY ride_id ORDER BY
                  CASE WHEN ride_id IN(SELECT ride_id FROM time_alias_ids)
                    AND (started_at!=date_trunc('second',started_at) OR ended_at!=date_trunc('second',ended_at)) THEN 0 ELSE 1 END,
                  valid_demand DESC,filename) AS rn
                FROM duplicate_rows) WHERE rn>1''')
            con.execute('CREATE TEMP TABLE legacy_remove(filename VARCHAR,period VARCHAR)')
            groups=con.execute("SELECT filename,strftime(started_at,'%Y-%m'),count(*) FROM spill_rows WHERE ride_id IS NULL GROUP BY 1,2 ORDER BY 1,2").fetchall()
            lookup={p['period']:p for p in current};legacy_proofs=[];unproven=[]
            for filename,period,count in groups:
                if period not in lookup:continue
                target=sql_path(Path(lookup[period]['path']))
                missing=con.execute(f'''SELECT count(*) FROM (
                    SELECT {columns} FROM spill_rows WHERE filename=? AND strftime(started_at,'%Y-%m')=? AND ride_id IS NULL
                    EXCEPT ALL SELECT {columns} FROM read_parquet({target},hive_partitioning=false) WHERE valid_demand)
                    ''',[filename,period]).fetchone()[0]
                if missing==0:
                    con.execute('INSERT INTO legacy_remove VALUES (?,?)',[filename,period])
                    legacy_proofs.append({'source_path':filename,'natural_month':period,'rows':count,
                        'proof':'complete off-month source segment contained with multiplicity in natural-month canonical multiset (EXCEPT ALL); no generic row deduplication'})
                else:
                    unproven.append({'source_path':filename,'natural_month':period,'rows':count,
                                     'unmatched_rows':missing,'potential_matching_rows':count-missing,
                                     'action':'segment not proven equivalent; preserve all legacy multiplicities'})
            con.execute("""CREATE VIEW accepted_spills AS SELECT s.* FROM spill_rows s
                WHERE NOT EXISTS(SELECT 1 FROM modern_remove r WHERE r.filename=s.filename AND r.ride_id=s.ride_id)
                AND NOT EXISTS(SELECT 1 FROM legacy_remove r WHERE r.filename=s.filename AND r.period=strftime(s.started_at,'%Y-%m') AND s.ride_id IS NULL)""")
            periods=','.join("'"+period+"'" for period in lookup)
            modern_count=con.execute('SELECT count(*) FROM modern_remove').fetchone()[0]
            area_audit={'area':area,'modern_equivalent_copies_removed':modern_count,
                        'station_id_serialization_alias_ids':[row[0] for row in raw_aliases if row[0] not in time_aliases],
                        'timestamp_precision_alias_ids':time_aliases,
                        'timestamp_precision_proof':'JC only: one complete whole-second version, identical second floors and other canonical fields; retain higher precision. NYC precision is unchanged.',
                        'station_alias_proof':'fractional trailing-zero serialization only; same ride-ID and all other canonical fields including known station name/coordinates; original ID strings retained',
                        'legacy_source_segment_proofs':legacy_proofs,
                        'legacy_equivalent_copies_removed':sum(p['rows'] for p in legacy_proofs),
                        'unproven_legacy_segments':unproven,
                        'actual_retained_timestamp_bounds':con.execute('SELECT min(started_at),max(started_at) FROM source_rows').fetchone(),
                        'cross_id_conflicts':0,'normalization':'actual local departure month; unknown coverage stays flagged in source partition',
                        'uncovered_boundary_rows':con.execute(f"SELECT count(*) FROM accepted_spills WHERE strftime(started_at,'%Y-%m') NOT IN ({periods})").fetchone()[0]}
            for part in current:
                period=part['period'];source_path=Path(part['path']).as_posix()
                outgoing=con.execute(f"SELECT count(*) FROM accepted_spills WHERE filename=? AND strftime(started_at,'%Y-%m') IN ({periods})",[source_path]).fetchone()[0]
                incoming=con.execute("SELECT count(*) FROM accepted_spills WHERE strftime(started_at,'%Y-%m')=?",[period]).fetchone()[0]
                removed=con.execute('SELECT count(*) FROM modern_remove WHERE filename=?',[source_path]).fetchone()[0]
                removed+=sum(p['rows'] for p in legacy_proofs if p['source_path']==source_path)
                if not (outgoing or incoming or removed):
                    results.append({**part,'rows_moved_in':0,'rows_moved_out':0,'cross_partition_duplicates_removed':0})
                    continue
                destination=DATA/'canonical'/area/f'year={period[:4]}'/f'month={period[5:]}'/'rides.parquet'
                receipt=DATA/'canonical_receipts'/area/f'{period}.json'
                if receipt.exists() and destination.exists():
                    saved=json.loads(receipt.read_text(encoding='utf-8'))
                    if saved['assembly_signature']==signature and sha256_file(destination)==saved['sha256']:
                        results.append(saved);print(f'Maandgrenscache {area} {period}',flush=True);continue
                started=time.perf_counter()
                own=sql_path(Path(part['path']))
                con.execute(f'''CREATE OR REPLACE VIEW final_rows AS
                    SELECT * REPLACE(
                      coalesce(strftime(started_at,'%Y-%m')='{period}',false) AS valid_demand,
                      coalesce(strftime(started_at,'%Y-%m')='{period}' AND duration_minutes>0,false) AS valid_duration)
                    FROM (
                      SELECT * EXCLUDE(filename) FROM read_parquet({own},hive_partitioning=false,filename=true) s
                      WHERE NOT EXISTS(SELECT 1 FROM modern_remove r WHERE r.filename=s.filename AND r.ride_id=s.ride_id)
                      AND NOT EXISTS(SELECT 1 FROM legacy_remove r WHERE r.filename=s.filename AND r.period=strftime(s.started_at,'%Y-%m') AND s.ride_id IS NULL)
                      AND (s.valid_demand OR s.started_at IS NULL OR strftime(s.started_at,'%Y-%m') NOT IN ({periods}))
                      UNION ALL SELECT * EXCLUDE(filename) FROM accepted_spills WHERE strftime(started_at,'%Y-%m')='{period}')''')
                after=quality(con,'final_rows')
                destination.parent.mkdir(parents=True,exist_ok=True)
                partial=destination.with_suffix('.parquet.part')
                con.execute(f'COPY final_rows TO {sql_path(partial)} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 100000)')
                partial.replace(destination)
                bounds=con.execute('SELECT min(started_at),max(started_at) FROM final_rows WHERE valid_demand').fetchone()
                result={**part,'source_cache_path':part['path'],'source_cache_sha256':part['sha256'],
                        'path':str(destination),'sha256':sha256_file(destination),'size_bytes':destination.stat().st_size,
                        'assembly_signature':signature,'retained_rows':after['rows'],
                        'removed_rows':part['removed_rows']+removed,'rows_moved_in':incoming,'rows_moved_out':outgoing,
                        'removed_percent':100*(part['removed_rows']+removed)/part['raw_rows'],
                        'cross_partition_duplicates_removed':removed,'quality_after':after,
                        'minimum_timestamp':bounds[0],'maximum_timestamp':bounds[1],
                        'assembly_seconds':time.perf_counter()-started}
                assert part['raw_rows']+incoming-outgoing==result['retained_rows']+result['removed_rows']
                atomic_json(receipt,result);results.append(result)
                print(f'Maandgrens {area} {period}: +{incoming}, -{outgoing} verplaatst; {removed} bewezen kopieën verwijderd',flush=True)
            audit.append(area_audit)
        finally:con.close()
    results.sort(key=lambda p:(p['area'],p['period']))
    assert sum(p['raw_rows'] for p in results)==sum(p['retained_rows']+p['removed_rows'] for p in results)
    atomic_json(REPORTS/'month_boundary_audit.json',audit)
    atomic_json(completion,{'signature':signature,'partitions':results,'audit':audit})
    return results,audit
