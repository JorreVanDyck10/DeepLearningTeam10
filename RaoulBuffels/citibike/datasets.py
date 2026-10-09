"""Datasettoegang die onderzoeksdata en eindtest fysiek gescheiden houdt."""
import json
from contextlib import contextmanager
from pathlib import Path
import pandas as pd
import numpy as np

from .config import REPORTS, DATA, TRAIN_END, TEST_START, TEST_END, connect, sql_path


def manifest():
    return json.loads((REPORTS / 'data_manifest.json').read_text(encoding='utf-8'))


def partition_paths(area: str, stage: str):
    if area not in ('NYC','JC'):
        raise ValueError('Gebied moet expliciet NYC of JC zijn')
    if stage not in ('development','validation','test'):
        raise ValueError('Onbekende datastage')
    if stage=='test' and not (REPORTS/'evaluation_lock.json').exists():
        raise PermissionError('Eindtest is gesloten tot modelkeuze en evaluatievergrendeling')
    paths=[]
    for p in manifest()['partitions']:
        if p['area']!=area:
            continue
        date=p['period']+'-01'
        if ((stage=='development' and date<TRAIN_END) or
            (stage=='validation' and TRAIN_END<=date<TEST_START) or
            (stage=='test' and TEST_START<=date<TEST_END)):
            paths.append(Path(p['path']))
    if not paths:
        raise ValueError(f'Geen partitions voor {area}/{stage}')
    return paths


@contextmanager
def rides(area='NYC',stage='development'):
    paths=partition_paths(area,stage)
    con=connect()
    literals='['+','.join(sql_path(p) for p in paths)+']'
    try:
        con.execute(f'CREATE VIEW rides AS SELECT * FROM read_parquet({literals},hive_partitioning=false)')
        areas=con.execute('SELECT DISTINCT area FROM rides').fetchall()
        if areas!=[(area,)]:
            raise AssertionError(f'Gebiedsvervuiling: verwacht {area}, kreeg {areas}')
        yield con
    finally:
        con.close()


def hourly(area='NYC',stage='development'):
    """Alleen nul binnen aanwezige volledige bronmaanden; gaten blijven fouten."""
    selected={str(path) for path in partition_paths(area,stage)}
    for part in manifest()['partitions']:
        if part['path'] in selected and part.get('quality_after',{}).get('valid_demand')==0:
            raise ValueError('Maand zonder bruikbare vertrektijden; geen nulimputatie bij parsing- of dekkingsfouten')
    with rides(area,stage) as con:
        values=con.execute("SELECT date_trunc('hour',started_at) AS \"timestamp\",count(*) AS rides FROM rides WHERE valid_demand GROUP BY 1 ORDER BY 1").df()
    if values.empty:
        raise ValueError('Geen bruikbare vertrekken; onbekende tellingen zijn geen nullen')
    paths=partition_paths(area,stage)
    grids=[]
    for path in paths:
        year=int(path.parent.parent.name.split('=')[1]);month=int(path.parent.name.split('=')[1])
        first=pd.Timestamp(year,month,1)
        grids.append(pd.date_range(first,first+pd.offsets.MonthBegin(1),freq='h',inclusive='left'))
    grid=pd.DatetimeIndex(np.concatenate([g.values for g in grids])).sort_values()
    if not grid.equals(pd.date_range(grid[0],grid[-1],freq='h')):
        raise ValueError('Ontbrekende bronmaanden: niet ingevuld met nul')
    frame=values.set_index('timestamp').reindex(grid,fill_value=0).rename_axis('timestamp').reset_index()
    localized=grid.tz_localize('America/New_York',ambiguous=True,nonexistent='NaT')
    if frame.loc[localized.isna(),'rides'].sum()!=0:
        raise ValueError('Geregistreerde rit in niet-bestaand zomertijduur: onderzoek bronfout')
    frame['area']=area
    return frame


def historical_hourly(area='NYC',include_validation=False):
    frame=hourly(area,'development')
    if include_validation:
        frame=pd.concat([frame,hourly(area,'validation')],ignore_index=True)
    return frame


def require_nyc(frame):
    if 'area' not in frame or set(frame['area'].unique())!={'NYC'}:
        raise ValueError('NYC-functie vereist uitsluitend expliciet gemarkeerde NYC-data')
