"""Betekenisvolle controles van bronverwerking, tijdcontract en gebiedsgrenzen."""
import json
import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from citibike import prepare, datasets
from citibike.config import connect,fingerprint
from citibike.features import make_features,FEATURES,folds,SavedChronologicalCV
from citibike.inference import predict_day
from citibike.modeling import SeasonalNaive,choose_model
from citibike.analysis import holm,block_test,install_duration_view


def history(start='2024-01-01',days=40):
    frame=pd.DataFrame({'timestamp':pd.date_range(start,periods=days*24,freq='h'),
                         'rides':np.arange(days*24)+10.,'area':'NYC'})
    invalid=pd.DatetimeIndex(frame.timestamp).tz_localize('America/New_York',ambiguous=True,nonexistent='NaT').isna()
    frame.loc[invalid,'rides']=0
    return frame


class TemporalTests(unittest.TestCase):
    def test_model_and_evaluation_block_changed_prediction_code(self):
        import joblib
        from citibike import modeling,inference
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);path=root/'model.joblib'
            artifact={'model':SeasonalNaive(),'features':FEATURES,'area':'NYC','target':'hourly_demand',
                      'selection':{'prediction_code_fingerprint':modeling.prediction_code_fingerprint()}}
            joblib.dump(artifact,path)
            self.assertEqual(inference.load_model(path)['area'],'NYC')
            with patch.object(modeling,'prediction_code_fingerprint',lambda:'changed'):
                with self.assertRaisesRegex(ValueError,'vastgelegde'):inference.load_model(path)
            (root/'model_selection.json').write_text(json.dumps({'prediction_code_fingerprint':'old'}))
            with patch.object(modeling,'REPORTS',root):
                with self.assertRaisesRegex(ValueError,'gewijzigd na modelvergrendeling'):modeling.evaluate()
            self.assertFalse((root/'evaluation_lock.json').exists())

    def test_canonical_model_checksum_and_selection_receipt(self):
        import joblib
        from citibike import modeling,inference
        from citibike.config import sha256_file,fingerprint
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);path=root/'citibike_model.joblib'
            selection={'prediction_code_fingerprint':modeling.prediction_code_fingerprint()}
            artifact={'model':SeasonalNaive(),'features':FEATURES,'area':'NYC',
                      'target':'hourly_demand','selection':selection}
            joblib.dump(artifact,path)
            receipt={'sha256':sha256_file(path),'selection_fingerprint':fingerprint(selection)}
            (root/'model_artifact.json').write_text(json.dumps(receipt))
            with patch.object(inference,'MODELS',root),patch.object(inference,'REPORTS',root):
                self.assertEqual(inference.load_model(path)['area'],'NYC')
                receipt['selection_fingerprint']='changed'
                (root/'model_artifact.json').write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError,'Modelselectie'):inference.load_model(path)
                path.write_bytes(path.read_bytes()+b'changed')
                with self.assertRaisesRegex(ValueError,'Modelbestand'):inference.load_model(path)

    def test_cached_final_evaluation_requires_identical_experiment(self):
        from citibike import modeling
        from citibike.config import sha256_file,fingerprint
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);path=root/'citibike_model.joblib';path.write_bytes(b'fixed model')
            selection={'dataset_fingerprint':'stable'}
            previous={'model_sha256':'wrong','selection_fingerprint':fingerprint(selection)}
            report=root/'final_evaluation.json';report.write_text(json.dumps(previous))
            with patch.object(modeling,'REPORTS',root),patch.object(modeling,'MODELS',root),\
                 patch.object(modeling,'load_model',lambda:{'selection':selection}),\
                 patch.object(modeling,'manifest',lambda:{'dataset_fingerprint':'stable'}),\
                 patch.object(modeling,'hourly') as test_reader:
                with self.assertRaisesRegex(ValueError,'Bestaande eindtest'):modeling.evaluate()
                previous['model_sha256']=sha256_file(path)
                report.write_text(json.dumps(previous))
                self.assertEqual(modeling.evaluate(),previous)
                with patch.object(modeling,'manifest',lambda:{'dataset_fingerprint':'changed'}):
                    with self.assertRaisesRegex(ValueError,'Bestaande eindtest'):modeling.evaluate()
                previous['selection_fingerprint']='changed'
                report.write_text(json.dumps(previous))
                with self.assertRaisesRegex(ValueError,'Bestaande eindtest'):modeling.evaluate()
                test_reader.assert_not_called()

    def test_future_and_same_day_cannot_change_features(self):
        original=history();changed=original.copy()
        changed.loc[changed.timestamp>='2024-02-01','rides']=999999
        left=make_features(original);right=make_features(changed)
        cutoff=left.day==pd.Timestamp('2024-02-01')
        pd.testing.assert_frame_equal(left.loc[cutoff,FEATURES],right.loc[cutoff,FEATURES])

    def test_mixed_city_rejected(self):
        frame=history();frame.loc[0,'area']='JC'
        with self.assertRaises(ValueError):make_features(frame)
        with self.assertRaises(ValueError):make_features(history().drop(columns='area'))
        with self.assertRaises(ValueError):
            predict_day('2024-02-10',history(),{'area':'JC','target':'hourly_demand'})

    def test_missing_hours_and_future_input_rejected(self):
        with self.assertRaises(ValueError):make_features(history().drop(index=10))
        with self.assertRaises(ValueError):predict_day('2024-01-20',history())

    def test_folds_full_days_and_identical_indices(self):
        frame=make_features(history(days=200));bounds=folds(frame)
        left=list(SavedChronologicalCV(bounds).split(frame))
        right=list(SavedChronologicalCV(json.loads(json.dumps(bounds))).split(frame))
        for (a,b),(c,d) in zip(left,right):
            np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)
            self.assertLess(frame.day.iloc[a[-1]],frame.day.iloc[b[0]])
            self.assertEqual(len(b)%24,0)

    def test_dst_has_24_labels_with_correct_clock_lengths(self):
        for date,expected_hour,expected_length in [('2026-03-08',2,0),('2026-11-01',1,2)]:
            origin=pd.Timestamp(date);frame=history(str((origin-pd.Timedelta(days=14)).date()),14)
            model=SeasonalNaive().fit(make_features(history())[FEATURES])
            artifact={'model':model,'features':FEATURES,'area':'NYC','target':'hourly_demand'}
            prediction=predict_day(date,frame,artifact)
            self.assertEqual(len(prediction),24)
            self.assertEqual(prediction.iloc[expected_hour].clock_hours,expected_length)
            if expected_length==0:self.assertEqual(prediction.iloc[expected_hour].predicted_rides,0)


class PipelineTests(unittest.TestCase):
    def test_jc_precision_alias_retains_milliseconds_and_nyc_stays_strict(self):
        from citibike import assembly
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/'temporary').mkdir();by_area={}
            text={'2024-05':'a,2024-05-31 23:58:12,2024-06-01 00:04:56\n',
                  '2024-06':'a,2024-05-31 23:58:12.780,2024-06-01 00:04:56.895\nb,2024-06-02 12:00:00,2024-06-02 12:10:00\n'}
            with patch.object(prepare,'DATA',root),patch.object(prepare,'PROCESSED',root/'source_cache'):
                for area in ['JC','NYC']:
                    by_area[area]=[]
                    for period,rows in text.items():
                        source=root/(area+period+'.zip')
                        with ZipFile(source,'w') as z:z.writestr('data.csv','ride_id,started_at,ended_at\n'+rows)
                        desc={'source':str(source),'nested':None,'members':['data.csv']}
                        by_area[area].append(prepare.prepare_partition(area,period,[desc],period))
            with patch.object(assembly,'DATA',root),patch.object(assembly,'REPORTS',root):
                ready,audit=assembly.assemble(by_area['JC'])
                may=pd.read_parquet(ready[0]['path'])
                self.assertEqual(may.started_at.iloc[0].microsecond,780000)
                self.assertTrue(may.valid_demand.all())
                self.assertEqual(sum(p['retained_rows'] for p in ready),2)
                self.assertEqual(audit[0]['timestamp_precision_alias_ids'],['a'])
                with self.assertRaisesRegex(ValueError,'Conflicterend ride-ID'):assembly.assemble(by_area['NYC'])

    def test_global_boundary_overlap_relocation_cache_and_conflicts(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from citibike import assembly
        from citibike.config import sha256_file
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/'temporary').mkdir();parts=[]
            header='ride_id,started_at,ended_at,bikeid,end_station_id,end_station_name,end_lat,end_lng\n'
            old=',2024-01-15 12:00:00,2024-01-15 12:10:00,001\n'
            a='a,2024-01-30 12:00:00,2024-01-30 12:10:00,\n'
            rows={'2024-01':a+old+old,'2024-02':a+'b,2024-01-31 23:50:00,2024-02-01 00:10:00,\n'+'c,2024-02-02 12:00:00,2024-02-02 12:10:00,\n'+old+old}
            with patch.object(prepare,'DATA',root),patch.object(prepare,'PROCESSED',root/'source_cache'):
                for period,text in rows.items():
                    text='\n'.join(line+(','+('4632.1' if period=='2024-01' else '4632.10')+',A,40.6,-73.9' if line.startswith('a,') else ',,,,') for line in text.splitlines())+'\n'
                    source=root/(period+'.zip')
                    with ZipFile(source,'w') as z:z.writestr('data.csv',header+text)
                    descriptor={'source':str(source),'nested':None,'members':['data.csv']}
                    parts.append(prepare.prepare_partition('NYC',period,[descriptor],period))
            with patch.object(assembly,'DATA',root),patch.object(assembly,'REPORTS',root):
                ready,audit=assembly.assemble(parts)
                self.assertEqual(sum(p['retained_rows'] for p in ready),5)
                self.assertEqual(sum(p['cross_partition_duplicates_removed'] for p in ready),3)
                self.assertEqual(audit[0]['station_id_serialization_alias_ids'],['a'])
                january=pd.read_parquet(ready[0]['path'])
                self.assertEqual(january.ride_id.isna().sum(),2)
                self.assertEqual(set(january.ride_id.dropna()),{'a','b'})
                self.assertTrue(january.valid_demand.all())
                repeated,_=assembly.assemble(parts)
                self.assertEqual([p['sha256'] for p in ready],[p['sha256'] for p in repeated])
                Path(ready[1]['path']).write_bytes(b'corrupt canonical cache')
                repaired,_=assembly.assemble(parts)
                self.assertEqual(len(pd.read_parquet(repaired[1]['path'])),1)
                changed=pd.read_parquet(parts[0]['path'])
                changed.loc[changed.ride_id=='a','ended_at']+=pd.Timedelta(minutes=5)
                original_schema=pq.read_schema(parts[0]['path'])
                pq.write_table(pa.Table.from_pandas(changed,schema=original_schema,preserve_index=False),parts[0]['path'])
                parts[0]['sha256']=sha256_file(Path(parts[0]['path']))
                with self.assertRaisesRegex(ValueError,'Conflicterend ride-ID'):assembly.assemble(parts)

    def test_mapping_region_and_malformed_member_name(self):
        self.assertEqual(prepare.canonical_name('Start Station Latitude'),'start_lat')
        self.assertEqual(prepare.canonical_name('User Type'),'member_casual')
        self.assertEqual(prepare.source_area('JC-202601-citibike-tripdata.zip'),'JC')
        self.assertEqual(prepare.source_area('2026-citibike-tripdata.zip'),'NYC')
        with self.assertRaises(ValueError):prepare.source_area('unknown.zip')
        self.assertEqual(prepare.member_period('JC-20161-citibike-tripdata.csv','2016-01'),'2016-01')

    def test_whole_csv_and_parts_same_folder_are_separate_versions(self):
        groups=prepare.logical_groups(['year/201804-citibike-tripdata.csv',
            'year/201804-citibike-tripdata_1.csv','year/201804-citibike-tripdata_2.csv',
            'year/4_April/201804-citibike-tripdata_1.csv','year/4_April/201804-citibike-tripdata_2.csv'])
        self.assertEqual(len(groups),3)
        self.assertEqual(sorted(len(values) for values in groups.values()),[1,2,2])

    def test_exact_multiset_preserves_identical_legacy_rides(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);source=root/'201301-citibike-tripdata.zip'
            header='tripduration,starttime,stoptime,start station id,start station name,start station latitude,start station longitude,end station id,end station name,end station latitude,end station longitude,bikeid,usertype,birth year,gender\n'
            row='600,2013-01-02 12:00:00,2013-01-02 12:10:00,001,A,40,-74,002,B,40,-74,1,Subscriber,1980,1\n'
            with ZipFile(source,'w') as z:
                z.writestr('201301-citibike-tripdata.csv',header+row+row)
                z.writestr('parts/201301-citibike-tripdata_1.csv',header+row+row)
                z.writestr('__MACOSX/._201301.csv','not csv')
            descriptors=[{'source':str(source),'nested':None,'members':[member]}
                         for member in ['201301-citibike-tripdata.csv','parts/201301-citibike-tripdata_1.csv']]
            (root/'data'/'temporary').mkdir(parents=True)
            with patch.object(prepare,'DATA',root/'data'),patch.object(prepare,'PROCESSED',root/'processed'):
                result=prepare.prepare_partition('NYC','2013-01',descriptors,'fixture')
                self.assertEqual(result['raw_rows'],4);self.assertEqual(result['retained_rows'],2)
                self.assertEqual(result['overlap_removed'],2);self.assertEqual(result['duplicate_ids_removed'],0)
                parquet=pd.read_parquet(result['path'])
                self.assertEqual(parquet.start_station_id.tolist(),['001','001'])
                self.assertEqual(set(parquet.member_casual),{'member'})
                again=prepare.prepare_partition('NYC','2013-01',descriptors,'fixture')
                self.assertEqual(again['sha256'],result['sha256'])
                source_changed=prepare.prepare_partition('NYC','2013-01',descriptors,'changed_signature')
                self.assertEqual(source_changed['signature'],'changed_signature')

    def test_nested_zip_and_metadata_ignored(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);inner=root/'inner.zip'
            with ZipFile(inner,'w') as z:z.writestr('202401.csv','a,b\n1,2\n')
            outer=root/'outer.zip'
            with ZipFile(outer,'w') as z:z.write(inner,'202401.zip')
            with patch.object(prepare,'DATA',root/'data'):
                with prepare.open_container(outer,'202401.zip') as z:
                    self.assertEqual(z.read('202401.csv'),b'a,b\n1,2\n')
                self.assertFalse((root/'data'/'temporary'/'nested_month.zip').exists())
            self.assertFalse(prepare.useful_member('__MACOSX/._202401.zip'))

    def test_conflicting_modern_id_stops(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);source=root/'202401-citibike-tripdata.zip'
            header='ride_id,started_at,ended_at\n'
            with ZipFile(source,'w') as z:
                z.writestr('202401.csv',header+'same,2024-01-01 01:00:00,2024-01-01 01:10:00\n'+'same,2024-01-01 02:00:00,2024-01-01 02:10:00\n')
            descriptors=[{'source':str(source),'nested':None,'members':['202401.csv']}]
            (root/'data'/'temporary').mkdir(parents=True)
            with patch.object(prepare,'DATA',root/'data'),patch.object(prepare,'PROCESSED',root/'processed'):
                with self.assertRaises(ValueError):prepare.prepare_partition('NYC','2024-01',descriptors,'fixture')

    def test_interruption_and_corrupt_cache_are_rebuilt(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);source=root/'202401-citibike-tripdata.zip'
            with ZipFile(source,'w') as z:
                z.writestr('202401.csv','ride_id,started_at,ended_at\na,2024-01-02 12:00:00,2024-01-02 12:10:00\n')
            descriptors=[{'source':str(source),'nested':None,'members':['202401.csv']}]
            (root/'data'/'temporary').mkdir(parents=True)
            with patch.object(prepare,'DATA',root/'data'),patch.object(prepare,'PROCESSED',root/'processed'):
                with patch.object(prepare,'atomic_json',side_effect=InterruptedError('fixture interruption')):
                    with self.assertRaises(InterruptedError):
                        prepare.prepare_partition('NYC','2024-01',descriptors,'fixture')
                self.assertFalse((root/'data'/'receipts'/'NYC'/'2024-01.json').exists())
                restored=prepare.prepare_partition('NYC','2024-01',descriptors,'fixture')
                self.assertEqual(restored['retained_rows'],1)
                Path(restored['path']).write_bytes(b'corrupt fixture cache')
                rebuilt=prepare.prepare_partition('NYC','2024-01',descriptors,'fixture')
                self.assertEqual(len(pd.read_parquet(rebuilt['path'])),1)

    def test_non_equivalent_source_versions_stop(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);source=root/'202401-citibike-tripdata.zip'
            with ZipFile(source,'w') as z:
                for member,hour in [('202401.csv',12),('parts/202401_1.csv',13)]:
                    z.writestr(member,f'started_at,ended_at\n2024-01-02 {hour}:00:00,2024-01-02 {hour}:10:00\n')
            descriptors=[{'source':str(source),'nested':None,'members':[member]}
                         for member in ['202401.csv','parts/202401_1.csv']]
            (root/'data'/'temporary').mkdir(parents=True)
            with patch.object(prepare,'DATA',root/'data'),patch.object(prepare,'PROCESSED',root/'processed'):
                with self.assertRaisesRegex(ValueError,'niet equivalent'):
                    prepare.prepare_partition('NYC','2024-01',descriptors,'fixture')

    def test_research_and_training_stop_after_test_opened(self):
        from citibike import analysis,modeling
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/'evaluation_lock.json').write_text('{}')
            with patch.object(analysis,'REPORTS',root),patch.object(modeling,'REPORTS',root):
                with self.assertRaises(RuntimeError):analysis.research()
                with self.assertRaises(RuntimeError):modeling.train()

    def test_invalid_fields_flagged_not_silently_dropped(self):
        import pyarrow as pa
        con=connect()
        values={c:[''] for c in prepare.COLUMNS}
        values.update(started_at=['bad'],ended_at=['2024-01-01 00:00:00'],start_lat=['200'])
        con.register('batch',pa.table(values))
        result=con.execute(prepare.transform_sql('NYC','2024-01')).df();con.close()
        self.assertEqual(len(result),1);self.assertFalse(result.valid_demand.iloc[0])
        self.assertFalse(result.valid_start_coordinates.iloc[0])
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'year=2024'/'month=01'/'rides.parquet';path.parent.mkdir(parents=True)
            result.to_parquet(path,index=False)
            m={'partitions':[{'area':'NYC','period':'2024-01','path':str(path),'quality_after':{'valid_demand':0}}]}
            with patch.object(datasets,'manifest',lambda:m):
                with self.assertRaisesRegex(ValueError,'geen nulimputatie'):datasets.hourly('NYC')

    def test_test_access_guard_and_region_partition_filter(self):
        value={'partitions':[{'area':a,'period':p,'path':f'{a}/{p}'} for a in ['NYC','JC'] for p in ['2026-04','2026-05','2026-07']]}
        with tempfile.TemporaryDirectory() as name,patch.object(datasets,'REPORTS',Path(name)),patch.object(datasets,'manifest',lambda:value):
            self.assertTrue(all('NYC' in str(p) for p in datasets.partition_paths('NYC','development')))
            self.assertEqual(len(datasets.partition_paths('NYC','development')),1)
            self.assertEqual(len(datasets.partition_paths('NYC','validation')),1)
            with self.assertRaises(PermissionError):datasets.partition_paths('NYC','test')

    def test_missing_source_month_is_never_filled_with_zero(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);partitions=[]
            for month in [1,3]:
                path=root/'NYC'/'year=2024'/f'month={month:02d}'/'rides.parquet'
                path.parent.mkdir(parents=True)
                pd.DataFrame({'started_at':[pd.Timestamp(2024,month,1)],
                              'valid_demand':[True],'area':['NYC']}).to_parquet(path,index=False)
                partitions.append({'area':'NYC','period':f'2024-{month:02d}','path':str(path)})
            with patch.object(datasets,'manifest',lambda:{'partitions':partitions}):
                with self.assertRaisesRegex(ValueError,'Ontbrekende bronmaanden'):datasets.hourly('NYC')

    def test_jersey_rows_in_nyc_partition_are_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);path=root/'rides.parquet'
            pd.DataFrame({'started_at':[pd.Timestamp('2024-01-01')],
                          'valid_demand':[True],'area':['JC']}).to_parquet(path,index=False)
            m={'partitions':[{'area':'NYC','period':'2024-01','path':str(path)}]}
            with patch.object(datasets,'manifest',lambda:m):
                with self.assertRaisesRegex(AssertionError,'Gebiedsvervuiling'):datasets.hourly('NYC')

    def test_raw_hour_aggregation_handles_both_dst_transitions(self):
        cases=[('2025-03','2025-03-09',['01:20:00','03:20:00'],2,0),
               ('2025-11','2025-11-02',['01:10:00','01:20:00'],1,2)]
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            for period,day,times,hour,expected in cases:
                path=root/'NYC'/f'year={period[:4]}'/f'month={period[5:]}'/'rides.parquet'
                path.parent.mkdir(parents=True)
                pd.DataFrame({'started_at':pd.to_datetime([day+' '+time for time in times]),
                              'valid_demand':[True,True],'area':['NYC','NYC']}).to_parquet(path,index=False)
                m={'partitions':[{'area':'NYC','period':period,'path':str(path)}]}
                with patch.object(datasets,'manifest',lambda:m):frame=datasets.hourly('NYC')
                selected=frame[frame.timestamp.dt.normalize()==pd.Timestamp(day)]
                self.assertEqual(len(selected),24)
                self.assertEqual(selected.rides.sum(),2)
                self.assertEqual(selected.loc[selected.timestamp.dt.hour==hour,'rides'].iloc[0],expected)


class StatisticalTests(unittest.TestCase):
    def test_development_aggregation_queries_accept_standard_schema(self):
        from contextlib import contextmanager
        import pyarrow as pa
        from citibike import analysis
        @contextmanager
        def fixture_rides(area):
            con=connect()
            try:
                values={column:['',''] for column in prepare.COLUMNS}
                values.update(started_at=['2024-01-02 12:00:00','2024-01-02 13:00:00'],
                              ended_at=['2024-01-02 12:10:00','2024-01-02 13:20:00'],
                              member_casual=['member','casual'],
                              start_station_id=['001','002'],end_station_id=['002','001'])
                con.register('batch',pa.table(values))
                con.execute('CREATE TABLE rides AS '+prepare.transform_sql(area,'2024-01'))
                yield con
            finally:con.close()
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            with patch.object(analysis,'rides',fixture_rides),patch.object(analysis,'REPORTS',root),patch.object(analysis,'DATA',root):
                result=analysis.summarize('NYC')
                self.assertEqual(result['monthly'].rides.sum(),2)
                self.assertEqual(result['duration_statistics'].rides.iloc[0],2)
                self.assertTrue((root/'NYC_missing_values.csv').exists())

    def test_physical_duration_dst_and_label_availability(self):
        import pyarrow as pa
        con=connect()
        cases=[('2026-03','2026-03-08 01:50:00','2026-03-08 03:10:00','',20.,True),
               ('2025-11','2025-11-02 01:50:00','2025-11-02 01:20:00','3000',50.,True),
               ('2025-11','2025-11-02 01:10:00','2025-11-02 01:40:00','',30.,False),
               ('2026-04','2026-04-30 23:50:00','2026-05-01 00:10:00','',20.,False),
               ('2026-01','2026-01-02 12:00:00','2026-01-02 12:10:00','Infinity',np.inf,False)]
        for index,(period,start,end,reported,expected,eligible) in enumerate(cases):
            values={c:[''] for c in prepare.COLUMNS}
            values.update(started_at=[start],ended_at=[end],reported_duration_seconds=[reported])
            con.register('batch',pa.table(values))
            query=prepare.transform_sql('NYC',period)
            con.execute(('CREATE TABLE rides AS ' if index==0 else 'INSERT INTO rides ')+query)
            con.unregister('batch')
        install_duration_view(con)
        results=con.execute('SELECT duration_minutes,valid_duration FROM duration_rides ORDER BY started_at').fetchall()
        expected=sorted(cases,key=lambda c:c[1])
        for (minutes,valid),case in zip(results,expected):
            # Ambigue duur blijft diagnostisch maar mag niet in de statistiek.
            if case[-1]:self.assertAlmostEqual(minutes,case[-2])
            self.assertEqual(valid,case[-1])
        con.close()
    def test_holm_and_block_test(self):
        np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])
        test=block_test(pd.Series(np.arange(80)+1),'test','unit')
        self.assertEqual(test['blocks'],20);self.assertLess(test['p_value'],.05)
    def test_selection_prefers_simple_within_one_percent(self):
        winner=choose_model([{'name':'lightgbm','RMSE':100},{'name':'ridge','RMSE':100.5}])
        self.assertEqual(winner['name'],'ridge')
        boundary=choose_model([{'name':'lightgbm','RMSE':100},{'name':'ridge','RMSE':101}])
        self.assertEqual(boundary['name'],'lightgbm')


if __name__=='__main__':unittest.main()
