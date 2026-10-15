"""Controleer de echte lokale team-API en Streamlit tegen gedeelde offline inference."""
from pathlib import Path
import argparse
import json
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from citibike.config import atomic_json,now
from citibike.inference import load_model,predict_day


def main(api):
    artifact=load_model()
    full=pd.read_parquet(ROOT/'data/NYC_hourly_development.parquet')
    checked=[]
    for date in ['2026-04-30','2026-03-08','2025-11-02']:
        origin=pd.Timestamp(date)
        history=full[(full.timestamp>=origin-pd.Timedelta(days=14))&(full.timestamp<origin)].copy()
        expected=predict_day(date,history,artifact)
        body=json.dumps({'date':date,'hour':8,'mode':'history',
                         'history':history.assign(timestamp=history.timestamp.astype(str)).to_dict('records')}).encode()
        request=urllib.request.Request(api+'/predict/citibike',data=body,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=90) as response: actual=json.load(response)
        np.testing.assert_allclose([r['predicted_ride_starts'] for r in actual['daily_predictions']],expected.predicted_rides,rtol=1e-12,atol=1e-9)
        assert [r['clock_hours'] for r in actual['daily_predictions']]==expected.clock_hours.tolist()
        assert actual['model_id']=='raoul_tuned_random_forest_320'
        checked.append({'date':date,'hours':24,'physical_hours':int(expected.clock_hours.sum()),'offline_API_equal':True})
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
    assert not app.exception,str(app.exception)
    app.date_input[0].set_value(pd.Timestamp('2026-04-30').date())
    app.button[0].click().run()
    assert not app.exception,str(app.exception)
    history=full[(full.timestamp>='2026-04-16')&(full.timestamp<'2026-04-30')]
    expected=predict_day('2026-04-30',history,artifact)
    np.testing.assert_allclose(app.dataframe[0].value['verwachte_ritten'],expected.predicted_rides,rtol=1e-12,atol=1e-9)
    result={'checked_at':now(),'actual_HTTP_API':api,'days':checked,'same_model_id':True,
            'streamlit_form_offline_equal':True,'frontend_API_contract_tests':'19 Python + 16 JavaScript tests; see implementation_checks.json',
            'test_period_used':False,'live_hosting_redeployed':False}
    atomic_json(ROOT/'reports/inference_parity.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--api',default='http://127.0.0.1:8765')
    main(parser.parse_args().api)
