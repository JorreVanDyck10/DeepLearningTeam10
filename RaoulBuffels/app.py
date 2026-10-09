"""Lokale inference; training gebeurt uitsluitend via de offline CLI."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st
from citibike.config import DATA,REPORTS,MODELS
from citibike.inference import load_model,predict_day

st.set_page_config(page_title='Citi Bike dagvooruit',page_icon=':material/directions_bike:')
st.title('Citi Bike: vertrekken per uur')
st.caption('NYC · voorspelling om 00:00 New York-tijd · één volledige kalenderdag')

@st.cache_resource(max_entries=2)
def cached_model(path,mtime):
    return load_model(Path(path))

@st.cache_data(max_entries=2)
def cached_history(path,mtime):
    return pd.read_parquet(path)

model_path=MODELS/'citibike_model.joblib'
if not model_path.exists():
    st.info('Train het model eerst offline volgens de README. De app traint zelf geen model.')
    st.stop()
artifact=cached_model(str(model_path),model_path.stat().st_mtime_ns)
st.caption(f"Trainingsdata tot en met {artifact['trained_through']}; {artifact['training_rows']:,} historische uurvakken.")
mode=st.segmented_control('Invoer',options=['Historische demonstratie','Eigen recente historie'],default='Historische demonstratie')
if mode=='Historische demonstratie':
    path=DATA/'NYC_hourly_development.parquet'
    if not path.exists():
        st.error('Ontwikkelingshistorie ontbreekt. Voer eerst het onderzoek uit.');st.stop()
    historical=cached_history(str(path),path.stat().st_mtime_ns)
    earliest=historical.timestamp.min().normalize()+pd.Timedelta(days=14)
    latest=historical.timestamp.max().normalize()
    with st.form('demonstration'):
        date=st.date_input('Voorspeldag in de historische dataset',value=latest.date(),min_value=earliest.date(),max_value=latest.date())
        submitted=st.form_submit_button('Voorspel de dag')
    history=historical[historical.timestamp<pd.Timestamp(date)]
    st.caption('Demonstratie met ontwikkelingsdata. De gekozen datum kan in de training zitten; dit is geen onafhankelijke prestatietest.')
else:
    st.write('Upload een CSV met timestamp, rides en area. Gebruik area=NYC en lokale kloklabels zonder UTC-offset. Lever minimaal de 14 volledige dagen vóór de voorspeldag.')
    with st.form('uploaded'):
        uploaded=st.file_uploader('Historische uurtellingen',type=['csv'])
        date=st.date_input('Voorspeldag')
        submitted=st.form_submit_button('Voorspel de dag')
    history=None
    if submitted and uploaded is not None:
        try:history=pd.read_csv(uploaded,parse_dates=['timestamp'])
        except (ValueError,KeyError) as error:st.error(str(error))

if submitted:
    if history is None:
        st.error('Upload eerst de historische tellingen.')
    else:
        try:
            prediction=predict_day(date,history,artifact)
            with st.container(border=True):
                st.metric('Verwacht dagtotaal',f"{prediction.predicted_rides.sum():,.0f} ritten")
                st.line_chart(prediction.set_index('timestamp')[['predicted_rides']],x_label='Lokaal uur in New York',y_label='Verwachte ritten per klokvak')
                st.dataframe(prediction.rename(columns={'predicted_rides':'verwachte_ritten','clock_hours':'fysieke_uren'}),hide_index=True)
                st.download_button('Download voorspelling',prediction.to_csv(index=False),'citibike_prediction.csv','text/csv')
        except (ValueError,KeyError,AssertionError) as error:
            st.error(str(error))

st.write('Het model schat geregistreerde vertrekken, geen beschikbare fietsen of onvervulde vraag. Weer en evenementen ontbreken. Historische prestaties garanderen geen prestaties op toekomstige dagen.')
st.caption('Bij zomertijd blijven er 24 kloklabels: het ontbrekende uur heeft nul ritten; het herhaalde uur bevat beide fysieke uren.')
st.caption('De historische evaluatie veronderstelt tijdig beschikbare uurtellingen. Publieke maandarchieven vormen geen live tellingenfeed; voor dagelijks gebruik moet je de recente historie zelf tijdig aanleveren.')
if (REPORTS/'final_evaluation.json').exists():
    result=json.loads((REPORTS/'final_evaluation.json').read_text(encoding='utf-8'))
    st.caption(f"Model: {result['selected_model']}. Eindtest juli–augustus 2026: RMSE {result['metrics']['RMSE']:.2f} ritten/uur; MAE {result['metrics']['MAE']:.2f} ritten/uur.")
