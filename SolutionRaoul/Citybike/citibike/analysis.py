"""Vragen → aggregaties → geblokte toetsen → vastgelegde doelkeuze."""
import json
import time
import numpy as np
import pandas as pd
from scipy import stats

from .config import REPORTS, DATA, SEED, TRAIN_END, atomic_json, fingerprint, now
from .datasets import rides, hourly, manifest

CONTRACTS = [
    {'target':'hourly_demand','prediction_time':'00:00 America/New_York op voorspeldag D',
     'horizon':'alle 24 lokale kloklabels op D (23/25 fysieke uren bij DST)',
     'available':'kalender van D; gemeten tellingen strikt vóór D',
     'forbidden':'tellingen eerder op D, weer achteraf, eindstation, duur van toekomstige ritten'},
    {'target':'daily_demand','prediction_time':'00:00 America/New_York op voorspeldag D',
     'horizon':'volledig dagtotaal op D', 'available':'kalender van D; historie strikt vóór D',
     'forbidden':'delen van dagtotaal D; informatie die pas op D beschikbaar komt'},
    {'target':'duration','prediction_time':'bij vertrek van één rit',
     'horizon':'duur tot beëindiging van deze rit (geen vaste kalenderhorizon)',
     'available':'starttijd, startstation, bekende gebruiker- en fietscategorie',
     'forbidden':'eindtijd, eindstation, ritduur; nog niet afgeronde trainingslabels'},
]


def holm(p_values):
    values=np.asarray(p_values,float); order=np.argsort(values)
    adjusted=np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values))))
    result=np.empty(len(values));result[order]=np.minimum(adjusted,1)
    return result


def block_test(weekly_effects, label, unit, block_weeks=4, repetitions=20000):
    """Niet-overlappende vierweekblokken; sign-flip en bootstrap op blokniveau."""
    series=pd.Series(weekly_effects).dropna()
    n=(len(series)//block_weeks)*block_weeks
    if n<block_weeks*8:
        raise ValueError('Te weinig tijdsblokken voor toets')
    effects=series.iloc[-n:].to_numpy().reshape(-1,block_weeks).mean(axis=1)
    rng=np.random.default_rng(SEED)
    observed=effects.mean()
    null=(rng.choice([-1,1],size=(repetitions,len(effects)))*effects).mean(axis=1)
    p=(1+np.sum(np.abs(null)>=abs(observed)))/(repetitions+1)
    boot=rng.choice(effects,size=(repetitions,len(effects)),replace=True).mean(axis=1)
    ci=np.quantile(boot,[.025,.975])
    return {'label':label,'H0':'Gemiddeld blokeffect is nul','H1':'Gemiddeld blokeffect verschilt van nul',
            'alpha':.05,'test':'two-sided sign-flip van niet-overlappende vierweekblokken',
            'statistic':float(observed),'effect_size':float(observed),'unit':unit,
            'p_value':float(p),'ci95':ci.tolist(),'weeks':n,'blocks':len(effects),
            'block_weeks':block_weeks,'repetitions':repetitions,
            'standardized_effect':float(observed/effects.std(ddof=1)) if effects.std(ddof=1)>0 else None,
            'assumptions':'Blokken voldoende onafhankelijk; sign-flip vereist symmetrie onder H0. Associatie, geen causaliteit.'}


def install_duration_view(con):
    """Fysieke duur waar identificeerbaar; labels uitsluitend vóór ontwikkelingscutoff."""
    con.execute("""CREATE VIEW duration_rides AS
          WITH clocks AS (
            SELECT *, (
              (month(started_at)=11 AND day(started_at)<=7 AND dayofweek(started_at)=0 AND hour(started_at)=1)
              OR (month(ended_at)=11 AND day(ended_at)<=7 AND dayofweek(ended_at)=0 AND hour(ended_at)=1)
            ) AS duration_ambiguous,
            CASE WHEN reported_duration_seconds IS NOT NULL THEN reported_duration_seconds/60.0
              ELSE epoch(timezone('America/New_York',ended_at)-timezone('America/New_York',started_at))/60.0 END AS physical_duration
            FROM rides
          ) SELECT * EXCLUDE(duration_minutes,valid_duration,physical_duration),
            physical_duration AS duration_minutes,
            coalesce(valid_demand AND ended_at<TIMESTAMP '2026-05-01' AND isfinite(physical_duration) AND physical_duration>0 AND
                     (reported_duration_seconds IS NOT NULL OR NOT duration_ambiguous),false) AS valid_duration
          FROM clocks""")


def audit_duration_sample(sample,area):
    """Onderzoek extreme ontwikkelingslabels zonder ze statistisch weg te filteren."""
    if area not in ('NYC','JC') or (sample.started_at>=pd.Timestamp(TRAIN_END)).any():
        raise ValueError('Duurdiagnose vereist afzonderlijke ontwikkelingsdata')
    extremes=sample.nlargest(10,'duration_minutes')[
        ['started_at','ended_at','duration_minutes','member_casual','rideable_type']].copy()
    extremes.to_csv(REPORTS/f'{area}_extreme_duration_examples.csv',index=False)
    atomic_json(REPORTS/f'{area}_duration_sample_audit.json',{
        'area':area,'development_only':True,'sample_rows':len(sample),
        'sample_over_24h':int((sample.duration_minutes>1440).sum()),
        'sample_over_7days':int((sample.duration_minutes>7*1440).sum()),
        'median_minutes':float(sample.duration_minutes.median()) if len(sample) else None,
        'maximum_minutes':float(sample.duration_minutes.max()) if len(sample) else None,
        'interpretation':'Extreme start/eindtijden kunnen weken of maanden uiteenliggen. Dit meet een geregistreerde open rit, niet aantoonbaar voortdurend fietsen. Registratiefouten of laat afgesloten ritten zijn mogelijke verklaringen, geen bewezen oorzaken. De mediaan beschrijft een typische geregistreerde duur beter dan het gemiddelde.',
        'action':'Positieve identificeerbare duur blijft behouden en langduur wordt gemarkeerd. Geen statistische cutoff of nieuwe verwijderregel; individuele ritduur is niet het gekozen modeldoel.',
        'sampling_limitation':'Deterministische, begrensde steekproef; voorbeelden bewijzen geen foutfrequentie in de hele dataset.'})


def summarize(area):
    with rides(area) as con:
        from .prepare import COLUMNS
        missing=con.execute('SELECT '+','.join(f'count(*) FILTER(WHERE {column} IS NULL)' for column in COLUMNS)+' FROM rides').fetchone()
        pd.DataFrame({'column':COLUMNS,'missing':missing}).to_csv(REPORTS/f'{area}_missing_values.csv',index=False)
        install_duration_view(con)
        summaries={
            'monthly':"SELECT date_trunc('month',started_at) period,count(*) rides,approx_quantile(duration_minutes,.5) FILTER(WHERE valid_duration) median_duration FROM rides WHERE valid_demand GROUP BY 1 ORDER BY 1",
            'user_bike':"SELECT member_casual,rideable_type,count(*) rides,avg(duration_minutes) FILTER(WHERE valid_duration) mean_duration,approx_quantile(duration_minutes,.5) FILTER(WHERE valid_duration) median_duration FROM rides WHERE valid_demand GROUP BY 1,2 ORDER BY 3 DESC",
            'start_stations':"SELECT start_station_id,count(*) rides,count(DISTINCT start_station_name) name_variants,count(DISTINCT year(started_at)) active_years,any_value(start_station_name) example_name FROM rides WHERE valid_demand AND start_station_id IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 25",
            'end_stations':"SELECT end_station_id,count(*) rides,any_value(end_station_name) example_name FROM rides WHERE valid_demand AND end_station_id IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 25",
            'routes':"SELECT start_station_id,end_station_id,count(*) rides FROM rides WHERE valid_demand AND start_station_id IS NOT NULL AND end_station_id IS NOT NULL GROUP BY 1,2 ORDER BY 3 DESC LIMIT 20",
            'inflow_outflow':"WITH flows AS (SELECT start_station_id station_id,-count(*) net FROM rides WHERE valid_demand AND start_station_id IS NOT NULL GROUP BY 1 UNION ALL SELECT end_station_id,count(*) FROM rides WHERE valid_demand AND end_station_id IS NOT NULL GROUP BY 1) SELECT station_id,sum(net) net_inflow FROM flows GROUP BY 1 ORDER BY abs(sum(net)) DESC LIMIT 25",
            'duration_daily':"SELECT date_trunc('day',started_at) AS \"day\",member_casual,count(*) rides,avg(duration_minutes) mean_duration,approx_quantile(duration_minutes,.5) median_duration FROM rides WHERE valid_duration AND started_at>='2023-01-01' AND member_casual IN ('member','casual') GROUP BY 1,2 ORDER BY 1",
            'duration_statistics':"SELECT count(*) rides,avg(duration_minutes) mean,stddev_samp(duration_minutes) sd,approx_quantile(duration_minutes,[.25,.5,.75,.95,.99]) quantiles,min(duration_minutes) minimum,max(duration_minutes) maximum FROM rides WHERE valid_duration",
            'coordinate_quality':"SELECT year(started_at) AS \"year\",count(*) rides,count(*) FILTER(WHERE valid_start_coordinates AND valid_end_coordinates) valid_coordinates,count(*) FILTER(WHERE start_station_id IS NULL) missing_start_station,count(*) FILTER(WHERE end_station_id IS NULL) missing_end_station FROM rides WHERE valid_demand GROUP BY 1 ORDER BY 1",
        }
        result={name:con.execute(query.replace('FROM rides','FROM duration_rides')).df() for name,query in summaries.items()}
        duration_policy=con.execute("SELECT count(*) FILTER(WHERE duration_ambiguous AND reported_duration_seconds IS NULL) uncertain_physical_duration,count(*) FILTER(WHERE valid_duration) usable_physical_duration FROM duration_rides").df()
        duration_policy.to_csv(REPORTS/f'{area}_duration_time_quality.csv',index=False)
        sample=con.execute("""SELECT ride_id,bike_id,started_at,ended_at,start_station_id,end_station_id,
           duration_minutes,member_casual,rideable_type,start_lat,start_lng
           FROM duration_rides WHERE valid_duration AND hash(started_at,ended_at,ride_id,bike_id)%1000=0
           ORDER BY hash(started_at,ended_at,ride_id,bike_id) LIMIT 200000""").df()
        atomic_json(REPORTS/f'{area}_sample_duplicates.json',{
            'sample_rows':len(sample),'duplicate_sample_rows':int(sample.duplicated().sum()),
            'interpretation':'Exact equality of selected sample fields is a diagnostic, not proof of erroneous duplicated trips. Legacy identical trips are retained. No population count extrapolation.'})
    for name,frame in result.items():
        frame.to_csv(REPORTS/f'{area}_{name}.csv',index=False)
    sample.to_parquet(DATA/f'{area}_eda_sample.parquet',index=False)
    audit_duration_sample(sample,area)
    return result


def hypothesis_family(hourly_frame,duration_daily):
    # Uitsluitend de drie meest recente ontwikkelingsjaren voor praktische relevantie.
    frame=hourly_frame[hourly_frame.timestamp>='2023-01-01'].copy()
    frame['day']=frame.timestamp.dt.normalize()
    frame['morning']=np.where(frame.timestamp.dt.hour.between(7,9),frame.rides,0)
    daily=frame.groupby('day')[['rides','morning']].sum()
    daily['weekday']=daily.index.dayofweek
    daily['week']=daily.index-pd.to_timedelta(daily.index.dayofweek,unit='D')
    daily['share']=100*daily.morning/daily.rides.replace(0,np.nan)
    complete=daily.groupby('week').size();daily=daily[daily.week.isin(complete[complete==7].index)]
    effects=daily.groupby('week').apply(lambda g:g.loc[g.weekday<5,'share'].mean()-g.loc[g.weekday>=5,'share'].mean())
    h1=block_test(effects,'Ochtendrit-aandeel: werkdag minus weekend','procentpunt')
    # De weekvertraging wordt hier onderzocht als associatie, nog niet als feature gebouwd.
    frame['previous_week']=frame.rides.shift(168)
    frame['week']=frame.timestamp.dt.normalize()-pd.to_timedelta(frame.timestamp.dt.dayofweek,unit='D')
    correlations=frame.dropna().groupby('week').apply(lambda g:g.rides.corr(g.previous_week) if len(g)==168 else np.nan)
    h2=block_test(correlations,'Associatie van uurtelling met hetzelfde uur vorige week','Pearson r')
    durations=duration_daily.copy();durations['day']=pd.to_datetime(durations.day)
    paired=durations.pivot(index='day',columns='member_casual',values='median_duration').dropna()
    paired['week']=paired.index-pd.to_timedelta(paired.index.dayofweek,unit='D')
    sizes=paired.groupby('week').size();paired=paired[paired.week.isin(sizes[sizes==7].index)]
    effects=(paired.casual-paired.member).groupby(paired.week).mean()
    h3=block_test(effects,'Dagmediaan ritduur: casual minus member','minuten')
    results=[h1,h2,h3]
    for record,adjusted in zip(results,holm([r['p_value'] for r in results])):
        record['holm_p_value']=float(adjusted);record['reject_H0']=bool(adjusted<.05)
        record['family']='drie kandidaatdoel-hypothesen; vooraf vastgelegd vóór berekening'
    # Sensitiviteit voor langere afhankelijkheid: 8-weekblokken, geen nieuwe selectie op p-waarde.
    for record,effect in zip(results,[daily.groupby('week').apply(lambda g:g.loc[g.weekday<5,'share'].mean()-g.loc[g.weekday>=5,'share'].mean()),correlations,effects]):
        sensitivity=block_test(effect,record['label'],record['unit'],block_weeks=8)
        record['sensitivity_8weeks']={k:sensitivity[k] for k in ['statistic','p_value','ci95','blocks']}
    return results


def research():
    if (REPORTS/'final_evaluation.json').exists() or (REPORTS/'evaluation_lock.json').exists():
        raise RuntimeError('Eindtest al bekeken: start geen nieuwe onderzoekskeuzes onder dezelfde testclaim')
    start=time.perf_counter()
    REPORTS.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    atomic_json(REPORTS/'prediction_contracts.json',{'created_at':now(),'contracts':CONTRACTS})
    results={}
    for area in ('NYC','JC'):
        print(f'Ontwikkelings-EDA {area}',flush=True)
        results[area]=summarize(area)
        frame=hourly(area);frame.to_parquet(DATA/f'{area}_hourly_development.parquet',index=False)
    frame=pd.read_parquet(DATA/'NYC_hourly_development.parquet')
    # EDA-grafieken daadwerkelijk uitvoeren vóór hypothesetoetsen en modeltraining.
    from .plots import figure
    import matplotlib.pyplot as plt
    figure_directory=REPORTS/'eda_figures';figure_directory.mkdir(exist_ok=True)
    interpretations=[]
    for area in ('NYC','JC'):
        kinds=['temporal','hour_weekday','duration','user_bike','stations','routes','spatial','acf'] if area=='NYC' else ['temporal','duration','user_bike']
        for kind in kinds:
            chart,interpretation=figure(kind,area)
            chart.savefig(figure_directory/f'{area}_{kind}.png',dpi=120)
            plt.close(chart)
            interpretations.append({'area':area,'question':kind,'interpretation':interpretation})
            print(f'EDA {area}/{kind}: {interpretation}',flush=True)
    atomic_json(REPORTS/'eda_interpretations.json',interpretations)
    atomic_json(REPORTS/'hypothesis_protocol.json',{'created_at':now(),'alpha':.05,'family_size':3,
          'multiple_testing':'Holm-Bonferroni','block_weeks':4,'sensitivity_block_weeks':8,
          'development_end':TRAIN_END,'features_not_yet_engineered':True,
          'hypotheses':['weekday_weekend_morning_share','weekly_hourly_association','casual_member_duration'],
          'selected_after_EDA':True,'EDA_evidence':interpretations,
          'interpretation':'Exploratory, data-informed hypotheses; p-values are not independent confirmatory evidence. Practical predictability must be established on later validation.'})
    from statsmodels.tsa.stattools import adfuller
    recent_daily=frame[frame.timestamp>='2023-01-01'].set_index('timestamp').rides.resample('D').sum()
    adf=adfuller(recent_daily,maxlag=28,autolag='AIC')
    atomic_json(REPORTS/'time_diagnostics.json',{
        'series':'dagtotalen vanaf 2023, uitsluitend ontwikkeling',
        'ADF_statistic':float(adf[0]),'ADF_p_value':float(adf[1]),'ADF_lags':int(adf[2]),
        'daily_ACF_1':float(recent_daily.autocorr(1)),'daily_ACF_7':float(recent_daily.autocorr(7)),
        'interpretation':'ADF toetst een unit root onder een specifiek model; seizoenen/structuurbreuken kunnen blijven bestaan. Dit is een diagnose en wordt niet gebruikt voor hypothese- of modelselectie.'})
    hypotheses=hypothesis_family(frame,results['NYC']['duration_daily'])
    atomic_json(REPORTS/'hypotheses.json',hypotheses)
    signal=hypotheses[1]
    supported=signal['reject_H0'] and signal['ci95'][0]>.3
    # Vastgelegde gate: sterke wekelijkse associatie en complete recente uurtellingen.
    candidates=[
      {'target':'hourly_demand','useful':'Schat geregistreerd vertrekvolume; geen onvervulde vraag',
       'type':'time-series forecasting via regression','target_present':True,'unit':'ritten per klokvak',
       'modeling_rows_before_features':len(frame),
       'data_quality':'Aaneengesloten volledige bronmaanden en gevalideerde vertrektijden; ontbrekende bronmaanden worden niet als nul ingevuld. Tellingen meten gebruik, geen onvervulde vraag.',
       'leakage_risk':'tellingen op D zijn verboden; alleen volledige dagen vóór D',
       'evidence':f"weekcorrelatie {signal['effect_size']:.4f}; Holm-p {signal['holm_p_value']:.6g}",
       'available_features':'kalender en uitsluitend afgeronde dagen',
       'deployment':'datum plus 14 volledige historische dagen',
       'selected':supported},
      {'target':'daily_demand','useful':'Dagplanning, minder detail voor piekuren',
       'type':'time-series forecasting via regression','target_present':True,'unit':'ritten per kalenderdag',
       'modeling_rows_before_features':len(frame)//24,
       'data_quality':'Dezelfde gevalideerde vertrekken als het uurdoel, samengevoegd per dag; minder waarnemingen en verlies van piekuurdetail.',
       'leakage_risk':'delen van het huidige dagtotaal mogen niet als invoer dienen',
       'deployment':'datum plus voorafgaande dagtotalen',
       'evidence':'dagaggregaties en seizoenstrends onderzocht; uurdoel behoudt operationele resolutie',
       'available_features':'kalender en voorafgaande dagtotalen','selected':False},
      {'target':'duration','useful':'Verwachting bij vertrek',
       'type':'regression at departure','target_present':True,'unit':'minuten',
       'modeling_rows_before_features':int(results['NYC']['duration_statistics'].rides.iloc[0]),
       'data_quality':'Alleen positieve identificeerbare fysieke duur en vóór fitmoment beschikbare eindlabels. Ambigue moderne najaarsuren uitgesloten; extreme open ritten en historische onbekende categorieën beperken de interpretatie.',
       'leakage_risk':'eindstation, eindtijd en routeafstand achteraf zijn verboden',
       'deployment':'starttijd, startstation en bekende gebruiker/fietscategorie',
       'evidence':f"casual-member dagmedianenverschil {hypotheses[2]['effect_size']:.4f} minuten; groepsverschil bewijst geen individuele voorspelbaarheid",
       'available_features':'startstation, starttijd, bekende categorieën',
       'limitations':'eindbestemming ontbreekt bij vertrek; zware staart; historische schema- en stationwijzigingen',
       'selected':False}]
    atomic_json(REPORTS/'target_comparison.json',candidates)
    if not supported:
        raise RuntimeError('Vooraf vastgelegde uurvraag-gate niet gehaald; geen doel of modelresultaten verzinnen. Onderzoek alternatieven expliciet.')
    atomic_json(REPORTS/'target_decision.json',{'chosen_prediction_problem':'NYC geregistreerde vertrekken per uurvak, dagvooruit om 00:00',
          'target':'hourly_demand','created_at':now(),'dataset_fingerprint':manifest()['dataset_fingerprint'],
          'reason':'Sterke wekelijkse associatie op ontwikkelingsdata plus nuttige uurresolutie; echte voorspelbaarheid volgt uit latere validatie.',
          'contract':CONTRACTS[0],'gate':'Holm-p<.05 en onderste 95%-CI weekcorrelatie>.3',
          'no_test_access':True,'eda_seconds':time.perf_counter()-start})
    print('Doelkeuze vastgelegd op ontwikkelingsdata.',flush=True)
