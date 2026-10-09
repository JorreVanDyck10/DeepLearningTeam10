"""Kleine, vraaggerichte grafieken en interpretaties uit echte uitvoer."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from .config import REPORTS,DATA


def read(name):
    return pd.read_csv(REPORTS/name,dtype={'start_station_id':'string','end_station_id':'string','station_id':'string'})


def figure(kind,area='NYC'):
    hourly=pd.read_parquet(DATA/f'{area}_hourly_development.parquet')
    recent=hourly[hourly.timestamp>='2023-01-01'].copy()
    recent['hour']=recent.timestamp.dt.hour;recent['weekday']=recent.timestamp.dt.dayofweek
    if kind=='temporal':
        monthly=read(f'{area}_monthly.csv');monthly['period']=pd.to_datetime(monthly.period)
        fig,axes=plt.subplots(2,1,figsize=(11,7),constrained_layout=True)
        monthly.set_index('period').rides.plot(ax=axes[0],color='#12796b')
        axes[0].set(title=f'{area}: hoe veranderde geregistreerd gebruik?',xlabel='Maand',ylabel='Ritten per maand')
        daily=recent.set_index('timestamp').rides.resample('D').sum()
        daily.plot(ax=axes[1],alpha=.65,color='#12796b')
        daily.rolling(28).mean().plot(ax=axes[1],color='#db6c35',label='28-daags gemiddelde')
        axes[1].set(title='Recente ontwikkelingsjaren: dagvolume en trend',xlabel='Datum',ylabel='Ritten per dag');axes[1].legend()
        text=f"De ontwikkelingsdata bevatten {int(monthly.rides.sum()):,} bruikbare vertrekken. Het recente daggemiddelde is {daily.mean():,.0f} ritten en het maximum {daily.max():,.0f}. Groei kan samenhangen met systeemuitbreiding; dit bewijst geen oorzaak. De onvolledige eindjaren zijn niet rechtstreeks met volledige jaren vergelijkbaar."
    elif kind=='hour_weekday':
        means=recent.groupby(['weekday','hour']).rides.mean().unstack()
        fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
        image=axes[0].imshow(means.to_numpy(),aspect='auto',cmap='YlGnBu')
        axes[0].set(title='Wanneer zijn er meer vertrekken?',xlabel='Lokaal uur',ylabel='Weekdag',yticks=range(7),yticklabels=['ma','di','wo','do','vr','za','zo']);fig.colorbar(image,ax=axes[0],label='Gemiddelde ritten per uur')
        recent['day_type']=np.where(recent.weekday<5,'werkdag','weekend')
        pattern=recent.groupby(['hour','day_type']).rides.mean().unstack()
        pattern.plot(ax=axes[1]);axes[1].set(title='Werkdagen tegenover weekend',xlabel='Lokaal uur',ylabel='Gemiddelde ritten per uur',xticks=range(0,24,3))
        axes[1].legend(title='Daggroep')
        text=f"Het gemiddelde uurvolume varieert van {means.min().min():,.1f} tot {means.max().max():,.1f} ritten over de weekdag/uur-combinaties. Dit beschrijft covariatie; de geblokte hypothesetoets onderzoekt een specifiek verschil. Alleen ontwikkeling vanaf 2023 is gebruikt."
    elif kind=='duration':
        sample=pd.read_parquet(DATA/f'{area}_eda_sample.parquet');duration=sample.duration_minutes
        fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
        bins=np.geomspace(max(duration.min(),np.finfo(float).tiny),duration.max(),60)
        axes[0].hist(duration,bins=bins,color='#12796b');axes[0].set_xscale('log')
        axes[0].set(title='Hoe scheef is de ritduur?',xlabel='Ritduur in minuten (logaritmisch)',ylabel='Ritten in steekproef')
        groups=[sample.loc[sample.member_casual==g,'duration_minutes'].to_numpy() for g in ['member','casual']]
        axes[1].boxplot(groups,labels=['member','casual'],showfliers=False)
        axes[1].set(title='Ritduur per gebruikerstype',xlabel='Gebruikerstype',ylabel='Minuten')
        q=duration.quantile([.25,.5,.75])
        text=f"De deterministische steekproef bevat {len(sample):,} ritten. Mediaan: {q.loc[.5]:.2f} minuten; IQR: {q.loc[.75]-q.loc[.25]:.2f} minuten; maximum: {duration.max():,.1f} minuten. Boxplot-uitbijters worden alleen verborgen in deze weergave, niet verwijderd uit de data. Exacte groepsaantallen en benaderde volledige-data-kwantielen staan in de aggregatietabellen."
    elif kind=='user_bike':
        data=read(f'{area}_user_bike.csv')
        counts=data.pivot(index='rideable_type',columns='member_casual',values='rides').fillna(0)
        fig,axis=plt.subplots(figsize=(10,4),constrained_layout=True);counts.plot.bar(ax=axis,stacked=True)
        axis.set(title='Welke fiets- en gebruikerstypen komen voor?',xlabel='Fietstype',ylabel='Aantal geregistreerde ritten');axis.tick_params(axis='x',rotation=20)
        unknown=int(data.loc[data.rideable_type=='unknown','rides'].sum())
        text=f"{unknown:,} ritten hebben geen bekend fietstype, onder meer door het historische schema. Unknown wordt daarom niet als een echte fietscategorie geïnterpreteerd. Veranderingen in categorieën kunnen zowel gedrag als registratie weerspiegelen."
    elif kind=='stations':
        data=read(f'{area}_start_stations.csv').head(12)
        fig,axis=plt.subplots(figsize=(11,5),constrained_layout=True)
        axis.barh(data.example_name.astype(str).iloc[::-1],data.rides.iloc[::-1],color='#12796b')
        axis.set(title='Welke vertrekstations hebben de meeste geregistreerde ritten?',xlabel='Vertrekken in ontwikkelingsperiode',ylabel='Voorbeeldnaam van station-ID')
        text=f"De drukste station-ID in deze aggregatie heeft {int(data.rides.iloc[0]):,} vertrekken en {int(data.name_variants.iloc[0])} verschillende naamlabels. Dit is een historisch totaal; stations met meer actieve jaren hebben meer gelegenheid om ritten te verzamelen. ID's uit verschillende schema's worden niet zonder bewijs samengevoegd."
    elif kind=='routes':
        data=read(f'{area}_routes.csv').head(10)
        labels=data.start_station_id.astype(str)+' → '+data.end_station_id.astype(str)
        fig,axis=plt.subplots(figsize=(11,5),constrained_layout=True);axis.barh(labels.iloc[::-1],data.rides.iloc[::-1],color='#12796b')
        axis.set(title='Welke geregistreerde routes komen het vaakst voor?',xlabel='Ritten in ontwikkelingsperiode',ylabel='Start-ID → eind-ID')
        text=f"De meest voorkomende geregistreerde route heeft {int(data.rides.iloc[0]):,} ritten. Routevolumes zeggen niets over onvervulde vraag en zijn geen geschikte invoer bij vertrek als de bestemming nog onbekend is."
    elif kind=='spatial':
        sample=pd.read_parquet(DATA/f'{area}_eda_sample.parquet')
        valid=sample[sample.start_lat.between(40,41.5)&sample.start_lng.between(-75,-73)]
        fig,axis=plt.subplots(figsize=(8,6),constrained_layout=True)
        image=axis.hexbin(valid.start_lng,valid.start_lat,gridsize=45,mincnt=1,cmap='YlGnBu',bins='log')
        fig.colorbar(image,ax=axis,label='Ritten per hexagon (logaritmische kleur)')
        axis.set(title=f'{area}: waar liggen vertrekpunten in de steekproef?',xlabel='Lengtegraad, graden',ylabel='Breedtegraad, graden')
        text=f"Deze geaggregeerde kaart toont {len(valid):,} steekproefritten binnen het getoonde geografische venster. Buitenliggende punten zijn alleen buiten beeld gehouden. Geldige wereldcoördinaten zijn nog geen bewijs van een betrouwbaar NYC-station."
    elif kind=='acf':
        fig,axis=plt.subplots(figsize=(11,4),constrained_layout=True)
        values=recent.rides.iloc[-365*24:]
        plot_acf(values,lags=336,ax=axis,fft=True)
        axis.set(title='Herhaalt uurgebruik zich op dag- en weekafstand?',xlabel='Vertraging in lokale klokvakken',ylabel='Autocorrelatie')
        text=f"Op deze recente ontwikkelingsperiode is de autocorrelatie bij 24 uur {values.autocorr(24):.3f} en bij 168 uur {values.autocorr(168):.3f}. Trend en seizoenen kunnen deze correlaties verhogen; dit is geen zelfstandig bewijs van voorspelprestatie. De latere modellen moeten de baseline op ongeziene perioden verslaan."
    elif kind=='comparison':
        data=read('model_comparison.csv').sort_values('RMSE',ascending=False)
        fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
        axes[0].barh(data.name,data.RMSE,color='#12796b');axes[0].set(title='Modelkeuze op mei–juni 2026',xlabel='Validatie-RMSE, ritten/uur')
        axes[1].scatter(data.train_RMSE,data.RMSE,color='#12796b')
        for _,row in data.iterrows():axes[1].annotate(row['name'],(row.train_RMSE,row.RMSE),fontsize=7)
        axes[1].set(title='Training tegenover validatie',xlabel='Train-RMSE, ritten/uur',ylabel='Validatie-RMSE, ritten/uur')
        best=data.sort_values('RMSE').iloc[0]
        text=f"De laagste validatie-RMSE is {best.RMSE:.2f} bij {best['name']}; de train-RMSE is {best.train_RMSE:.2f}, een verschil van {best.RMSE-best.train_RMSE:.2f} ritten/uur. Een hogere validatiefout kan op overfit of veranderde omstandigheden wijzen. De vastgelegde selectieregel mag binnen 1% het eenvoudigere model kiezen. Een lage trainingsfout alleen rechtvaardigt geen modelkeuze."
    elif kind.startswith('residual'):
        stage='test' if kind.endswith('test') else 'validation'
        data=read('test_predictions.csv' if stage=='test' else 'validation_residuals.csv');data.timestamp=pd.to_datetime(data.timestamp)
        fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
        axes[0,0].hist(data.residual,bins=50,color='#12796b');axes[0,0].set(title='Verdeling van fouten',xlabel='Werkelijk − voorspeld, ritten',ylabel='Uurvakken')
        axes[0,1].plot(data.timestamp,data.residual,lw=.6);axes[0,1].axhline(0,color='black',lw=.5)
        axes[0,1].set(title='Fouten over tijd',xlabel='Datum',ylabel='Residueel, ritten')
        axes[1,0].scatter(data.predicted_rides,data.residual,s=3,alpha=.3);axes[1,0].set(title='Fout versus voorspelling',xlabel='Voorspelde ritten',ylabel='Residueel, ritten')
        data.groupby('hour').absolute_error.mean().plot.bar(ax=axes[1,1],color='#12796b');axes[1,1].set(title='Fout per uur',xlabel='Lokaal uur',ylabel='MAE, ritten/uur')
        fig.suptitle(f'Residuanalyse — {stage}')
        means=data.groupby('hour').absolute_error.mean();worst=means.idxmax()
        text=f"Gemiddelde bias: {data.residual.mean():.2f} ritten/uur. Het uur met de hoogste MAE is {worst}:00 ({means.loc[worst]:.2f} ritten/uur). Overblijvende patronen wijzen op beperkingen. Eindtestdiagnostiek wordt uitsluitend achteraf beschreven en leidt niet tot nieuwe tuning onder dezelfde eindtestclaim."
    elif kind=='importance':
        data=read('permutation_importance.csv').sort_values('importance_RMSE')
        fig,axis=plt.subplots(figsize=(10,5),constrained_layout=True)
        axis.barh(data.feature,data.importance_RMSE,xerr=data.sd,color='#12796b')
        axis.set(title='Wat helpt het model op de validatieperiode?',xlabel='RMSE-toename na permutatie, ritten/uur',ylabel='Feature')
        text=f"De sterkste gemeten permutation importance hoort bij {data.iloc[-1].feature}: {data.iloc[-1].importance_RMSE:.2f} ritten/uur RMSE-toename. Afhankelijke features kunnen elkaars belang maskeren; onafhankelijke permutatie kan onrealistische combinaties maken. Dit toont geen causaliteit."
    elif kind=='test_forecast':
        data=read('test_predictions.csv');data.timestamp=pd.to_datetime(data.timestamp)
        fig,axis=plt.subplots(figsize=(12,4),constrained_layout=True)
        data.iloc[:7*24].set_index('timestamp')[['rides','predicted_rides','baseline_rides']].plot(ax=axis)
        axis.set(title='Eerste eindtestweek: werkelijkheid, model en baseline',xlabel='Datum',ylabel='Ritten per klokvak')
        result=json.loads((REPORTS/'final_evaluation.json').read_text(encoding='utf-8'))
        text=f"Over de volledige eindtest is de model-RMSE {result['metrics']['RMSE']:.2f}, tegenover {result['baseline_metrics']['RMSE']:.2f} voor het vorige weekuur. Verbetering: {result['RMSE_improvement_percent']:.2f}%. Iedere dag gebruikt alleen eerder gemeten dagen."
    else:
        raise ValueError(kind)
    return fig,text


def show(kind,area='NYC'):
    from IPython.display import display,Markdown
    fig,text=figure(kind,area)
    display(fig);plt.close(fig);display(Markdown(text))
