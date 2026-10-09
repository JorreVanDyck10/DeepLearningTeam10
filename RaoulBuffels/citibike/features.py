"""Alle kenmerken voor dag D zijn beschikbaar op D om 00:00, New York-tijd."""
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, TimeSeriesSplit
from .datasets import require_nyc

FEATURES=['hour_sin','hour_cos','weekday_sin','weekday_cos','year_sin','year_cos',
          'is_weekend','clock_hours','lag_1d','lag_2d','lag_7d','lag_14d',
          'same_hour_mean_7d','previous_day_total']
CALENDAR=FEATURES[:8]


def validate_history(frame: pd.DataFrame, future_day: bool=False) -> pd.DataFrame:
    require_nyc(frame)
    frame=frame[['timestamp','rides','area']].copy()
    frame['timestamp']=pd.to_datetime(frame.timestamp,errors='raise')
    if not pd.api.types.is_datetime64_any_dtype(frame.timestamp):
        raise ValueError('Timestamps moeten één consistente lokale klokconventie hebben')
    if frame.timestamp.dt.tz is not None:
        raise ValueError('Gebruik offsetloze lokale kloklabels in America/New_York')
    frame=frame.sort_values('timestamp').reset_index(drop=True)
    if frame.empty or frame.timestamp.duplicated().any():
        raise ValueError('Lege historie of dubbele kloklabels')
    if frame.timestamp.iloc[0].hour!=0 or frame.timestamp.iloc[-1].hour!=23:
        raise ValueError('Veertien volledige dagen met 24 kloklabels vereist')
    expected=pd.date_range(frame.timestamp.iloc[0],frame.timestamp.iloc[-1],freq='h')
    if not pd.DatetimeIndex(frame.timestamp).equals(expected):
        raise ValueError('Historie mist uren; onbekende tellingen niet met nul invullen')
    frame['rides']=pd.to_numeric(frame.rides,errors='raise')
    checked=frame.rides.iloc[:-24] if future_day else frame.rides
    if checked.isna().any() or not np.isfinite(checked).all() or (checked<0).any():
        raise ValueError('Historie moet eindige niet-negatieve tellingen bevatten')
    if future_day and not frame.rides.iloc[-24:].isna().all():
        raise ValueError('Alleen de voorspeldag mag ontbrekende tellingen hebben')
    grid=pd.DatetimeIndex(frame.timestamp)
    nonexistent=grid.tz_localize('America/New_York',ambiguous=True,nonexistent='NaT').isna()
    historic_nonexistent=nonexistent[:-24] if future_day else nonexistent
    if checked.loc[historic_nonexistent].sum()!=0:
        raise ValueError('Niet-bestaand voorjaaruur moet nul zijn')
    return frame


def make_features(hourly: pd.DataFrame, future_day: bool=False) -> pd.DataFrame:
    frame=validate_history(hourly,future_day)
    times=pd.DatetimeIndex(frame.timestamp)
    daily=pd.DataFrame(frame.rides.to_numpy().reshape(-1,24),index=times[::24].normalize())
    result=frame.copy();result['day']=times.normalize()
    for label,values,period in [('hour',times.hour,24),('weekday',times.dayofweek,7),('year',times.dayofyear-1,365.2425)]:
        result[label+'_sin']=np.sin(2*np.pi*values/period)
        result[label+'_cos']=np.cos(2*np.pi*values/period)
    result['is_weekend']=(times.dayofweek>=5).astype(int)
    early=times.tz_localize('America/New_York',ambiguous=True,nonexistent='NaT')
    late=times.tz_localize('America/New_York',ambiguous=False,nonexistent='NaT')
    result['clock_hours']=np.where(early.isna(),0,np.where(early!=late,2,1))
    for lag in (1,2,7,14):
        result[f'lag_{lag}d']=daily.shift(lag).to_numpy().ravel()
    result['same_hour_mean_7d']=daily.shift(1).rolling(7,min_periods=7).mean().to_numpy().ravel()
    result['previous_day_total']=np.repeat(daily.sum(axis=1,min_count=24).shift(1).to_numpy(),24)
    result=result.dropna(subset=FEATURES).reset_index(drop=True)
    if result.empty:
        raise ValueError('Minstens veertien voorafgaande volledige dagen vereist')
    return result


def folds(frame):
    require_nyc(frame)
    if len(frame)%24 or not frame.groupby('day').size().eq(24).all():
        raise ValueError('Folds vereisen volledige dagen')
    bounds=[]
    for train,valid in TimeSeriesSplit(n_splits=5,test_size=28*24).split(frame):
        if frame.day.iloc[train[-1]]>=frame.day.iloc[valid[0]]:
            raise AssertionError('Fold bevat toekomstige trainingsdata')
        bounds.append({'train_start':int(train[0]),'train_stop':int(train[-1]+1),
                       'validation_start':int(valid[0]),'validation_stop':int(valid[-1]+1),
                       'train_last_day':str(frame.day.iloc[train[-1]].date()),
                       'validation_first_day':str(frame.day.iloc[valid[0]].date()),
                       'validation_last_day':str(frame.day.iloc[valid[-1]].date())})
    return bounds


class SavedChronologicalCV(BaseCrossValidator):
    """Exact dezelfde opgeslagen indexgrenzen voor sklearn en PyCaret."""
    def __init__(self,bounds):
        self.bounds=bounds

    def split(self,X,y=None,groups=None):
        for bound in self.bounds:
            yield (np.arange(bound['train_start'],bound['train_stop']),
                   np.arange(bound['validation_start'],bound['validation_stop']))

    def get_n_splits(self,X=None,y=None,groups=None):
        return len(self.bounds)


def leakage_audit():
    rows=[]
    for feature in FEATURES:
        available=('kalender bekend vóór 00:00 op D' if feature in CALENDAR else
                   'uitsluitend volledige dagen vóór D; rolling eerst shift(1)')
        rows.append({'feature':feature,'available_at':available,
                     'available_at_prediction':True if feature in CALENDAR else 'voorwaardelijk: externe tellingen bekend vóór D',
                     'availability_assumption':'kalender deterministisch bekend' if feature in CALENDAR else
                         'Offline backtest veronderstelt tijdige externe vertrektellingen. Publieke maandarchieven bewijzen geen realtime beschikbaarheid; publicatielatentie is niet gemodelleerd.',
                     'target_information':'historische doelwaarden toegestaan' if feature not in CALENDAR else 'geen doelwaarden',
                     'leaks':False})
    return rows
