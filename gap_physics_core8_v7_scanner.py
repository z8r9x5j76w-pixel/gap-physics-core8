"""Fixed Core8 paper-monitor scanner. No passwords, secrets, orders or optimization."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import pandas_market_calendars as mcal
@dataclass(frozen=True)
class GapConfig:
    ticker: str
    label: str
    gap_atr_max: float
    vix_min: Optional[float]
    daily_trend: str
    weekly_trend: str
    min_volume_z: Optional[float]
    tp_kind: str
    tp_atr_mult: Optional[float]
    sl_atr_mult: float
    max_hold: int
    priority: int
CORE8: List[GapConfig] = [GapConfig('TGT', 'TGT', -1.0, None, 'none', 'sma20', 0.0, 'atr', 1.5, 0.75, 2, 1), GapConfig('CAT', 'CAT', -1.0, None, 'sma100', 'none', None, 'atr', 1.0, 1.25, 5, 2), GapConfig('INTU', 'INTU', -1.0, None, 'sma100', 'none', 0.0, 'atr', 1.5, 1.5, 5, 3), GapConfig('ASML', 'ASML', -1.0, 20.0, 'sma100', 'none', None, 'atr', 1.75, 1.25, 2, 4), GapConfig('AXP', 'AXP', -1.0, None, 'sma50', 'sma20', None, 'atr', 1.0, 1.5, 3, 5), GapConfig('AMAT', 'AMAT', -1.0, 20.0, 'none', 'none', 0.0, 'atr', 1.75, 1.5, 3, 6), GapConfig('TSM', 'TSM', -1.0, 20.0, 'none', 'sma20', None, 'gapfill', None, 1.5, 2, 7), GapConfig('AMZN', 'AMZN', -1.0, None, 'sma100', 'none', None, 'atr', 1.0, 1.25, 2, 8)]

def _true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    pc = c.shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)

def _weekly_trend(df: pd.DataFrame) -> pd.DataFrame:
    wk = pd.DataFrame({'High': df['High'].resample('W-FRI').max(), 'Low': df['Low'].resample('W-FRI').min(), 'Close': df['Close'].resample('W-FRI').last()}).dropna()
    wk['W_ATR14'] = _true_range(wk['High'], wk['Low'], wk['Close']).rolling(14, min_periods=5).mean()
    wk['W_SMA20'] = wk['Close'].rolling(20, min_periods=10).mean()
    wk['W_Trend_SMA20_ATR'] = (wk['Close'] - wk['W_SMA20']) / wk['W_ATR14']
    return wk[['W_Trend_SMA20_ATR']].reindex(df.index, method='ffill')

def compute_features(price: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    df = price.copy()
    df['Prev_Close'] = df['Close'].shift(1)
    df['ATR14'] = _true_range(df['High'], df['Low'], df['Close']).rolling(14, min_periods=14).mean()
    df['Gap_ATR'] = (df['Open'] - df['Prev_Close']) / df['ATR14']
    df['Gap_Pct'] = (df['Open'] / df['Prev_Close'] - 1.0) * 100.0
    vol_ma = df['Volume'].rolling(20, min_periods=10).mean()
    df['Volume_Z'] = np.log(df['Volume'].replace(0, np.nan) / vol_ma.replace(0, np.nan))
    for n in [20, 50, 100, 200]:
        sma = df['Close'].rolling(n, min_periods=max(10, n // 2)).mean()
        df[f'Trend_SMA{n}_ATR'] = (df['Close'] - sma) / df['ATR14']
    df = df.join(_weekly_trend(df))
    df = df.join(vix, how='left')
    df['VIX_Close'] = df['VIX_Close'].ffill()
    return df.dropna(subset=['Open', 'High', 'Low', 'Close', 'ATR14', 'Gap_ATR'])
_TREND_COL = {'sma20': 'Trend_SMA20_ATR', 'sma50': 'Trend_SMA50_ATR', 'sma100': 'Trend_SMA100_ATR', 'sma200': 'Trend_SMA200_ATR'}

def check_conditions(row: pd.Series, cfg: GapConfig) -> Dict[str, Tuple[bool, str]]:
    """
    Returns dict of condition_name → (passed: bool, detail: str).
    """
    out: Dict[str, Tuple[bool, str]] = {}
    g = float(row['Gap_ATR'])
    out['Gap ATR'] = (g <= cfg.gap_atr_max, f'{g:.2f} (need ≤ {cfg.gap_atr_max:.1f})')
    vix_val = float(row.get('VIX_Close', np.nan))
    if cfg.vix_min is None:
        out['VIX'] = (True, 'off')
    else:
        out['VIX'] = (np.isfinite(vix_val) and vix_val >= cfg.vix_min, f'{vix_val:.1f} (need ≥ {cfg.vix_min:.0f})')
    if cfg.daily_trend == 'none':
        out['Daily trend'] = (True, 'off')
    else:
        col = _TREND_COL.get(cfg.daily_trend, '')
        val = float(row.get(col, np.nan))
        out['Daily trend'] = (np.isfinite(val) and val > 0, f'{cfg.daily_trend.upper()} ratio={val:.2f} (need > 0)')
    if cfg.weekly_trend == 'none':
        out['Weekly trend'] = (True, 'off')
    else:
        val = float(row.get('W_Trend_SMA20_ATR', np.nan))
        out['Weekly trend'] = (np.isfinite(val) and val > 0, f'W_SMA20 ratio={val:.2f} (need > 0)')
    if cfg.min_volume_z is None:
        out['Volume'] = (True, 'off')
    else:
        vol = float(row.get('Volume_Z', np.nan))
        out['Volume'] = (np.isfinite(vol) and vol >= cfg.min_volume_z, f'log-ratio={vol:.2f} (need ≥ {cfg.min_volume_z:.1f})')
    return out

st.set_page_config(page_title='Gap Physics · fixed Core8', page_icon='⚡',layout='wide')
st.title('Gap Physics · fixed Core8')
st.caption('Paper monitoring · unchanged eight rules · completed sessions only')
st.info('Hourly IBKR audit was positive overall, with small samples. Live signals use Yahoo and may differ. This scanner does not track your positions; skip new entries while already holding the same stock.')

@st.cache_data(ttl=900,show_spinner=False)
def prices(symbol,start):
    x=yf.download(symbol,start=start,auto_adjust=False,progress=False,threads=False)
    if x.empty:raise ValueError('No price data returned')
    if isinstance(x.columns,pd.MultiIndex):
        x.columns=x.columns.get_level_values(0)
    x.index=pd.to_datetime(x.index).tz_localize(None).normalize()
    x=x[['Open','High','Low','Close','Volume']].dropna().sort_index()
    if x.index.duplicated().any():raise ValueError('Duplicate daily prices')
    return x

def levels(entry,atr,cfg,previous_close):
    stop=entry-cfg.sl_atr_mult*atr
    target=previous_close if cfg.tp_kind=='gapfill' else entry+cfg.tp_atr_mult*atr
    return stop,target

now=pd.Timestamp.now(tz='UTC')
start=(now-pd.Timedelta(days=1000)).date().isoformat()
calendar=mcal.get_calendar('NYSE')
schedule=calendar.schedule(start_date=start,end_date=(now+pd.Timedelta(days=40)).date())
# Allow daily stock/VIX data to settle after the session ends.
completed=schedule.index[schedule.market_close+pd.Timedelta(minutes=30)<=now]
latest=completed[-1]
st.caption(f'Expected latest completed session: {latest.date()} · refresh after close + 30 minutes')
if st.button('Refresh data',type='primary'):
    st.cache_data.clear()

results=[];details={}
with st.spinner('Checking eight fixed rules…'):
    try:
        vx=prices('^VIX',start)
        if latest not in vx.index:raise ValueError(f'VIX missing {latest.date()}; wait for data update')
        vix=vx.loc[:latest,'Close'].rename('VIX_Close')
    except Exception as e:
        st.error(f'VIX unavailable: {e}');st.stop()
    for cfg in CORE8:
        try:
            x=prices(cfg.ticker,start).loc[:latest]
            if x.empty or x.index[-1]!=latest:raise ValueError('Latest completed session missing; signal withheld')
            f=compute_features(x,vix)
            r=f.iloc[-1];conds=check_conditions(r,cfg)
            sig=all(a[0] for a in conds.values())
            entry_day=schedule.index[schedule.index>latest][0]
            next_open=schedule.loc[entry_day,'market_open']
            status='NO SIGNAL'
            if sig:status='PENDING NEXT OPEN' if now<next_open else 'ENTRY WINDOW PASSED — CHECK YOUR TRADE'
            exit_day=schedule.index[schedule.index>=entry_day][cfg.max_hold-1]
            results.append({'Ticker':cfg.ticker,'Status':status,'Signal session':str(latest.date()),'Next entry session':str(entry_day.date()) if sig else '—','ATR14':round(float(r.ATR14),4),'Max hold':cfg.max_hold})
            details[cfg.ticker]=(cfg,r,conds,sig,entry_day,exit_day,next_open)
        except Exception as e:
            results.append({'Ticker':cfg.ticker,'Status':f'DATA ERROR: {e}'})
st.dataframe(pd.DataFrame(results),hide_index=True,use_container_width=True)
st.download_button('Download scan',pd.DataFrame(results).to_csv(index=False),'gap_physics_scan.csv','text/csv')
selected=st.selectbox('Rule details',[c.ticker for c in CORE8])
if selected in details:
    cfg,r,conds,sig,entry_day,exit_day,next_open=details[selected]
    st.dataframe(pd.DataFrame([{'Condition':k,'Pass':a,'Detail':b} for k,(a,b) in conds.items()]),hide_index=True)
    st.write(f'Signal ATR14: **{r.ATR14:.4f}** · stop distance **{cfg.sl_atr_mult*r.ATR14:.2f}**')
    if cfg.tp_kind=='gapfill':st.write(f'Fixed gap-fill target: **{r.Prev_Close:.2f}**; skip if actual next open is at or above it.')
    else:st.write(f'Target distance: **{cfg.tp_atr_mult*r.ATR14:.2f}** from actual entry.')
    if sig:
        st.write(f'Entry: **{entry_day.date()} regular-session open**. Time exit: **{exit_day.date()} regular-session close**, unless stop/target exits earlier. Entry is session one.')
        if now>=next_open:st.warning('The audited entry time has passed. A late entry is not the tested rule.')
        entry=st.number_input('Actual next-session opening fill price',min_value=0.0,value=0.0,step=0.01,key=f'entry_{selected}_{latest.date()}')
        if entry>0:
            stop,target=levels(entry,float(r.ATR14),cfg,float(r.Prev_Close))
            if not 0<stop<entry<target:st.error('INVALID ENTRY — skip: stop/entry/target ordering is invalid.')
            else:
                a,b=st.columns(2);a.metric('Stop',f'{stop:.2f}');b.metric('Target',f'{target:.2f}')
                st.caption('Fixed levels for this actual fill; do not recalculate from later-session ATR. Adverse stop gaps can fill below the stop.')
    else:st.write('No entry under this rule for the latest completed session.')
with st.expander('Fixed settings'):
    st.dataframe(pd.DataFrame([vars(c) for c in CORE8]),hide_index=True)
