"""Frozen V18 C3 .001 tier30 baseline. No parameter fitting or future features.

Keep assets/ beside this file. Seed history is used strictly before the first
provided candle, so Kalman initialization matches the registered research.
"""
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy, stoploss_from_absolute
from freqtrade.persistence import Trade

ASSETS=Path(__file__).resolve().parent/'assets'

def build_features(raw):
    f=raw.copy().reset_index(drop=True)
    if not f.date.is_unique or not f.date.diff().dropna().eq(pd.Timedelta(minutes=15)).all():
        raise ValueError('V18 requires unique contiguous 15m candles; repair gaps first')
    dt=.25;R=.01
    F=np.array([[1.,dt,.5*dt**2],[0.,1.,dt],[0.,0.,1.]])
    Q=1e-7*np.array([[dt**5/20,dt**4/8,dt**3/6],[dt**4/8,dt**3/3,dt**2/2],[dt**3/6,dt**2/2,dt]])
    y=100*np.log(f.close.to_numpy());x=np.array([y[0],0.,0.]);P=np.diag([R,1.,.1]);I=np.eye(3)
    v=np.empty(len(f));a=np.empty(len(f))
    # Same Joseph-form update/order as the frozen research implementation.
    H=np.array([[1.,0.,0.]])
    for i,obs in enumerate(y):
        if i:x=F@x;P=F@P@F.T+Q
        innovation=obs-(H@x)[0];innovation_var=(H@P@H.T)[0,0]+R
        K=(P@H.T/innovation_var).ravel();x=x+K*innovation
        residual=I-np.outer(K,H.ravel());P=residual@P@residual.T+np.outer(K,K)*R
        v[i],a[i]=x[1],x[2]
    f['slope']=v;f['model_accel']=a
    for src,dst in [('slope','slope_scaled'),('model_accel','model_accel_scaled')]:
        f[dst]=f[src]/f[src].shift(1).rolling(96,min_periods=96).std().replace(0,np.nan)
    f['momentum4']=100*(f.close/f.close.shift(4)-1)
    xx=np.arange(8)*dt;w=(xx-xx.mean())/np.sum((xx-xx.mean())**2)
    trend=np.full(len(f),np.nan);trend[7:]=np.convolve(v,w[::-1],mode='valid');f['trend8']=trend
    h=f.set_index('date').resample('4h',label='right',closed='left').agg(close=('close','last'),bars=('close','size'))
    h=h[h.bars.eq(16)]
    h['fast']=h.close.ewm(span=20,adjust=False,min_periods=60).mean()
    h['slow']=h.close.ewm(span=60,adjust=False,min_periods=60).mean()
    f['decision']=f.date+pd.Timedelta(minutes=15)
    return pd.merge_asof(f,h.reset_index().rename(columns={'date':'available_at'})[['available_at','fast','slow']],left_on='decision',right_on='available_at',direction='backward')

class ETH_KalmanMomentum_V18C3_TwoConfirm(IStrategy):
    INTERFACE_VERSION=3
    timeframe='15m'
    can_short=True
    process_only_new_candles=True
    startup_candle_count=1000
    minimal_roi={}
    stoploss=-.0814
    use_custom_stoploss=True
    trailing_stop=False
    use_exit_signal=True
    exit_profit_only=False
    position_adjustment_enable=False
    order_types={'entry':'market','exit':'market','stoploss':'market','stoploss_on_exchange':True,'stoploss_on_exchange_interval':60}
    order_time_in_force={'entry':'GTC','exit':'GTC'}
    LONG=(.69,.21,.340,2,0.,1.49,3.87)
    SHORT=(.76,.49,.505,2,.38,1.43,4.07)

    def __init__(self,config):
        super().__init__(config)
        self._models=json.loads((ASSETS/'selected_models_30.json').read_text(encoding='utf-8'))
        self._seed=pd.read_feather(ASSETS/'seed_ohlcv.feather')
        self._runtime_history=ASSETS/'runtime_ohlcv.feather'
        self._mode=str(getattr(config.get('runmode'),'value',config.get('runmode','')))
        if self._mode in ('live','dry_run') and self._runtime_history.exists():
            saved=pd.read_feather(self._runtime_history)
            self._seed=pd.concat([self._seed,saved[saved.date>self._seed.date.iloc[-1]]],ignore_index=True)
        self._started_at=None

    def bot_loop_start(self,current_time:datetime,**kwargs):
        if self._started_at is None:
            self._started_at=pd.Timestamp(current_time).floor('15min')
            if self._mode=='backtest':self._started_at-=pd.Timedelta(minutes=15)

    def leverage(self,pair,current_time,current_rate,proposed_leverage,max_leverage,entry_tag,side,**kwargs):
        if max_leverage<2:raise ValueError('Frozen V18 requires 2x leverage')
        return 2.

    def custom_stake_amount(self,pair,current_time,current_rate,proposed_stake,min_stake,max_stake,leverage,entry_tag,side,**kwargs):
        # Research reserves the entry fee from the 2000 notional budget.
        return min(proposed_stake/(1+.0005),max_stake)

    def populate_indicators(self,dataframe:DataFrame,metadata:dict)->DataFrame:
        if metadata['pair']!='ETH/USDT:USDT':raise ValueError('Frozen V18 is registered for ETH/USDT:USDT only')
        cols=['date','open','high','low','close','volume']
        raw=dataframe[cols].copy();raw['date']=pd.to_datetime(raw.date,utc=True)
        # Strict prefix only. Never read a seed candle at/after incoming first date.
        prefix=self._seed[self._seed.date<raw.date.iloc[0]]
        if len(prefix)==0 or prefix.date.iloc[0]!=pd.Timestamp('2020-01-01',tz='UTC'):
            raise ValueError('Missing exact Kalman historical seed from 2020-01-01')
        allraw=pd.concat([prefix[cols],raw],ignore_index=True)
        f=build_features(allraw)
        if self._mode in ('live','dry_run') and raw.date.iloc[-1]>self._seed.date.iloc[-1]:
            # Persist observed candles before Freqtrade's rolling window drops them.
            self._seed=pd.concat([self._seed,raw[raw.date>self._seed.date.iloc[-1]]],ignore_index=True)
            tmp=self._runtime_history.with_suffix('.tmp.feather')
            self._seed.to_feather(tmp);tmp.replace(self._runtime_history)
        ret=f.close.pct_change();vol=100*ret.rolling(96,min_periods=96).std().shift(1)
        for side,s in [('long',1),('short',-1)]:
            p=self.LONG if s==1 else self.SHORT
            model=next(z for z in self._models if z['case']==f'unfiltered_{side}')
            features=np.column_stack([s*f.momentum4,s*f.slope_scaled,s*f.model_accel_scaled,s*100*(f.fast-f.slow)/f.close,vol,s*(f.momentum4-f.momentum4.shift(4))])
            z=np.nan_to_num(np.clip((features-np.array(model['mean']))/np.array(model['scale']),-5,5),nan=0.,posinf=5.,neginf=-5.)
            f[f'quality_{side}']=z@np.array(model['coef'])+model['intercept']
            f[f'condition_{side}']=(s*f.momentum4>p[0])&(s*f.slope_scaled>p[1])&(s*f.model_accel_scaled>p[2])&(s*f.trend8>0)&(f[f'quality_{side}']>=model['threshold'])
        f=f.iloc[len(prefix):].reset_index(drop=True)
        out=dataframe.copy()
        for c in f.columns:
            if c not in cols:out[c]=f[c].to_numpy()
        return out

    def populate_entry_trend(self,dataframe:DataFrame,metadata:dict)->DataFrame:
        df=dataframe
        # Month resets follow execution-bar UTC month, including pending orders.
        month=df.date.dt.strftime('%Y-%m')
        same_month=month.eq(month.shift(1))&month.eq((df.date+pd.Timedelta(minutes=15)).dt.strftime('%Y-%m'))
        for side in ['long','short']:
            cond=df[f'condition_{side}'].fillna(False)
            df[f'enter_{side}']=(cond&cond.shift(1,fill_value=False)&same_month).astype(int)
        df['enter_tag']='V18_C3_001_30_baseline_2confirm'
        return df

    def populate_exit_trend(self,dataframe:DataFrame,metadata:dict)->DataFrame:
        dataframe['exit_long']=0;dataframe['exit_short']=0
        return dataframe

    def confirm_trade_entry(self,pair,order_type,amount,rate,time_in_force,current_time,entry_tag,side,**kwargs):
        now=pd.Timestamp(current_time)
        if self._started_at is not None and now<self._started_at+pd.Timedelta(minutes=30):return False
        closed=Trade.get_trades_proxy(pair=pair,is_open=False)
        if closed:
            latest=max(pd.Timestamp(t.close_date_utc) for t in closed)
            # Sold candle cannot start confirmations; two subsequent closes needed.
            if now<latest.floor('15min')+pd.Timedelta(minutes=45):return False
        return True

    def custom_stoploss(self,pair,trade,current_time,current_rate,current_profit,after_fill=False,**kwargs):
        pct=(self.SHORT if trade.is_short else self.LONG)[6]/100
        level=trade.open_rate*(1+pct if trade.is_short else 1-pct)
        return stoploss_from_absolute(level,current_rate,is_short=trade.is_short,leverage=trade.leverage)

    def custom_exit(self,pair,trade,current_time,current_rate,current_profit,**kwargs):
        df,_=self.dp.get_analyzed_dataframe(pair,self.timeframe)
        now=pd.Timestamp(current_time).floor('15min')
        # No incomplete current candle; use only confirmed closes since entry.
        bars=df[(df.date>=pd.Timestamp(trade.open_date_utc))&(df.date<now)]
        if len(bars)<2:return None
        s=-1 if trade.is_short else 1;p=self.SHORT if trade.is_short else self.LONG
        level=trade.open_rate*(1-s*p[6]/100)
        if s*(current_rate-level)<=0:return None # native gap stop has priority
        arming=df[(df.date>=pd.Timestamp(trade.open_date_utc)-pd.Timedelta(minutes=15))&(df.date<now)]
        armed=bool(trade.get_custom_data(key='v18_slope_armed',default=False))
        if not armed and (s*arming.slope>0).any():
            armed=True
            trade.set_custom_data(key='v18_slope_armed',value=True)
        tail=bars.iloc[-2:]
        normal=(s*tail.slope<0)
        if p[4]!=0:normal=normal&(s*tail.momentum4 < -p[4])
        if armed and normal.all():
            return 'core_sign' if p[4]==0 else 'core_momentum'
        early=(100*s*(tail.close/trade.open_rate-1)<=-p[5])&(s*tail.momentum4<0)&(s*tail.model_accel_scaled<0)
        if early.all():return 'dual_loss_any'
        return None
