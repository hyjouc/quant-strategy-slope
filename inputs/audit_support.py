"""Minimal frozen reference helpers; detached from exploratory research imports."""
import numpy as np
import pandas as pd
COLS=["C1","C2","C3","C4","L"]
NAMES=["C1","C2","C3","C4","exit_value","early_loss_pct","stop_pct"]
def schedules_from_table(p):
    entries, exits = {1: {}, -1: {}}, {1: {}, -1: {}}
    for r in p.itertuples():
        sign = 1 if r.side == 'long' else -1
        entries[sign][r.month] = tuple(getattr(r, key) for key in COLS)
        exits[sign][r.month] = (r.exit_kind, r.exit_value)
    return entries, exits

def schedule(d,key,long,short):
 rows=[]
 for month in d.date.dt.strftime('%Y-%m').unique():
  for side,p in [('long',long),('short',short)]:
   rows.append(dict(month=month,strategy=key,side=side,R=.01,L=8,exit_kind='momentum' if p[4]>0 else 'sign',**dict(zip(NAMES,p))))
 return pd.DataFrame(rows)

def feature(d,sign):
    ret=d.close.pct_change()
    return np.column_stack([sign*d.momentum4,sign*d.slope_scaled,sign*d.model_accel_scaled,
        sign*100*(d.fast-d.slow)/d.close,100*ret.rolling(96,min_periods=96).std().shift(1),
        sign*(d.momentum4-d.momentum4.shift(4))])
