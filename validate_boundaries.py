"""Meaningful boundary regressions using only the already-truncated research input."""
import json
from pathlib import Path
import run_audit as audit
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent
results=[]


def verify(name,ok):
    results.append(dict(check=name,passed=bool(ok)))
    assert ok,name
    print('PASS',name,flush=True)


def main():
    t=pd.read_csv(P/'trades_diagnostic.csv')
    models=json.loads((P/'inputs/selected_models_30.json').read_text(encoding='utf-8'))
    tree=audit.ast.parse((P/'inputs/ETH_KalmanMomentum_V18C3_TwoConfirm.py').read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,audit.ast.ClassDef))
    cfg={n.targets[0].id:audit.ast.literal_eval(n.value) for n in cls.body if isinstance(n,audit.ast.Assign) and isinstance(n.targets[0],audit.ast.Name) and n.targets[0].id in ['LONG','SHORT']}
    for side in audit.SIGN:
        verify('source and model parameters '+side,list(cfg[side.upper()])==next(m for m in models if m['case']=='unfiltered_'+side)['core_parameters'])
    raw=pd.read_feather(P/'inputs/ohlcv_closed_pre2024.feather')
    # Stop-touch extremes must never be used as complete held bars.
    stops=t[t.exit_time_precision=='within_15m_bar_unknown']
    verify('all 58 stop-touch bars excluded',len(stops)==58 and bool(stops.mfe_is_lower_bound.all()))
    verify('unknown intrabar exit time is missing',bool(stops.actual_exit_time.isna().all()))
    verify('no complete bars yields missing MFE',bool(t.loc[t.complete_held_bars.eq(0),'MFE_pct'].isna().all()))
    verify('complete held bars match actual exposure',bool((pd.to_datetime(t.entry_time,utc=True)+pd.to_timedelta(t.complete_held_bars*15,unit='min')==pd.to_datetime(t.exit_bar_open_time,utc=True)).all()))
    signal=pd.read_csv(P/'per_bar_conditions.csv.gz')
    verify('signal timestamps all closed before cutoff',bool((pd.to_datetime(signal.decision_time,utc=True)<=audit.END).all()))
    verify('expected signal rows',len(signal)==2*140255)
    filt=pd.read_csv(P/'signal_filter_summary.csv')
    verify('quality partition reconciles technical counts',bool((filt.technical_signal_bars==filt.quality_rejected_bars+filt.quality_passed_bars).all()))
    annual=pd.read_csv(P/'annual_performance.csv');monthly=pd.read_csv(P/'monthly_performance.csv')
    for side in ['long','short','combined']:
        total=annual[(annual.period=='2020-2023')&(annual.direction==side)].net_profit_marked.iloc[0]
        verify('monthly PNL reconciles '+side,np.isclose(monthly[monthly.direction==side].net_pnl_marked.sum(),total,atol=1e-8))
    # A historical prefix truncates a real normal-exit trade shortly after entry.
    chosen=t[t.exit_group.eq('normal')].iloc[0]
    entry=pd.Timestamp(chosen.entry_time)
    prefix=raw[raw.date<=entry].copy().reset_index(drop=True)
    f=audit.original_features(prefix);arrays=audit.get_signals(f,models)
    offset=int(np.flatnonzero(f.date>=audit.FORMAL)[0])-2
    d=f.iloc[offset:].reset_index(drop=True)
    da={s:{k:(v[offset:] if isinstance(v,np.ndarray) else v) for k,v in a.items()} for s,a in arrays.items()}
    gates={s:a['gate'] for s,a in da.items()}
    trades,q,_=audit.execute(d,models,gates,audit.SLIP)
    diag=audit.trade_diagnostics(trades,d,da)
    op=diag[diag.status.eq('censored/open')]
    verify('prefix preserves open trade rather than forced exit',len(op)==1)
    verify('open trade has no invented exit or realized outcome',bool(op[['actual_exit_time','actual_exit_price','gross_realized_return_pct','net_realized_return_pct','net_realized_pnl','giveback_pct','giveback_ratio']].isna().all().all()))
    verify('prefix mark only uses last complete close',pd.Timestamp(op.censored_mark_time.iloc[0])==prefix.date.iloc[-1]+audit.BAR)
    verify('open marked equity reconciles',np.isclose(op.censored_unrealized_net_pnl.sum()+diag.net_realized_pnl.sum(),1000*(q[-1]-1),atol=1e-8))
    # Remove that entry candle: pending entry cannot be backfilled from a future open.
    before,_,_=audit.execute(d.iloc[:-1].copy(),models,{s:g[:-1] for s,g in gates.items()},audit.SLIP)
    verify('pending entry outside prefix is not an actual trade',len(before)==len(trades)-1)
    pd.DataFrame(results).to_csv(P/'boundary_validation.csv',index=False)
    print('Boundary validations passed',len(results))


if __name__=='__main__':main()
