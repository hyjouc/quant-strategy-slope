"""Frozen V18 historical mechanism audit. No fitting, optimization or post-2023 input.

Run from any directory: python path/to/run_audit.py
Dependencies: numpy, pandas, pyarrow, matplotlib (workspace .research_deps supported).
"""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import platform
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
sys.path.insert(0, str(ROOT / '.research_deps'))
sys.path.insert(0, str(OUT / 'inputs'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import reference_replay as replay
from reference_independent_zero_slip import independent
from audit_support import feature, schedule, schedules_from_table

START = pd.Timestamp('2020-01-01', tz='UTC')
FORMAL = pd.Timestamp('2020-01-10 23:45', tz='UTC')
END = pd.Timestamp('2023-12-31 23:59:59', tz='UTC')
BAR = pd.Timedelta(minutes=15)
FEE, SLIP, MARGIN, LEV = .0005, .0002, 1000., 2.
BASE = ROOT / 'freqtrade_v18_baseline'
SIGN = {'long': 1, 'short': -1}
CHECKS = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(name, ok):
    CHECKS.append({'check': name, 'passed': bool(ok)})
    if not ok:
        raise RuntimeError('STOP: ' + name)
    print('PASS', name, flush=True)


def save(df, name):
    df.to_csv(OUT / name, index=False, encoding='utf-8-sig')


def original_features(raw):
    path = OUT / 'inputs/ETH_KalmanMomentum_V18C3_TwoConfirm.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_features')
    ns = {'np': np, 'pd': pd}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), ns)
    return ns['build_features'](raw)


def get_signals(f, models):
    arrays = {}
    for side, s in SIGN.items():
        m = next(z for z in models if z['case'] == 'unfiltered_' + side)
        x = feature(f, s)
        z = np.nan_to_num(np.clip((x - np.array(m['mean'])) / np.array(m['scale']), -5, 5), nan=0., posinf=5., neginf=-5.)
        score = z @ np.array(m['coef']) + m['intercept']
        p = m['core_parameters']
        bits = np.column_stack([s*f.momentum4 > p[0], s*f.slope_scaled > p[1], s*f.model_accel_scaled > p[2], s*f.trend8 > 0])
        tech = bits.all(axis=1)
        gate = score >= m['threshold']
        arrays[s] = dict(bits=bits, tech=tech, gate=gate, full=tech & gate, score=score, model=m)
    return arrays


def execute(d, models, gates, slip):
    cfg = {s: next(z for z in models if z['case'] == 'unfiltered_' + side)['core_parameters'] for side,s in SIGN.items()}
    plan = schedule(d, 'baseline', cfg[1], cfg[-1])
    es, xs = schedules_from_table(plan)
    risks = {s: {z.month: (z.early_loss_pct, z.stop_pct/100) for z in plan[plan.side.eq(side)].itertuples()} for side,s in SIGN.items()}
    replay.SLIP = slip
    t, q = replay.run(d, es, xs, None, early_exit=True, entry_gate=gates, fixed_account_ratio=10, risk_schedule=risks)
    return t, q[:,1], plan


def first_idx(mask, lo, hi):
    z = np.flatnonzero(np.asarray(mask)[lo:hi])
    return lo + int(z[0]) if len(z) else None


def trade_diagnostics(t, f, arrays, exit_kind_override=None):
    records = []
    last_exit = -1
    for tid, r in enumerate(t.itertuples(), 1):
        s = SIGN[r.side]; a = arrays[s]; p = a['model']['core_parameters']
        e, x = int(r.entry_i), int(r.exit_i)
        opened = r.status != 'closed'
        # A stop-touch candle is only partially held: its extremes have unknown ordering.
        stop_touch = r.exit_time_precision == 'within_15m_bar_unknown'
        end = len(f) if opened else x
        held = f.iloc[e:end]
        ep = r.entry_price
        if len(held):
            fav = 100*s*((held.high if s==1 else held.low).to_numpy()/ep-1)
            adv = 100*s*((held.low if s==1 else held.high).to_numpy()/ep-1)
            mfe = max(0., float(fav.max())); mae = min(0., float(adv.min()))
            peak = e + int(np.argmax(fav)) if fav.max()>0 else None
        else:
            mfe = mae = np.nan; peak = None
        gross = 100*s*(r.exit_price/ep-1) if not opened else np.nan
        netp = 1000*r.simple_2x_pnl_initial if not opened else np.nan
        # Trace backwards only within this available flat/technical episode.
        month = f.date.iloc[e].strftime('%Y-%m')
        lo = e-2
        while lo > max(last_exit+1, 0) and f.date.iloc[lo-1].strftime('%Y-%m') == month and a['tech'][lo-1]:
            lo -= 1
        tech_i = lo
        full_i = first_idx(a['full'], lo, e)
        qual_i = first_idx(a['gate'], lo, e)
        first_conf, second_conf = e-2, e-1
        check(f'trade_{tid}_two_confirm', a['full'][first_conf] and a['full'][second_conf] and f.date.iloc[first_conf].strftime('%Y-%m') == month)
        def tm(i):
            return pd.NaT if i is None else f.decision.iloc[i]
        dentry = 100*s*(ep/f.close.iloc[full_i]-1)
        confirm_cost = 100*s*(f.close.iloc[second_conf]/f.close.iloc[first_conf]-1)
        next_cost = 100*s*(ep/f.close.iloc[second_conf]-1)
        # Normal arming includes the last completed pre-entry candle, as frozen custom_exit.
        arm_i = first_idx(s*f.slope.to_numpy()>0, max(e-1,0), end)
        rev_i = first_idx(s*f.slope.to_numpy()<0, e, end)
        normal = (s*f.slope.to_numpy()<0)
        exit_kind = exit_kind_override if (exit_kind_override and r.side == 'short') else ('momentum' if p[4] != 0 else 'sign')
        if exit_kind == 'momentum':
            normal &= s*f.momentum4.to_numpy() < -p[4]
        full_exit_i = first_idx(normal, max(e,arm_i if arm_i is not None else len(f)), end)
        pairs = normal & np.r_[False, normal[:-1]]
        second_exit_i = first_idx(pairs, max(e+1,arm_i if arm_i is not None else len(f)), end)
        exit_time = pd.NaT if opened else f.date.iloc[x]
        # Native stop-touch has no exact time within its 15m candle.
        exact_exit = pd.NaT if stop_touch or opened else exit_time
        mark_time = f.decision.iloc[-1] if opened else pd.NaT
        hold = ((f.decision.iloc[-1] if opened else exit_time)-f.date.iloc[e]).total_seconds()/3600
        rec = dict(trade_id=tid,direction=r.side,status='censored/open' if opened else 'closed',entry_year=f.date.iloc[e].year,
            first_technical_signal_time=tm(tech_i),first_quality_pass_time=tm(qual_i),first_full_signal_time=tm(full_i),
            first_confirmation_time=tm(first_conf),second_confirmation_time=tm(second_conf),actual_entry_time=f.date.iloc[e],
            first_signal_close=f.close.iloc[full_i],confirmation_close=f.close.iloc[second_conf],actual_entry_price=ep,
            M=f.momentum4.iloc[second_conf],S=f.slope_scaled.iloc[second_conf],A=f.model_accel_scaled.iloc[second_conf],T=f.trend8.iloc[second_conf],
            quality_prediction=a['score'][second_conf],quality_threshold=a['model']['threshold'],
            entry_wait_minutes=(f.date.iloc[e]-tm(full_i)).total_seconds()/60,
            technical_to_full_minutes=(tm(full_i)-tm(tech_i)).total_seconds()/60,
            confirmation_wait_minutes=(tm(second_conf)-tm(first_conf)).total_seconds()/60,
            D_entry_pct=dentry,confirmation_cost_pct=confirm_cost,next_open_cost_pct=next_cost,
            quality_wait_price_pct=100*s*(f.close.iloc[full_i]/f.close.iloc[tech_i]-1),
            entry_time=f.date.iloc[e],entry_price=ep,normal_exit_enabled_time=tm(arm_i),first_raw_slope_reversal_time=tm(rev_i),
            first_full_exit_signal_time=tm(full_exit_i),second_exit_confirmation_time=tm(second_exit_i),actual_exit_time=exact_exit,
            exit_bar_open_time=exit_time,exit_time_precision=r.exit_time_precision,actual_exit_price=r.exit_price,exit_reason=r.exit_reason,
            holding_duration_hours=hold,holding_duration_upper_hours=hold+(.25 if stop_touch else 0),
            normal_delay_from_raw_bars=(x-rev_i-1) if not opened and rev_i is not None else np.nan,
            normal_delay_from_full_bars=(x-full_exit_i-1) if not opened and full_exit_i is not None else np.nan,
            complete_held_bars=len(held),MFE_pct=mfe,MAE_pct=mae,mfe_peak_bar_close_time=tm(peak),
            time_to_MFE_hours=(tm(peak)-f.date.iloc[e]).total_seconds()/3600 if peak is not None else np.nan,
            gross_realized_return_pct=gross,net_realized_return_pct=netp/10,net_realized_pnl=netp,
            giveback_pct=mfe-gross,giveback_ratio=(mfe-gross)/mfe if mfe>0 else np.nan,
            censored_mark_time=mark_time,censored_mark_price=f.close.iloc[-1] if opened else np.nan,
            censored_unrealized_net_pnl=1000*r.simple_2x_pnl_initial if opened else np.nan,
            mfe_is_lower_bound=bool(stop_touch),mae_excludes_stop_partial_bar=bool(stop_touch),
            exit_group='censored/open' if opened else ('normal' if r.exit_reason.startswith('core_') else 'early_loss' if r.exit_reason=='dual_loss_any' else 'fixed_stop'))
        records.append(rec)
        if not opened: last_exit=x
    return pd.DataFrame(records)


def group_sets(df, yearcol='entry_year'):
    yield '2020-2023', df
    yield '2020-2022', df[df[yearcol]<=2022]
    for y in range(2020,2024):
        yield str(y), df[df[yearcol]==y]


def delay_summary(t):
    rows=[]
    for period,g in group_sets(t):
        for side in SIGN:
            h=g[g.direction.eq(side)]
            if not len(h):continue
            w=h.entry_wait_minutes;d=h.D_entry_pct
            rows.append(dict(period=period,direction=side,n=len(h),mean_wait_minutes=w.mean(),median_wait_minutes=w.median(),p75_wait_minutes=w.quantile(.75),p90_wait_minutes=w.quantile(.9),mean_D_entry_pct=d.mean(),median_D_entry_pct=d.median(),p90_D_entry_pct=d.quantile(.9),price_worse_rate=(d>0).mean(),price_better_rate=(d<0).mean(),cost_over_0_2pct=(d>.2).mean(),cost_over_0_5pct=(d>.5).mean(),cost_over_1pct=(d>1).mean(),mean_confirmation_minutes=h.confirmation_wait_minutes.mean(),mean_confirmation_cost_pct=h.confirmation_cost_pct.mean(),median_confirmation_cost_pct=h.confirmation_cost_pct.median(),mean_next_open_cost_pct=h.next_open_cost_pct.mean(),mean_quality_wait_minutes=h.technical_to_full_minutes.mean(),quality_delayed_rate=(h.technical_to_full_minutes>0).mean(),mean_quality_price_pct=h.quality_wait_price_pct.mean()))
    return pd.DataFrame(rows)


def exit_summary(t):
    rows=[]
    closed=t[t.status.eq('closed')]
    for period,g in group_sets(closed):
        for side in SIGN:
            for kind in ['all','normal','early_loss','fixed_stop']:
                h=g[g.direction.eq(side)]
                if kind!='all': h=h[h.exit_group.eq(kind)]
                if not len(h):continue
                wins=h[h.net_realized_pnl>0]
                rows.append(dict(period=period,direction=side,exit_group=kind,n=len(h),mean_MFE_pct=h.MFE_pct.mean(),mean_MAE_pct=h.MAE_pct.mean(),mean_gross_return_pct=h.gross_realized_return_pct.mean(),mean_giveback_pct=h.giveback_pct.mean(),median_giveback_pct=h.giveback_pct.median(),p90_giveback_pct=h.giveback_pct.quantile(.9),mean_giveback_ratio=h.giveback_ratio.mean(),median_giveback_ratio=h.giveback_ratio.median(),mean_time_to_MFE_hours=h.time_to_MFE_hours.mean(),mean_holding_hours=h.holding_duration_hours.mean(),normal_raw_delay_bars=h.normal_delay_from_raw_bars.mean() if kind=='normal' else np.nan,normal_full_delay_bars=h.normal_delay_from_full_bars.mean() if kind=='normal' else np.nan,profitable_n=len(wins),profitable_giveback_over_1pp=int((wins.giveback_pct>1).sum()),profitable_giveback_ratio_over_half=int((wins.giveback_ratio>.5).sum()),stop_partial_excluded_n=int(h.mfe_is_lower_bound.sum()),long_low_contribution_n=int(((h.holding_duration_hours>24)&(h.net_realized_pnl.abs()<10)).sum())))
    return pd.DataFrame(rows)


def signal_stats(f, arrays, trades):
    rows=[];combos=[];relations=[];forward=[]
    eligible=(f.decision>=FORMAL)&(f.decision<=END)
    perbar=[]
    for side,s in SIGN.items():
        a=arrays[s];bits=a['bits'];code=bits.astype(int)@np.array([8,4,2,1])
        same=(f.date.dt.strftime('%Y-%m')==f.date.shift(1).dt.strftime('%Y-%m'))&(f.date.dt.strftime('%Y-%m')==(f.date+BAR).dt.strftime('%Y-%m'))
        confirms=a['full']&np.r_[False,a['full'][:-1]]&same.to_numpy()
        sig=pd.DataFrame(dict(bar_open_time=f.date,decision_time=f.decision,direction=side,M_pass=bits[:,0],S_pass=bits[:,1],A_pass=bits[:,2],T_pass=bits[:,3],quality_pass=a['gate'],quality_prediction=a['score'],technical_pass=a['tech'],full_pass=a['full'],two_confirm_signal=confirms,combination=code,eligible=eligible))
        perbar.append(sig)
        for period,mask in [('2020-2023',eligible),('2020-2022',eligible&(f.date.dt.year<=2022))]+[(str(y),eligible&(f.date.dt.year==y)) for y in range(2020,2024)]:
            mask=np.asarray(mask)
            aa=bits[:,2];tt=bits[:,3];ms=bits[:,0]&bits[:,1]&mask
            n=int(mask.sum());tech=a['tech']&mask
            rows.append(dict(period=period,direction=side,eligible_bars=n,technical_signal_bars=int(tech.sum()),quality_rejected_bars=int((tech&~a['gate']).sum()),quality_passed_bars=int((tech&a['gate']).sum()),completed_confirmation_signal_bars=int((confirms&mask).sum()),actual_entries=int(((trades.direction==side)&(trades.entry_year.isin([int(period)] if len(period)==4 else range(2020,2023) if period=='2020-2022' else range(2020,2024)))).sum())))
            av=s*f.model_accel_scaled;tv=s*f.trend8
            relations.append(dict(period=period,direction=side,n=n,A_T_pearson=av[mask].corr(tv[mask]),A_T_spearman=av[mask].corr(tv[mask],method='spearman'),A_pass_T_fail_rate=float((aa&~tt&mask).sum()/n),T_pass_A_fail_rate=float((tt&~aa&mask).sum()/n),both_pass_rate=float((aa&tt&mask).sum()/n),pass_indicator_phi=pd.Series(aa[mask].astype(float)).corr(pd.Series(tt[mask].astype(float))),MS_eligible_n=int(ms.sum()),MS_A_only_rate=float((aa&~tt&ms).sum()/ms.sum()) if ms.sum() else np.nan,MS_T_only_rate=float((tt&~aa&ms).sum()/ms.sum()) if ms.sum() else np.nan))
            for c in range(16):
                cm=mask&(code==c)
                combos.append(dict(period=period,direction=side,combination=format(c,'04b'),n=int(cm.sum()),quality_passed=int((cm&a['gate']).sum())))
            for hr in [1,2,4,8]:
                ret=100*s*(f.close.shift(-4*hr)/f.close-1)
                valid=mask&ret.notna().to_numpy()&((f.decision+pd.Timedelta(hours=hr))<=END).to_numpy()
                for name,subset in [('technical_quality_pass',a['tech']&a['gate']),('technical_quality_reject',a['tech']&~a['gate'])]+[(format(c,'04b'),code==c) for c in range(16)]:
                    v=ret[valid&subset]
                    forward.append(dict(period=period,direction=side,signal_group=name,horizon_hours=hr,n=len(v),mean_directional_pct=v.mean(),median_directional_pct=v.median(),positive_rate=(v>0).mean() if len(v) else np.nan,label='overlapping_signal_level_not_strategy_profit'))
    pd.concat(perbar,ignore_index=True).to_csv(OUT/'per_bar_conditions.csv.gz',index=False,compression='gzip')
    save(pd.DataFrame(combos),'condition_combinations.csv')
    save(pd.DataFrame(relations),'A_T_relationship.csv')
    save(pd.DataFrame(forward),'signal_forward_returns.csv')
    return pd.DataFrame(rows)


def equity_paths(f, trades):
    pnl=np.zeros((len(f),2))
    for r in trades.itertuples():
        j=0 if r.direction=='long' else 1;s=SIGN[r.direction]
        e=int(f.date.searchsorted(r.entry_time))
        x=len(f) if r.status!='closed' else int(f.date.searchsorted(r.exit_bar_open_time))
        units=2000/(r.entry_price*(1+FEE))
        pnl[e:x,j]+=s*units*(f.close.iloc[e:x].to_numpy()-r.entry_price)-units*r.entry_price*FEE
        if r.status=='closed':pnl[x:,j]+=r.net_realized_pnl
    eq=pd.DataFrame(dict(time=f.decision,long_pnl=pnl[:,0],short_pnl=pnl[:,1],combined_pnl=pnl.sum(axis=1)))
    return eq


def perf(t, eq):
    # Period PNL is close-to-close marked equity difference; closed trade statistics use exit year.
    formal_eq=eq[eq.time>=FORMAL].copy()
    daily=formal_eq.set_index('time')[['long_pnl','short_pnl','combined_pnl']].resample('1D').last().ffill()
    daily_delta=daily.diff();daily_delta.iloc[0]=daily.iloc[0]
    periods=[('2020-2023',2020,2023),('2020-2022',2020,2022)]+[(str(y),y,y) for y in range(2020,2024)]
    rows=[];months=[]
    exit_year=pd.to_datetime(t.exit_bar_open_time,utc=True).dt.year
    for period,lo,hi in periods:
        em=(formal_eq.time.dt.year>=lo)&(formal_eq.time.dt.year<=hi)
        for side in ['long','short','combined']:
            col=side+'_pnl';curve=formal_eq.loc[em,col].to_numpy()
            if not len(curve):continue
            prior=formal_eq.loc[formal_eq.time.dt.year<lo,col]
            base=prior.iloc[-1] if len(prior) else 0.
            levels=np.r_[base,curve]
            dd=np.maximum.accumulate(levels)-levels
            den=1000+np.maximum.accumulate(levels)
            returns=daily_delta.loc[(daily_delta.index.year>=lo)&(daily_delta.index.year<=hi),col]/1000
            sd=returns.std(ddof=1);down=np.sqrt(np.mean(np.minimum(returns,0)**2))
            g=t[(t.status=='closed')&(exit_year>=lo)&(exit_year<=hi)]
            if side!='combined':g=g[g.direction==side]
            wins=g.loc[g.net_realized_pnl>0,'net_realized_pnl'];loss=g.loc[g.net_realized_pnl<0,'net_realized_pnl']
            op=t[t.status!='closed']
            if side!='combined':op=op[op.direction==side]
            rows.append(dict(period=period,direction=side,net_profit_marked=curve[-1]-base,closed_trade_net_profit=g.net_realized_pnl.sum(),max_drawdown_currency=dd.max(),max_drawdown_pct_of_running_equity=100*np.max(dd/den),max_drawdown_pct_fixed1000=dd.max()/10,trades_closed=len(g),entries=int(((t.entry_year>=lo)&(t.entry_year<=hi)&((t.direction==side) if side!='combined' else True)).sum()),win_rate=(g.net_realized_pnl>0).mean(),profit_factor=wins.sum()/(-loss.sum()) if len(loss) else np.nan,sharpe=np.sqrt(365)*returns.mean()/sd if sd>0 else np.nan,sortino=np.sqrt(365)*returns.mean()/down if down>0 else np.nan,mean_win=wins.mean(),mean_loss=loss.mean(),payoff_ratio=wins.mean()/(-loss.mean()) if len(wins) and len(loss) else np.nan,mean_holding_hours=g.holding_duration_hours.mean(),median_holding_hours=g.holding_duration_hours.median(),mean_MFE_pct=g.MFE_pct.mean(),mean_MAE_pct=g.MAE_pct.mean(),mean_giveback_pct=g.giveback_pct.mean(),censored_at_final_cutoff_n=len(op) if hi==2023 else 0))
    for month, g in formal_eq.groupby(formal_eq.time.dt.strftime('%Y-%m')):
        for side in ['long','short','combined']:
            col=side+'_pnl';before=formal_eq.loc[formal_eq.time<g.time.iloc[0],col]
            base=before.iloc[-1] if len(before) else 0.
            delta=g[col].iloc[-1]-base
            months.append(dict(month=month,direction=side,net_pnl_marked=delta,return_pct_fixed1000=delta/10))
    return pd.DataFrame(rows),pd.DataFrame(months)


def draw(t, eq, arrays, f):
    # Internal research report, not a publisher-specific figure. Full observations, no trimming.
    colors={'long':'#0072B2','short':'#A64B00'}
    fig,axs=plt.subplots(3,2,figsize=(13,12),layout='constrained')
    for side in SIGN:
        g=t[(t.direction==side)&(t.status=='closed')];c=colors[side]
        for ax,col,label in [(axs[0,1],'entry_wait_minutes','Entry wait (minutes)'),(axs[1,0],'MFE_pct','MFE (% price)'),(axs[1,1],'MAE_pct','MAE (% price)'),(axs[2,0],'giveback_pct','Giveback (percentage points)')]:
            v=np.sort(g[col].dropna());ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',label=f'{side} n={len(v)}',color=c,linestyle='-' if side=='long' else '--');ax.set(xlabel=label,ylabel='Empirical cumulative fraction');ax.legend()
        yearly=g.groupby('entry_year').size().reindex(range(2020,2024),fill_value=0)
        axs[0,0].plot(yearly.index,yearly.values,label=side,color=c,marker='o' if side=='long' else 's',linestyle='-' if side=='long' else '--')
        axs[2,1].plot(eq.time,eq[side+'_pnl'],color=c,label=side,linestyle='-' if side=='long' else '--')
    axs[0,0].set(xlabel='Entry year',ylabel='Closed trades',ylim=(0,None),xticks=list(range(2020,2024)));axs[0,0].legend()
    axs[2,1].set(xlabel='UTC',ylabel='Cumulative marked PNL (budget units)');axs[2,1].legend()
    axs[2,1].set_xlim(eq.time.iloc[0],eq.time.iloc[-1])
    fig.suptitle('Frozen V18, 2020–2023 | ECDFs retain all observations; funding excluded')
    fig.savefig(OUT/'figures/diagnostic_distributions.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained',sharex=True,sharey=True)
    mask=(f.decision>=FORMAL)
    xmax=float(f.model_accel_scaled[mask].abs().max())*1.03
    ymax=float(f.trend8[mask].abs().max())*1.03
    maps=[]
    for ax,(side,s) in zip(axs,SIGN.items()):
        maps.append(ax.hexbin((s*f.model_accel_scaled)[mask],(s*f.trend8)[mask],gridsize=65,mincnt=1,cmap='cividis',bins='log',extent=(-xmax,xmax,-ymax,ymax)))
        ax.axvline(arrays[s]['model']['core_parameters'][2],color='black',linestyle='--');ax.axhline(0,color='black',linestyle=':')
        ax.set(title=side,xlabel='Direction-adjusted scaled acceleration A',ylabel='Direction-adjusted slope trend T')
    vmax=max(float(m.get_array().max()) for m in maps)
    for m in maps:m.set_clim(1,vmax)
    fig.colorbar(maps[0],ax=list(axs),label='Candles per cell (logarithmic scale)')
    fig.suptitle('A versus T | log10 cell counts, all eligible candles')
    fig.savefig(OUT/'figures/A_T_relationship.png',dpi=160);plt.close(fig)


def main():
    source_files=['ETH_KalmanMomentum_V18C3_TwoConfirm.py','reference_replay.py','reference_independent_zero_slip.py','audit_support.py']
    binding=json.loads((BASE/'manifest.json').read_text(encoding='utf-8'))
    checksums={}
    for name in source_files:
        checksums[name]=sha(BASE/name)
        check('frozen hash '+name,checksums[name]==binding[name] and sha(OUT/'inputs'/name)==checksums[name])
    check('frozen model hash',sha(OUT/'inputs/selected_models_30.json')==binding['assets\\selected_models_30.json'])
    # This pre-2024-only OHLCV file avoids loading any post-2023 market rows.
    data_path=ROOT/'stage4_trade_attribution/inputs/ohlcv_pre2024.feather'
    data_manifest=json.loads((ROOT/'stage4_trade_attribution/manifest.json').read_text(encoding='utf-8'))
    check('pre2024 input provenance',sha(data_path)==data_manifest['inputs/ohlcv_pre2024.feather'])
    dates=pd.read_feather(data_path,columns=['date']).date
    check('input has no post2023 open timestamp',bool((dates< pd.Timestamp('2024-01-01',tz='UTC')).all()))
    raw=pd.read_feather(data_path)
    raw=raw[(raw.date>=START)&(raw.date+BAR<=END)].reset_index(drop=True)
    check('strict closed-bar range',raw.date.iloc[0]==START and raw.date.iloc[-1]+BAR<=END and len(raw)==140255)
    check('contiguous and unique',raw.date.is_unique and raw.date.diff().dropna().eq(BAR).all())
    raw.to_feather(OUT/'inputs/ohlcv_closed_pre2024.feather')
    f=original_features(raw)
    check('4h EMA availability',bool((f.available_at.dropna()<=f.decision[f.available_at.notna()]).all()))
    models=json.loads((OUT/'inputs/selected_models_30.json').read_text(encoding='utf-8'))
    arrays=get_signals(f,models)
    # Prefix test checks causality without any future data in either computation.
    prefix=original_features(raw.iloc[:35000].copy());pa=get_signals(prefix,models)
    cols=['slope','model_accel','slope_scaled','model_accel_scaled','momentum4','trend8','fast','slow']
    check('indicator prefix invariance',np.allclose(prefix[cols],f.iloc[:len(prefix)][cols],rtol=0,atol=1e-12,equal_nan=True))
    for s in [1,-1]:check('quality prefix invariance '+str(s),np.allclose(pa[s]['score'],arrays[s]['score'][:len(prefix)],rtol=0,atol=1e-12))
    # Allow exactly the two confirmation closes needed for the formal 23:45 open.
    offset=int(np.flatnonzero(f.date>=FORMAL)[0])-2
    d=f.iloc[offset:].reset_index(drop=True)
    da={s:{k:(v[offset:] if isinstance(v,np.ndarray) else v) for k,v in a.items()} for s,a in arrays.items()}
    gates={s:a['gate'] for s,a in da.items()}
    zero,qzero,plan=execute(d,models,gates,0.)
    it,ic=independent(d,plan,'baseline',allow_open=True,entry_gate=gates)
    cl=zero[zero.status.eq('closed')]
    check('independent frozen zero-slip orders',it[['entry_i','exit_i','side','exit_reason']].values.tolist()==cl[['entry_i','exit_i','side','exit_reason']].values.tolist())
    check('independent frozen zero-slip equity',np.allclose(qzero,ic,rtol=0,atol=1e-11))
    t,q,plan=execute(d,models,gates,SLIP)
    # Independent scalar engine uses explicit zero literals: parameterize these in memory only.
    src=(OUT/'inputs/reference_independent_zero_slip.py').read_text(encoding='utf-8')
    src=src.replace('price*(1-s*0.)','price*(1-s*0.0002)').replace('b.open*(1+s*0.)','b.open*(1+s*0.0002)')
    ns={};exec(compile(src,'independent_frozen_cost_audit','exec'),ns)
    ci,ce=ns['independent'](d,plan,'baseline',allow_open=True,entry_gate=gates)
    cl=t[t.status.eq('closed')]
    check('independent original-cost orders',ci[['entry_i','exit_i','side','exit_reason']].values.tolist()==cl[['entry_i','exit_i','side','exit_reason']].values.tolist())
    check('independent original-cost equity',np.allclose(q,ce,rtol=0,atol=1e-11))
    check('no entry before formal start',bool((t.entry_time>=FORMAL).all()))
    td=trade_diagnostics(t,d,da)
    eq=equity_paths(d,td)
    check('diagnostic PNL reconciles original engine',np.allclose(1+eq.combined_pnl.to_numpy()/1000,q,rtol=0,atol=1e-11))
    check('no actual 2024 exits',bool((pd.to_datetime(td.exit_bar_open_time,utc=True).dropna()<=END).all()))
    save(td,'trades_diagnostic.csv');save(eq[eq.time>=FORMAL],'equity_curve.csv')
    entry=delay_summary(td);ex=exit_summary(td);sf=signal_stats(f,arrays,td)
    annual,monthly=perf(td,eq)
    save(entry,'entry_delay_summary.csv');save(ex,'exit_giveback_summary.csv');save(sf,'signal_filter_summary.csv')
    save(annual,'annual_performance.csv');save(monthly,'monthly_performance.csv')
    save(td[td.status!='closed'],'censored_open_trades.csv')
    draw(td,eq,da,d)
    for name in source_files:check('source unchanged '+name,sha(BASE/name)==checksums[name])
    check('model unchanged',sha(BASE/'assets/selected_models_30.json')==sha(OUT/'inputs/selected_models_30.json'))
    save(pd.DataFrame(CHECKS),'verification.csv')
    meta=dict(task='V18_C3_001_30_BASELINE_DIAGNOSTIC_2020_2023',status='completed',version='V18_C3_001_30_20261021',strategy='ETH_KalmanMomentum_V18C3_TwoConfirm',data_source=str(data_path),data_source_sha256=sha(data_path),frozen_source_hashes=checksums,model_sha256=sha(OUT/'inputs/selected_models_30.json'),loaded_market_open_min=str(raw.date.iloc[0]),loaded_market_open_max=str(raw.date.iloc[-1]),last_complete_close=str(raw.date.iloc[-1]+BAR),cutoff=str(END),formal_start=str(FORMAL),bars=len(raw),excluded_incomplete_final_candles=1,first_confirmation_candidate=str(d.decision.iloc[0]),continuous_single_position=True,year_resets=False,margin=1000,leverage=2,compound=False,fee_per_side=FEE,slippage_per_side=SLIP,funding_included=False,liquidation_model=False,closed_trades=int(td.status.eq('closed').sum()),censored_open_trades=int(td.status.ne('closed').sum()),post2023_market_data_used=False,precomputed_features_used=False,refit_or_optimization=False,independent_reconciliation=True,shadow_trades_simulated=False,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__,checks_passed=len(CHECKS),figure_semantics='ECDF, all observations, no trimming; complete-held-bars-only MFE/MAE; stop-touch partial bar excluded; internal research report, 160 DPI',statistical_status='2020-2022 training/parameter optimization; 2023 recipe selection; final caps selected using 2023-2024 historically. Retrospective audit, not independent OOS.')
    (OUT/'diagnostic_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:meta[k] for k in ['bars','closed_trades','censored_open_trades','checks_passed']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
