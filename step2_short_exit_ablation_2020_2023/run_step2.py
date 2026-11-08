"""Step 2 A/B ablation: frozen short momentum exit qualification versus sign-only exit."""
from pathlib import Path
import sys,json,hashlib,platform
HERE=Path(__file__).resolve().parent
STEP1=HERE.parent
sys.path.insert(0,str(STEP1.parent.parent/'.research_deps'))
import numpy as np,pandas as pd
sys.path.insert(0,str(HERE)); import run_audit as audit
audit.OUT=STEP1
OUT=HERE
FEE,SLIP= audit.FEE,audit.SLIP

def plan_for(d,models,variant):
    cfg={s:next(z for z in models if z['case']=='unfiltered_'+side)['core_parameters'] for side,s in audit.SIGN.items()}
    p=audit.schedule(d,'baseline',cfg[1],cfg[-1])
    if variant=='B':
        p.loc[p.side.eq('short'),'exit_kind']='sign'
        # Retain frozen value as an audit record; replay ignores it for sign exits.
    return p

def run_variant(d,models,gates,variant,slip):
    p=plan_for(d,models,variant);es,xs=audit.schedules_from_table(p)
    risks={s:{z.month:(z.early_loss_pct,z.stop_pct/100) for z in p[p.side.eq(side)].itertuples()} for side,s in audit.SIGN.items()}
    audit.replay.SLIP=slip
    t,q=audit.replay.run(d,es,xs,None,early_exit=True,entry_gate=gates,fixed_account_ratio=10,risk_schedule=risks)
    return t,q[:,1],p

def independent_variant(d,p,gates,slip):
    # Independent verifier shipped by Step 1 has explicit zero-cost literals. Make a
    # temporary in-memory copy with only the registered cost constants substituted.
    src=(HERE/'reference_independent_zero_slip.py').read_text(encoding='utf-8')
    if slip:
        src=src.replace('price*(1-s*0.)','price*(1-s*0.0002)').replace('b.open*(1+s*0.)','b.open*(1+s*0.0002)')
    ns={};exec(compile(src,'independent_step2','exec'),ns)
    return ns['independent'](d,p,'baseline',allow_open=True,entry_gate=gates)

def eq_diag(d,td):
    return audit.equity_paths(d,td)

def perf(t,eq):
    # Reuse Step 1 definitions; this produces identical A values when A is restored.
    return audit.perf(t,eq)

def timing(td,variant):
    x=td[(td.direction=='short')&(td.exit_group=='normal')].copy()
    return dict(variant=variant,n=len(x),mean_raw_delay_bars=x.normal_delay_from_raw_bars.mean(),median_raw_delay_bars=x.normal_delay_from_raw_bars.median(),mean_full_delay_bars=x.normal_delay_from_full_bars.mean(),median_full_delay_bars=x.normal_delay_from_full_bars.median(),mean_hold_hours=x.holding_duration_hours.mean(),median_hold_hours=x.holding_duration_hours.median(),mean_MFE_pct=x.MFE_pct.mean(),mean_MAE_pct=x.MAE_pct.mean(),mean_giveback_pct=x.giveback_pct.mean(),normal_net_pnl=x.net_realized_pnl.sum(),exit_reason_counts='; '.join(f'{k}={v}' for k,v in td[td.direction=='short'].exit_group.value_counts().sort_index().items()))

def main():
    verified=[]
    def checked(name,ok):
        assert bool(ok),'STOP '+name
        verified.append(dict(check=name,passed=True))
        print('PASS',name,flush=True)
    source_hashes={}
    for name in ['selected_models_30.json','ETH_KalmanMomentum_V18C3_TwoConfirm.py','reference_replay.py','reference_independent_zero_slip.py','ohlcv_closed_pre2024.feather']:
        source_hashes[name]=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
        checked('Step1 input exact hash '+name,source_hashes[name]==hashlib.sha256((STEP1/'inputs'/name).read_bytes()).hexdigest())
    models=json.loads((HERE/'selected_models_30.json').read_text(encoding='utf-8'))
    raw=pd.read_feather(HERE/'ohlcv_closed_pre2024.feather')
    checked('strict complete candle range and count',len(raw)==140255 and raw.date.iloc[0]==audit.START and bool((raw.date+audit.BAR<=audit.END).all()))
    checked('unique contiguous candles',raw.date.is_unique and raw.date.diff().dropna().eq(audit.BAR).all())
    f=audit.original_features(raw); arrays=audit.get_signals(f,models)
    off=int(np.flatnonzero(f.date>=audit.FORMAL)[0])-2; d=f.iloc[off:].reset_index(drop=True)
    da={s:{k:(v[off:] if isinstance(v,np.ndarray) else v) for k,v in a.items()} for s,a in arrays.items()}; gates={s:a['gate'] for s,a in da.items()}
    variants={}
    plana=plan_for(d,models,'A');planb=plan_for(d,models,'B')
    altered=plana.copy();altered.loc[altered.side.eq('short'),'exit_kind']='sign'
    checked('single-factor monthly plan difference',altered.equals(planb))
    checked('short recorded exit_value retained',planb.loc[planb.side.eq('short'),'exit_value'].eq(.38).all())
    plana.to_csv(OUT/'monthly_plan_A.csv',index=False);planb.to_csv(OUT/'monthly_plan_B.csv',index=False)
    for v in ['A','B']:
        t0,q0,p=run_variant(d,models,gates,v,0.)
        i0,iq=independent_variant(d,p,gates,0.)
        cl=t0[t0.status.eq('closed')]
        assert len(cl)==len(i0) and cl[['entry_i','exit_i','side','exit_reason']].values.tolist()==i0[['entry_i','exit_i','side','exit_reason']].values.tolist(),v+' zero-slip order mismatch'
        assert np.allclose(q0,iq,rtol=0,atol=1e-11),v+' zero-slip curve mismatch'
        checked(v+' independent zero-slip order sequence',True)
        checked(v+' independent zero-slip equity curve',True)
        checked(v+' independent zero-slip fills',np.allclose(cl[['entry_price','exit_price']],i0[['entry_price','exit_price']],rtol=0,atol=1e-10))
        t,q,p=run_variant(d,models,gates,v,SLIP)
        i,iq=independent_variant(d,p,gates,SLIP); cl=t[t.status.eq('closed')]
        assert cl[['entry_i','exit_i','side','exit_reason']].values.tolist()==i[['entry_i','exit_i','side','exit_reason']].values.tolist(),v+' cost order mismatch'
        assert np.allclose(q,iq,rtol=0,atol=1e-11),v+' cost curve mismatch'
        checked(v+' independent original-cost order sequence',True)
        checked(v+' independent original-cost equity curve',True)
        checked(v+' independent original-cost fills',np.allclose(cl[['entry_price','exit_price']],i[['entry_price','exit_price']],rtol=0,atol=1e-10))
        td=audit.trade_diagnostics(t,d,da,exit_kind_override='sign' if v=='B' else None)
        eq=eq_diag(d,td); annual,monthly=perf(td,eq)
        if v=='A':
            expected=pd.read_csv(STEP1/'trades_diagnostic.csv')
            for col in ['actual_entry_time','exit_bar_open_time','direction','exit_reason']:
                assert td[col].astype(str).tolist()==expected[col].astype(str).tolist(),'STOP A Step1 trade mismatch '+col
            oldeq=pd.read_csv(STEP1/'equity_curve.csv')
            assert np.allclose(eq.loc[eq.time>=audit.FORMAL,['long_pnl','short_pnl','combined_pnl']],oldeq[['long_pnl','short_pnl','combined_pnl']],rtol=0,atol=1e-8),'STOP A Step1 equity mismatch'
            assert len(td)==710 and td.status.eq('closed').all(),'STOP A trade count'
            assert abs(float(annual.query("period=='2020-2023' and direction=='combined'").net_profit_marked.iloc[0])-6954.3495)<1e-3,'STOP A total PNL'
            assert abs(float(annual.query("period=='2023' and direction=='combined'").net_profit_marked.iloc[0])+94.9795)<1e-3,'STOP A 2023 PNL'
            print('A exact Step1 reconciliation passed; B now authorized to run',flush=True)
            checked('A Step1 all entry exit direction reasons exactly equal',True)
            checked('A Step1 full directional and combined equity equal',True)
        td.to_csv(OUT/('trades_A_baseline.csv' if v=='A' else 'trades_B_sign_exit.csv'),index=False,encoding='utf-8-sig')
        eq[eq.time>=audit.FORMAL].to_csv(OUT/('equity_curve_A.csv' if v=='A' else 'equity_curve_B.csv'),index=False,encoding='utf-8-sig')
        annual.to_csv(OUT/('annual_A.csv' if v=='A' else 'annual_B.csv'),index=False,encoding='utf-8-sig')
        monthly.to_csv(OUT/('monthly_A.csv' if v=='A' else 'monthly_B.csv'),index=False,encoding='utf-8-sig')
        variants[v]=(t,td,eq,annual,monthly,p)
        print(v,'trades',len(t),'closed',int(t.status.eq('closed').sum()),'net',float(annual.query("period=='2020-2023' and direction=='combined'").net_profit_marked.iloc[0]),flush=True)
    ta,tda,eqa,aa,ma,pa=variants['A']; tb,tdb,eqb,ab,mb,pb=variants['B']
    # Exact Step 1 baseline gates.
    a_net=float(aa.query("period=='2020-2023' and direction=='combined'").net_profit_marked.iloc[0]); a23=float(aa.query("period=='2023' and direction=='combined'").net_profit_marked.iloc[0])
    assert int(ta.status.eq('closed').sum())==710 and abs(a_net-6954.3495)<1e-3 and abs(a23-(-94.9795))<1e-3
    periods=aa[['period','direction']].drop_duplicates(); rows=[]
    for _,key in periods.iterrows():
        p=key.period;s=key.direction
        xa=aa[(aa.period==p)&(aa.direction==s)].iloc[0];xb=ab[(ab.period==p)&(ab.direction==s)].iloc[0]
        row={'period':p,'direction':s}
        for col in ['net_profit_marked','max_drawdown_currency','max_drawdown_pct_fixed1000','trades_closed','entries','win_rate','profit_factor','sharpe','sortino','mean_win','mean_loss','payoff_ratio','mean_holding_hours','median_holding_hours','mean_MFE_pct','mean_MAE_pct','mean_giveback_pct','censored_at_final_cutoff_n']:
            row['A_'+col]=xa[col];row['B_'+col]=xb[col];row['B_minus_A_'+col]=xb[col]-xa[col] if pd.notna(xa[col]) and pd.notna(xb[col]) else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'comparison_period_direction.csv',index=False,encoding='utf-8-sig')
    m=ma.merge(mb,on=['month','direction'],suffixes=('_A','_B'));m['B_minus_A_net_pnl_marked']=m.net_pnl_marked_B-m.net_pnl_marked_A;m['B_minus_A_return_pct_fixed1000']=m.return_pct_fixed1000_B-m.return_pct_fixed1000_A;m.to_csv(OUT/'monthly_comparison.csv',index=False,encoding='utf-8-sig')
    st=pd.DataFrame([timing(tda,'A'),timing(tdb,'B')]);st['B_minus_A_mean_raw_delay_bars']=np.nan;st.loc[st.variant=='B','B_minus_A_mean_raw_delay_bars']=st.loc[st.variant=='B','mean_raw_delay_bars'].iloc[0]-st.loc[st.variant=='A','mean_raw_delay_bars'].iloc[0];st.to_csv(OUT/'short_exit_timing.csv',index=False,encoding='utf-8-sig')
    # Exit counts / order and curve reconciliations.
    checks=list(verified)
    def ck(n,ok): checks.append({'check':n,'passed':bool(ok)});assert ok,n
    ck('A Step1 closed count',int(ta.status.eq('closed').sum())==710);ck('A Step1 net',abs(a_net-6954.3495)<1e-3);ck('A Step1 2023',abs(a23+94.9795)<1e-3)
    for v,(t,td,eq,ann,mon,p) in variants.items():
        ck(v+' diagnostic record count',len(t)==len(td));ck(v+' no open at cutoff',int(td.status.ne('closed').sum())==0);ck(v+' no post2023 exit',bool(pd.to_datetime(td.exit_bar_open_time,utc=True).dropna().le(audit.END).all()));ck(v+' PNL reconciliation',np.allclose(1+eq.combined_pnl.to_numpy()/1000,run_variant(d,models,gates,v,SLIP)[1],rtol=0,atol=1e-11))
    ck('B short exit kind changed only',pb[pb.side=='long'].exit_kind.eq('sign').all() and pb[pb.side=='short'].exit_value.eq(.38).all())
    for name,digest in source_hashes.items():ck('input unchanged after run '+name,hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest and hashlib.sha256((STEP1/'inputs'/name).read_bytes()).hexdigest()==digest)
    # Plot: equity, drawdown, short delay, annual combined.
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained');
    for eq,c,l in [(eqa,'#0072B2','A baseline'),(eqb,'#D55E00','B sign-only')]:axs[0,0].plot(eq.time,eq.combined_pnl,label=l,color=c)
    axs[0,0].set(title='A/B combined marked PNL',ylabel='USDT budget units');axs[0,0].legend()
    for eq,c,l in [(eqa,'#0072B2','A baseline'),(eqb,'#D55E00','B sign-only')]:
        dd=eq.combined_pnl.cummax()-eq.combined_pnl;axs[0,1].plot(eq.time,dd,label=l,color=c)
    axs[0,1].set(title='A/B close-based drawdown',ylabel='USDT budget units');axs[0,1].legend()
    for td,c,l in [(tda,'#0072B2','A baseline'),(tdb,'#D55E00','B sign-only')]:
        z=np.sort(td[(td.direction=='short')&(td.exit_group=='normal')].normal_delay_from_raw_bars.dropna());axs[1,0].step(z,np.arange(1,len(z)+1)/len(z),where='post',label=l,color=c)
    axs[1,0].set(title='Short normal exit delay from raw slope reversal',xlabel='15-minute bars',ylabel='ECDF');axs[1,0].legend()
    piv=pd.DataFrame(rows);piv=piv[(piv.period.isin(['2020','2021','2022','2023']))&(piv.direction=='combined')]
    x=np.arange(len(piv));axs[1,1].bar(x-.18,piv.A_net_profit_marked,.36,label='A baseline',color='#0072B2');axs[1,1].bar(x+.18,piv.B_net_profit_marked,.36,label='B sign-only',color='#D55E00');axs[1,1].set(xticks=x,xticklabels=piv.period,title='Annual combined marked PNL',ylabel='USDT budget units');axs[1,1].axhline(0,color='black',lw=.8);axs[1,1].legend()
    fig.savefig(OUT/'step2_comparison.png',dpi=160);plt.close(fig)
    pd.DataFrame(checks).to_csv(OUT/'verification.csv',index=False,encoding='utf-8-sig')
    meta=dict(task='V19 Step 2 short exit momentum ablation 2020-2023',source_sha256=source_hashes,base_commit='03a71cf8c587bf7bf67b17aef7513fab3e6719e4',A='frozen V18 momentum exit',B='short normal exit sign-only; exit_value 0.38 retained as audit record',data='Step 1 verified inputs/ohlcv_closed_pre2024.feather',formal_start=str(audit.FORMAL),cutoff=str(audit.END),no_post2023=True,fee_per_side=FEE,slippage_per_side=SLIP,margin=1000,leverage=2,compound=False,funding=False,annual_restarts=False,model_unchanged=True,source_unchanged=True,only_factor='short exit_kind momentum -> sign',A_expected={'closed_trades':710,'net_profit_combined':6954.3495,'net_profit_2023_combined':-94.9795},A_actual={'closed_trades':int(ta.status.eq('closed').sum()),'net_profit_combined':a_net,'net_profit_2023_combined':a23},B_actual={'closed_trades':int(tb.status.eq('closed').sum()),'net_profit_combined':float(ab.query("period=='2020-2023' and direction=='combined'").net_profit_marked.iloc[0])},checks_passed=len(checks),python=platform.python_version())
    (OUT/'diagnostic_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
