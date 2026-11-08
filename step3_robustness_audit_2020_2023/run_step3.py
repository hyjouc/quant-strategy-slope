"""Retrospective robustness audit. No parameter/model fitting or new thresholds.

Only market input: Step1 strictly complete pre-2024 OHLCV. Historical optimizer
scripts are read as text for provenance; never imported or executed.
"""
from pathlib import Path
import sys,json,hashlib,subprocess,math,re
HERE=Path(__file__).resolve().parent;REPO=HERE.parent;ROOT=REPO.parent.parent
sys.path.insert(0,str(ROOT/'.research_deps'))
import numpy as np,pandas as pd
sys.path.insert(0,str(REPO/'step2_short_exit_ablation_2020_2023'))
import run_step2 as step2
audit=step2.audit
CHECKS=[];SEED=20261108;REPS=20000

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(n,ok):
    CHECKS.append(dict(check=n,passed=bool(ok)))
    if not ok:raise RuntimeError('STOP '+n)
    print('PASS',n,flush=True)
def save(df,name):df.to_csv(HERE/name,index=False,encoding='utf-8-sig')

def provenance():
    src=ROOT/'parameter_selection_lab_20261020/second_accel'
    paths=[src/'run_v18_c3_span001.py',ROOT/'research_split_2020_2022_validate_2023_2024_v18.py',ROOT/'research_directional_independent_v5.py',src/'v18_c3_span001/protocol.json',src/'v18_c3_span001/source_audit.json',src/'v18_c3_span001/locked_selection.json']
    evidence=[]
    for i,p in enumerate(paths,1):
        content=p.read_bytes();target=HERE/'evidence'/f'{i:02}_{p.name}';target.write_bytes(content)
        evidence.append(dict(source=str(p),snapshot=target.relative_to(HERE).as_posix(),sha256=digest(p),type='source text or metadata only; never executed'))
    method=ROOT/'V18_基准质量筛选版_完整回测方法.md'
    # Restrict to parameter/model-method paragraphs, not historical performance tables.
    lines=method.read_text(encoding='utf-8').splitlines();excerpt='\n'.join(lines[142:250])
    (HERE/'evidence/parameter_method_excerpt.md').write_text(excerpt,encoding='utf-8')
    evidence.append(dict(source=str(method),snapshot='evidence/parameter_method_excerpt.md',sha256=digest(method),type='method-only excerpt, original lines 143-250'))
    git=['git','-c',f'safe.directory={REPO.as_posix()}']
    history=subprocess.check_output(git+['log','--all','--format=%H %ad %s','--date=iso-strict'],cwd=REPO,text=True)
    (HERE/'evidence/available_git_history.txt').write_text(history,encoding='utf-8')
    models=json.loads((REPO/'inputs/selected_models_30.json').read_text(encoding='utf-8'))
    historic=json.loads((src/'v18_c3_span001/frozen_models.json').read_text(encoding='utf-8'))
    link=[]
    for m in models:
        if not m['case'].startswith('unfiltered'):continue
        h=[z for z in historic if z.get('minimum')==30 and z['case']==m['case'] and float(z['cap'])==m['cap']]
        check('selected model matches historical frozen model '+m['case'],len(h)==1 and all(h[0][k]==m[k] for k in ['coef','intercept','mean','scale','threshold','training_trades','last_training_label_exit','core_parameters']))
        link.append(dict(case=m['case'],minimum=30,cap=m['cap'],model_matches=True,training_trades=m['training_trades'],last_training_label_exit=m['last_training_label_exit']))
    save(pd.DataFrame(link),'model_provenance_link.csv')
    # Preserve only linked selected model metadata; no historical market/results.
    (HERE/'evidence/selected_model_metadata.json').write_text(json.dumps([m for m in models if m['case'].startswith('unfiltered')],indent=2),encoding='utf-8')
    rows=[]
    core_basis='2020-2022 directional coordinate search; annual positive PNL/minimum30; drawdown>32 soft penalty; neighbor/plateau selection; 3 starts, up to20 rounds; C3 mesh .001'
    for m in models:
        if not m['case'].startswith('unfiltered'):continue
        side=m['case'].split('_')[1];p=m['core_parameters']
        for name,val in zip(['M_entry_C1','S_entry_C2','A_entry_C3','exit_confirmation_C4','normal_exit_momentum_pct','early_loss_pct','fixed_stop_pct'],p):
            fixed=name=='exit_confirmation_C4'
            rows.append(dict(direction=side,parameter=name,value=val,selection_period='UNKNOWN initial mechanism choice; fixed during final 2020-2022 search' if fixed else '2020-2022 (recorded final core search)',selection_basis='fixed2 in source; original choice rationale UNKNOWN' if fixed else core_basis,actual_attempt_count='UNKNOWN',known_search_scope='3 starts x max20 rounds x6 axes; not actual trial count',complete_experiment_record='NO: source_audit reports 8 anchors but original anchor JSON and .steps.csv unavailable',evidence='01_run_v18_c3_span001.py; 02_research_split_2020_2022_validate_2023_2024_v18.py; 03_research_directional_independent_v5.py; 05_source_audit.json',initial_design_history='UNKNOWN; final documented search window does not establish untouched initial design'))
        for name,val in [('quality_coefficients',m['coef']),('quality_intercept',m['intercept']),('quality_feature_mean',m['mean']),('quality_feature_scale',m['scale']),('quality_threshold',m['threshold'])]:
            rows.append(dict(direction=side,parameter=name,value=json.dumps(val),selection_period='fit 2020-2022; model/cap choice2023-2024',selection_basis='Ridge alpha100; 6 features; train-label clipping by cap; threshold median training prediction',actual_attempt_count='UNKNOWN lifetime; 3 cap fits/direction/tier specified, 4 recipes incl none',known_search_scope='2 minimum-trade tiers x4 directional cases x3 cap fits in final pipeline; actual historical repetitions UNKNOWN',complete_experiment_record='PARTIAL: selected metadata exact match; original training sample ledger/environment incomplete',evidence='selected_model_metadata.json; 02_*.py; 06_locked_selection.json',initial_design_history='UNKNOWN prior alpha/feature/recipe design search'))
        rows.append(dict(direction=side,parameter='selected_cap',value=m['cap'],selection_period='2023-01-01 through2024-12-31',selection_basis='choose among none,.02,.04,.08: improve directional net and no worse full DD vs none; adjacent-cap worst score>none; maximum stable score',actual_attempt_count='4 candidates per direction/tier in documented final selection; lifetime repeats UNKNOWN',known_search_scope='2 tiers x4 cases; actual repeated choices UNKNOWN',complete_experiment_record='PARTIAL locked_selection and source retained; full candidate-level validation scores absent',evidence='06_locked_selection.json; 02_*.py; selected_model_metadata.json',initial_design_history='2024 explicitly used for recipe selection; no 2024 market read in this audit'))
    for name,val,basis in [('T_threshold',0,'fixed direction-adjusted trend8>0'),('Kalman_q',1e-7,'three-state Joseph recursion'),('Kalman_R',.01,'observation variance'),('Kalman_dt_hours',.25,'15-minute step'),('T_window',8,'OLS slope of last8 slope states'),('normalization_prior_bars',96,'exclude current bar, prior sample std'),('momentum_window_bars',4,'price momentum'),('entry_confirmation',2,'consecutive complete closes'),('early_confirmation',2,'consecutive risk conditions'),('armed',True,'requires ever favorable raw slope since pre-entry last bar'),('normal_exit_slope_threshold',0,'direction-adjusted raw slope<0'),('Ridge_alpha',100,'fixed in final training recipe'),('quality_EMA_spans',[20,60],'complete4h feature')]:
        rows.append(dict(direction='both',parameter=name,value=json.dumps(val),selection_period='UNKNOWN initial choice; fixed during final pipeline',selection_basis=basis,actual_attempt_count='UNKNOWN',known_search_scope='not searched in documented final core fitting',complete_experiment_record='NO complete initial design history',evidence='Step1 frozen strategy; parameter_method_excerpt.md; 02_*.py',initial_design_history='UNKNOWN'))
    save(pd.DataFrame(rows),'parameter_provenance.csv');save(pd.DataFrame(evidence),'evidence_index.csv')
    check('original anchors unavailable explicitly recorded',not(src/'v18_c3_span001/anchors/30/unfiltered_short.json').exists())

def load_baseline():
    dct={}
    for v in ['A','B']:
        p=REPO/'step2_short_exit_ablation_2020_2023'
        t=pd.read_csv(p/('trades_A_baseline.csv' if v=='A' else 'trades_B_sign_exit.csv'))
        for col in ['entry_time','exit_bar_open_time']:t[col]=pd.to_datetime(t[col],utc=True)
        eq=pd.read_csv(p/f'equity_curve_{v}.csv');eq['time']=pd.to_datetime(eq.time,utc=True)
        check(v+' pre2024 curve only',eq.time.max()<=audit.END)
        check(v+' no censored trades',t.status.eq('closed').all())
        dct[v]=(t,eq)
    return dct

def concentration(dct):
    a,b=(dct[v][0] for v in ['A','B'])
    pairs=a.merge(b,on=['entry_time','direction'],suffixes=('_A','_B'))
    joint=pairs[(pairs.direction=='short')&(pairs.exit_group_A=='normal')&(pairs.exit_group_B=='normal')].copy()
    joint['A_minus_B']=joint.net_realized_pnl_A-joint.net_realized_pnl_B
    check('238 common normal short exits',len(joint)==238)
    joint['comparison']=np.select([joint.A_minus_B>1e-8,joint.A_minus_B< -1e-8],['A_better','B_better'],default='equal')
    check('58 A-better common normal shorts',(joint.comparison=='A_better').sum()==58)
    ranked=joint.sort_values('A_minus_B',ascending=False).copy();ranked['rank']=np.arange(1,len(ranked)+1);ranked['cum_A_minus_B']=ranked.A_minus_B.cumsum()
    save(ranked,'common_normal_short_ranked.csv')
    positive=ranked[ranked.comparison=='A_better'];net=ranked.A_minus_B.sum();pos=positive.A_minus_B.sum()
    rows=[]
    for name,k in [('top5',5),('top10',10),('top20',20)]+[(f'top{pct}pct_of238',math.ceil(238*pct/100)) for pct in [5,10,20]]+[(f'top{pct}pct_of58',math.ceil(58*pct/100)) for pct in [5,10,20]]:
        subset=ranked.iloc[:k];rest=ranked.iloc[k:]
        # Paired deletion removes the same matched entries from both ledgers.
        pa=a.net_realized_pnl.sum()-subset.net_realized_pnl_A.sum();pb=b.net_realized_pnl.sum()-subset.net_realized_pnl_B.sum()
        rows.append(dict(deletion=name,removed_n=k,removed_A_minus_B=subset.A_minus_B.sum(),share_of_common_net_advantage=subset.A_minus_B.sum()/net,share_of_58_positive_advantage=subset.A_minus_B.sum()/pos,remaining_common_A_pnl=rest.net_realized_pnl_A.sum(),remaining_common_B_pnl=rest.net_realized_pnl_B.sum(),remaining_common_A_minus_B=rest.A_minus_B.sum(),diagnostic_ledger_A_minus_B=pa-pb,diagnostic_ledger_A_pnl=pa,diagnostic_ledger_B_pnl=pb,not_a_replayed_strategy=True))
    save(pd.DataFrame(rows),'concentration_deletions.csv')
    summary=pd.DataFrame([dict(cohort='common_normal_short_238',n=len(joint),A_pnl=joint.net_realized_pnl_A.sum(),B_pnl=joint.net_realized_pnl_B.sum(),A_minus_B=net),dict(cohort='A_better_58',n=58,A_pnl=positive.net_realized_pnl_A.sum(),B_pnl=positive.net_realized_pnl_B.sum(),A_minus_B=pos),dict(cohort='B_better_108',n=108,A_pnl=ranked.loc[ranked.comparison=='B_better','net_realized_pnl_A'].sum(),B_pnl=ranked.loc[ranked.comparison=='B_better','net_realized_pnl_B'].sum(),A_minus_B=ranked.loc[ranked.comparison=='B_better','A_minus_B'].sum()),dict(cohort='full_actual_portfolio',n=np.nan,A_pnl=a.net_realized_pnl.sum(),B_pnl=b.net_realized_pnl.sum(),A_minus_B=a.net_realized_pnl.sum()-b.net_realized_pnl.sum())])
    save(summary,'concentration_summary.csv')
    return ranked

def daily_curves(dct):
    out=[]
    for v,(t,eq) in dct.items():
        # Calendar UTC daily last complete mark; no artificial 2024 mark is appended.
        day=eq.set_index('time')[['long_pnl','short_pnl','combined_pnl']].resample('1D').last().ffill()
        delta=day.diff();delta.iloc[0]=day.iloc[0]
        delta.columns=[f'{v}_{x}' for x in ['long','short','combined']];out.append(delta)
    daily=pd.concat(out,axis=1);check('A/B daily aligned complete index',daily.notna().all().all() and daily.index.is_unique)
    for side in ['long','short','combined']:daily['B_minus_A_'+side]=daily['B_'+side]-daily['A_'+side]
    save(daily.reset_index(),'daily_marked_pnl.csv')
    return daily

def windows(dct,daily):
    # Non-overlapping calendar windows and every full rolling calendar month endpoint.
    specs=[]
    for y in range(2020,2024):specs.append(('annual',str(y),pd.Timestamp(f'{y}-01-01',tz='UTC'),pd.Timestamp(f'{y+1}-01-01',tz='UTC')))
    for months,kind in [(3,'quarterly'),(1,'monthly')]:
        for st in pd.date_range('2020-01-01','2023-12-01',freq=f'{months}MS',tz='UTC'):
            specs.append((kind,st.strftime('%Y-%m'),st,st+pd.DateOffset(months=months)))
    for n in [6,12]:
        for end in pd.date_range(pd.Timestamp('2020-01-01',tz='UTC')+pd.DateOffset(months=n),pd.Timestamp('2024-01-01',tz='UTC'),freq='MS'):
            st=end-pd.DateOffset(months=n);specs.append((f'rolling{n}m',st.strftime('%Y-%m')+'_'+(end-pd.DateOffset(months=1)).strftime('%Y-%m'),st,end))
    rows=[]
    for kind,label,st,en in specs:
        for side in ['long','short','combined']:
            rec=dict(window_type=kind,window=label,start_utc=str(max(st,audit.FORMAL)),end_exclusive_utc=str(en),direction=side)
            for v,(t,eq) in dct.items():
                mask=(eq.time>=max(st,audit.FORMAL))&(eq.time<en);curve=eq.loc[mask,side+'_pnl'].to_numpy();before=eq.loc[eq.time<max(st,audit.FORMAL),side+'_pnl'];base=before.iloc[-1] if len(before) else 0.
                lev=np.r_[base,curve];dd=np.maximum.accumulate(lev)-lev
                dr=daily.loc[(daily.index>=st)&(daily.index<en),v+'_'+side]/1000
                trades=t[(t.exit_bar_open_time>=max(st,audit.FORMAL))&(t.exit_bar_open_time<en)]
                if side!='combined':trades=trades[trades.direction==side]
                pn=trades.net_realized_pnl;gp=pn[pn>0].sum();gl=-pn[pn<0].sum()
                metrics=dict(net_profit=curve[-1]-base,max_drawdown=dd.max(),profit_factor=gp/gl if gl else np.nan,sharpe=np.sqrt(365)*dr.mean()/dr.std(ddof=1) if dr.std(ddof=1)>0 else np.nan,trades_closed=len(pn),closed_trade_pnl=pn.sum(),days=len(dr))
                for k,val in metrics.items():rec[v+'_'+k]=val
            for k in ['net_profit','max_drawdown','profit_factor','sharpe','trades_closed','closed_trade_pnl']:rec['B_minus_A_'+k]=rec['B_'+k]-rec['A_'+k]
            rows.append(rec)
    w=pd.DataFrame(rows);save(w,'time_stability_all_windows.csv')
    sm=[]
    for (kind,side),g in w.groupby(['window_type','direction']):
        sm.append(dict(window_type=kind,direction=side,windows=len(g),A_profit_above_B_n=int((g.B_minus_A_net_profit< -1e-8).sum()),B_profit_above_A_n=int((g.B_minus_A_net_profit>1e-8).sum()),equal_n=int((g.B_minus_A_net_profit.abs()<=1e-8).sum()),A_lower_drawdown_n=int((g.B_minus_A_max_drawdown>1e-8).sum()),A_higher_PF_n=int((g.B_minus_A_profit_factor< -1e-8).sum()),A_higher_sharpe_n=int((g.B_minus_A_sharpe< -1e-8).sum()),median_B_minus_A_net=g.B_minus_A_net_profit.median(),min_B_minus_A_net=g.B_minus_A_net_profit.min(),max_B_minus_A_net=g.B_minus_A_net_profit.max()))
    save(pd.DataFrame(sm),'time_stability_summary.csv');return w

def bootstrap(daily):
    # Paired noncircular moving-block bootstrap, all overlapping daily starts,
    # block28/56 consecutive days; last sampled block truncated to sample length.
    allrows=[];summ=[];n=len(daily)
    for L in [28,56]:
        rng=np.random.default_rng(SEED+L);blocks=math.ceil(n/L);last=n-(blocks-1)*L
        cols=[v+'_'+s for v in ['A','B'] for s in ['long','short','combined']];x=daily[cols].to_numpy()
        cs=np.vstack([np.zeros((1,6)),np.cumsum(x,axis=0)])
        for chunk in range(0,REPS,1000):
            st=rng.integers(0,n-L+1,size=(min(1000,REPS-chunk),blocks))
            total=(cs[st[:,:-1]+L]-cs[st[:,:-1]]).sum(axis=1)+(cs[st[:,-1]+last]-cs[st[:,-1]])
            for j in range(len(st)):
                row=dict(block_days=L,replicate=chunk+j)
                for k,col in enumerate(cols):row[col]=total[j,k]
                for side in ['long','short','combined']:row['B_minus_A_'+side]=row['B_'+side]-row['A_'+side]
                allrows.append(row)
        # Demonstrate exact same paired draw indices: linearity yields identical
        # sampled delta whether subtracting sampled totals or sampling daily deltas.
        direct=daily.B_minus_A_combined.to_numpy();dc=np.r_[0.,direct.cumsum()]
        dt=(dc[st[:,:-1]+L]-dc[st[:,:-1]]).sum(axis=1)+(dc[st[:,-1]+last]-dc[st[:,-1]])
        check(f'paired draw linearity block{L}',np.allclose(total[:,5]-total[:,2],dt,atol=1e-8))
    reps=pd.DataFrame(allrows);reps.to_csv(HERE/'bootstrap_paired_replicates.csv.gz',index=False,compression='gzip')
    for L,g in reps.groupby('block_days'):
        for side in ['long','short','combined']:
            v=g['B_minus_A_'+side];summ.append(dict(block_days=L,replicates=len(g),direction=side,observed_B_minus_A=daily['B_minus_A_'+side].sum(),bootstrap_mean=v.mean(),bootstrap_median=v.median(),percentile95_low=v.quantile(.025),percentile95_high=v.quantile(.975),fraction_B_minus_A_gt0=(v>0).mean(),seed=SEED+L,days=n,method='paired overlapping noncircular moving blocks; truncate final block; no centering'))
    bs=pd.DataFrame(summ);save(bs,'bootstrap_summary.csv');return bs

def cost_stress():
    raw=pd.read_feather(REPO/'inputs/ohlcv_closed_pre2024.feather')
    check('cost data complete pre2024',len(raw)==140255 and bool((raw.date+audit.BAR<=audit.END).all()))
    f=audit.original_features(raw);models=json.loads((REPO/'inputs/selected_models_30.json').read_text(encoding='utf-8'));arr=audit.get_signals(f,models)
    offset=int(np.flatnonzero(f.date>=audit.FORMAL)[0])-2;d=f.iloc[offset:].reset_index(drop=True);gates={s:z['gate'][offset:] for s,z in arr.items()}
    replay_src=(REPO/'inputs/reference_replay.py').read_text(encoding='utf-8')
    scalar_src=(REPO/'inputs/reference_independent_zero_slip.py').read_text(encoding='utf-8')
    rows=[]
    for multiple in [1.,1.5,2.]:
        fee=.0005*multiple;slip=.0002*multiple
        ns={};exec(compile(replay_src,'unchanged_frozen_replay_in_memory','exec'),ns);ns['FEE']=fee;ns['SLIP']=slip
        sc=scalar_src.replace('price*(1-s*0.)',f'price*(1-s*{slip!r})').replace('b.open*(1+s*0.)',f'b.open*(1+s*{slip!r})')
        sc=re.sub(r'(?<![\d.])(?:1\.0005|\.0005)\b',lambda m:repr(1+fee) if m.group()=='1.0005' else repr(fee),sc)
        independent={};exec(compile(sc,'independent_registered_cost_stress','exec'),independent)
        for variant in ['A','B']:
            p=step2.plan_for(d,models,variant);es,xs=audit.schedules_from_table(p)
            risk={s:{z.month:(z.early_loss_pct,z.stop_pct/100) for z in p[p.side==side].itertuples()} for side,s in audit.SIGN.items()}
            t,q=ns['run'](d,es,xs,None,early_exit=True,entry_gate=gates,fixed_account_ratio=10,risk_schedule=risk)
            it,ic=independent['independent'](d,p,'baseline',allow_open=True,entry_gate=gates)
            cl=t[t.status=='closed']
            check(f'cost{multiple} {variant} independent orders',cl[['entry_i','exit_i','side','exit_reason']].values.tolist()==it[['entry_i','exit_i','side','exit_reason']].values.tolist())
            check(f'cost{multiple} {variant} independent equity',np.allclose(q[:,1],ic,rtol=0,atol=1e-11))
            check(f'cost{multiple} {variant} independent fills',np.allclose(cl[['entry_price','exit_price']],it[['entry_price','exit_price']],rtol=0,atol=1e-10))
            # Diagnostic timing is unchanged code; fee affects only equity units.
            td=pd.DataFrame(dict(direction=t.side,entry_time=t.entry_time,exit_bar_open_time=t.exit_bar_time,status=np.where(t.status=='closed','closed','censored/open'),entry_price=t.entry_price,net_realized_pnl=np.where(t.status=='closed',1000*t.simple_2x_pnl_initial,np.nan)))
            oldfee=audit.FEE;audit.FEE=fee
            eq=audit.equity_paths(d,td);audit.FEE=oldfee
            check(f'cost{multiple} {variant} directional mark sum',np.allclose(1+eq.combined_pnl.to_numpy()/1000,q[:,1],rtol=0,atol=1e-11))
            t.to_csv(HERE/f'cost_{multiple:g}_{variant}_trades.csv',index=False,encoding='utf-8-sig')
            eq=eq[eq.time>=audit.FORMAL].copy();eq.to_csv(HERE/f'cost_{multiple:g}_{variant}_equity.csv.gz',index=False,compression='gzip')
            if multiple==1:
                old=pd.read_csv(REPO/f'step2_short_exit_ablation_2020_2023/equity_curve_{variant}.csv')
                check(f'cost1 {variant} Step2 exact baseline',np.allclose(eq[['long_pnl','short_pnl','combined_pnl']],old[['long_pnl','short_pnl','combined_pnl']],rtol=0,atol=1e-8))
            for side in ['long','short','combined']:
                curve=eq[side+'_pnl'].to_numpy();lev=np.r_[0,curve];dd=np.maximum.accumulate(lev)-lev
                day=eq.set_index('time')[side+'_pnl'].resample('1D').last().ffill();delta=day.diff();delta.iloc[0]=day.iloc[0];ret=delta/1000
                g=cl if side=='combined' else cl[cl.side==side];pn=1000*g.simple_2x_pnl_initial
                rows.append(dict(cost_multiplier=multiple,variant=variant,direction=side,fee_per_side=fee,slippage_per_side=slip,net_profit=curve[-1],max_drawdown=dd.max(),profit_factor=pn[pn>0].sum()/(-pn[pn<0].sum()),sharpe=np.sqrt(365)*ret.mean()/ret.std(ddof=1),trades_closed=len(g),open_n=int(t.status.ne('closed').sum())))
            print('COST',multiple,variant,'net',float(1000*(q[-1,1]-1)),flush=True)
    df=pd.DataFrame(rows);save(df,'cost_stress_all.csv')
    ab=df[df.variant=='A'].merge(df[df.variant=='B'],on=['cost_multiplier','direction'],suffixes=('_A','_B'))
    for col in ['net_profit','max_drawdown','profit_factor','sharpe','trades_closed']:ab['B_minus_A_'+col]=ab[col+'_B']-ab[col+'_A']
    save(ab,'cost_stress_comparison.csv');return df

def main():
    original={p.relative_to(REPO).as_posix():digest(p) for directory in [REPO,REPO/'step2_short_exit_ablation_2020_2023'] for p in directory.glob('*') if p.is_file() and p.name!='git_delivery.txt'}
    # Include all versioned frozen input snapshots and manifests in preservation checks.
    for p in (REPO/'inputs').glob('*'):
        if p.is_file():original[p.relative_to(REPO).as_posix()]=digest(p)
    provenance();dct=load_baseline();ranked=concentration(dct);daily=daily_curves(dct);w=windows(dct,daily);bs=bootstrap(daily);cost=cost_stress()
    for name,h in original.items():check('prior artifact unchanged '+name,digest(REPO/name)==h)
    save(pd.DataFrame(CHECKS),'verification.csv')
    meta=dict(task='Step3 sample-selection bias and robustness audit',base_commit='c2cedb8524ebc7c6574c03d41cb0decdbaf086a5',data_only_2020_2023=True,cutoff=str(audit.END),formal_start=str(audit.FORMAL),bootstrap_replicates_per_length=REPS,bootstrap_days=[28,56],bootstrap_seed=SEED,bootstrap_resampling='paired overlapping noncircular moving block daily marked PNL; identical A/B indices; last block truncated',bootstrap_selection_bias_correction=False,cost_multipliers=[1,1.5,2],cost_definition='multiply BOTH fee and slippage per side; full continuous replay with frozen signal thresholds; no new signal rules',common_normal_short_n=len(ranked),A_better_n=int((ranked.comparison=='A_better').sum()),history_includes_2024_cap_selection=True,model_fitting_period='2020-2022',full_parameter_search_attempts='UNKNOWN',original_anchor_logs_available=False,funding=False,independent_cost_checks=True,source_models_unchanged=True,checks_passed=len(CHECKS),step4_started=False,prior_artifact_hashes=original)
    (HERE/'diagnostic_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print('COMPLETED',len(CHECKS),'checks',flush=True)

if __name__=='__main__':main()
