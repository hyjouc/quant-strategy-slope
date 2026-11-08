"""Predeclared single-factor short explicit S-gate ablation, 2020-2023 only.

Imports Step2 helpers; only B short plan.C2 is bypassed. No model/exit changes.
Strictly sequence A -> Step1 reconciliation -> signal gate -> full B replay.
"""
from pathlib import Path
import sys,json,hashlib,subprocess,platform
P=Path(__file__).resolve().parent;BASE=P.parent;STEP2=BASE/'step2_short_exit_ablation_2020_2023';STEP3=BASE/'step3_entry_redundancy_audit_2020_2023'
sys.path.insert(0,str(BASE.parent.parent/'.research_deps'));sys.path.insert(0,str(STEP2))
import run_step2 as step2
import numpy as np,pandas as pd
audit=step2.audit
CHECKS=[]
EXPECTED='9604c90acb727c2ed9c49c5eae94c11e041ddb8c'
EXPECTED_HASHES={'ohlcv_closed_pre2024.feather':'2c6eb56c8db8498263cd6a5e56eb75a8d27fdcf9225003b02ed60a077f112c79','selected_models_30.json':'f7f86d63452bb81a1acbb52a4a5553139443d89582a2d10643adad0db481eba0','ETH_KalmanMomentum_V18C3_TwoConfirm.py':'39b5ee58b32a75c9a775e57fd5dba9bdb67d426393c9bcd90642276c11502fd9','reference_replay.py':'4b805f4068a168d66722a66821c90f9f497ea012942f5aaf6b7c620a2e56c782','reference_independent_zero_slip.py':'930e0cd03912db833aab024648efec79d938ed6e1ed2585101d4ac12ad3a1697'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ck(name,ok):
    CHECKS.append(dict(check=name,passed=bool(ok)));assert bool(ok),'STOP '+name
    print('PASS',name,flush=True)
def save(d,n):
    if not isinstance(d,pd.DataFrame):d=pd.DataFrame(d)
    d.to_csv(P/n,index=False,encoding='utf-8-sig');return d
def plan_for(d,models,variant):
    p=step2.plan_for(d,models,'A') # Step2 B exit experiment is deliberately NOT used.
    if variant=='B':p.loc[p.side.eq('short'),'C2']=-np.inf
    return p
def replay(d,plan,gates,slip):
    entries,exits=audit.schedules_from_table(plan)
    risks={s:{r.month:(r.early_loss_pct,r.stop_pct/100) for r in plan[plan.side.eq(side)].itertuples()} for side,s in audit.SIGN.items()}
    audit.replay.SLIP=slip
    t,q=audit.replay.run(d,entries,exits,None,early_exit=True,entry_gate=gates,fixed_account_ratio=10,risk_schedule=risks)
    return t,q[:,1]
def verify_executor(d,p,gates,v):
    result=None
    for slip,label in [(0.,'zero_slip'),(.0002,'registered_cost')]:
        t,q=replay(d,p,gates,slip);it,iq=step2.independent_variant(d,p,gates,slip)
        closed=t[t.status.eq('closed')]
        ck(v+' '+label+' independent order sequence',closed[['entry_i','exit_i','side','exit_reason']].values.tolist()==it[['entry_i','exit_i','side','exit_reason']].values.tolist())
        ck(v+' '+label+' independent entry fills',np.allclose(closed.entry_price,it.entry_price,rtol=0,atol=1e-10))
        ck(v+' '+label+' independent exit fills',np.allclose(closed.exit_price,it.exit_price,rtol=0,atol=1e-10))
        ck(v+' '+label+' independent full equity',np.allclose(q,iq,rtol=0,atol=1e-11))
        if slip:result=(t,q)
    return result
def diagnostic(t,d,arrays,v):
    # Separate variant-specific diagnostic masks; never alter the score/features.
    masks={s:dict(a) for s,a in arrays.items()}
    if v=='B':
        masks[-1]['tech']=np.delete(arrays[-1]['bits'],1,axis=1).all(axis=1)
        masks[-1]['full']=masks[-1]['tech']&arrays[-1]['gate']
    td=audit.trade_diagnostics(t,d,masks)
    for i,r in enumerate(t.itertuples()):
        e=int(r.entry_i);s=audit.SIGN[r.side]
        td.loc[i,'entry_i']=e;td.loc[i,'exit_i']=int(r.exit_i)
        td.loc[i,'S_pass_first_confirmation']=bool(arrays[s]['bits'][e-2,1]);td.loc[i,'S_pass_second_confirmation']=bool(arrays[s]['bits'][e-1,1])
        td.loc[i,'slope_scaled_first_confirmation']=d.slope_scaled.iloc[e-2];td.loc[i,'slope_scaled_second_confirmation']=d.slope_scaled.iloc[e-1]
        td.loc[i,'requires_explicit_S']=(v=='A' or s==1)
        td.loc[i,'entry_technical_definition']='M/S/A/T' if v=='A' or s==1 else 'M/A/T'
        td.loc[i,'ridge_S_input_unchanged']=True
        td.loc[i,'quality_first_confirmation']=arrays[s]['score'][e-2]
        td.loc[i,'quality_second_confirmation']=arrays[s]['score'][e-1]
    return td
def performance(td,eq):
    a,m=step2.perf(td,eq);a['return_drawdown_ratio']=a.net_profit_marked/a.max_drawdown_currency.replace(0,np.nan)
    exits=pd.to_datetime(td.exit_bar_open_time,utc=True);entries=pd.to_datetime(td.entry_time,utc=True)
    for i,r in m.iterrows():
        mask=td.direction.eq(r.direction) if r.direction!='combined' else np.ones(len(td),bool)
        m.loc[i,'closed_trades']=int((mask&exits.dt.strftime('%Y-%m').eq(r.month)&td.status.eq('closed')).sum())
        m.loc[i,'entries']=int((mask&entries.dt.strftime('%Y-%m').eq(r.month)).sum())
    return a,m
def compare(a,b,keys):
    d=a.merge(b,on=keys,suffixes=('_A','_B'))
    for c in a.columns:
        if c not in keys and pd.api.types.is_numeric_dtype(a[c]):d['B_minus_A_'+c]=d[c+'_B']-d[c+'_A']
    return d
def candidate_masks(f,arrays):
    eligible=(f.decision>=audit.FORMAL).to_numpy();bits=arrays[-1]['bits'];gate=arrays[-1]['gate'];techA=bits.all(axis=1);techB=np.delete(bits,1,axis=1).all(axis=1)
    fullA=techA&gate;fullB=techB&gate;month=f.date.dt.strftime('%Y-%m')
    same=month.eq(month.shift(1))&month.eq((f.date+audit.BAR).dt.strftime('%Y-%m'))
    confirmA=fullA&np.r_[False,fullA[:-1]]&same.to_numpy();confirmB=fullB&np.r_[False,fullB[:-1]]&same.to_numpy()
    addedtech=techB&~techA&eligible;addedfull=fullB&~fullA&eligible;added2=confirmB&~confirmA&eligible
    ck('signal gate added technical 1419',addedtech.sum()==1419);ck('signal gate Ridge rejects 1379',(addedtech&~gate).sum()==1379);ck('signal gate added full 40',addedfull.sum()==40);ck('signal gate added two-confirm candidate bars 24',added2.sum()==24)
    old=pd.read_csv(STEP3/'marginal_signal_counts.csv');row=old[(old.direction=='short')&(old.period=='2020-2023')&(old.layer=='pipeline')&(old.removed_condition=='S')].iloc[0]
    ck('Step3 counts agree with explicit signal reconstruction',row.added_signal_bars==40 and row.new_confirmed_candidate_bars==24)
    return dict(eligible=eligible,techA=techA,techB=techB,fullA=fullA,fullB=fullB,confirmA=confirmA,confirmB=confirmB,addedtech=addedtech,addedfull=addedfull,added2=added2)
def matches(ta,tb):
    keys=['direction','entry_time'];m=ta.merge(tb,on=keys,suffixes=('_A','_B'))
    for c in ['exit_bar_open_time','exit_reason','status','exit_time_precision']:
        ck('shared entries same '+c,m[c+'_A'].fillna('NA').astype(str).equals(m[c+'_B'].fillna('NA').astype(str)))
    for c in ['entry_price','actual_exit_price','net_realized_pnl']:
        ck('shared entries identical '+c,np.allclose(m[c+'_A'],m[c+'_B'],atol=1e-9,rtol=0,equal_nan=True))
    pairsA=set(zip(m.direction,m.entry_time));pairsB=set(pairsA)
    ao=ta[[k not in pairsA for k in zip(ta.direction,ta.entry_time)]].copy();bo=tb[[k not in pairsB for k in zip(tb.direction,tb.entry_time)]].copy()
    save(m,'matched_trade_comparison.csv');save(ao,'A_only_trades.csv');save(bo,'B_only_trades.csv')
    for frame,other,label in [(ao,tb,'A_only'),(bo,ta,'B_only')]:
        for idx,r in frame.iterrows():
            when=pd.Timestamp(r.entry_time);occ=other[(other.entry_time<=when)&((other.exit_bar_open_time>when)|other.status.ne('closed'))]
            frame.loc[idx,'other_variant_occupied_trade_ids']=';'.join(str(int(i)) for i in occ.trade_id)
            frame.loc[idx,'other_variant_occupied_directions']=';'.join(occ.direction)
            frame.loc[idx,'other_variant_occupier_net_pnl_sum']=occ.net_realized_pnl.sum() if len(occ) else np.nan
        save(frame,label+'_trades.csv')
    return m,ao,bo
def candidate_flow(f,cm,tdA,tdB,offset):
    differences=[]
    for i in np.flatnonzero(cm['addedtech']|cm['added2']):
        differences.append(dict(signal_i=i,signal_close_time=f.decision.iloc[i],signal_bar_open_time=f.date.iloc[i],next_open_time=f.date.iloc[i+1] if i+1<len(f) else pd.NaT,added_technical=cm['addedtech'][i],ridge_pass=bool(cm['fullB'][i]),added_full_signal=cm['addedfull'][i],added_two_confirm=cm['added2'][i],original_S_pass=bool(f.slope_scaled.iloc[i]*-1>.49),original_S_value=-f.slope_scaled.iloc[i],month=f.date.iloc[i].strftime('%Y-%m')))
    dif=pd.DataFrame(differences)
    # Signal candidate i requires both i-1 and i free at their closes. A sold
    # candle cannot count. This reconstructs frozen availability, not extra rules.
    details=[]
    for i in np.flatnonzero(cm['added2']):
        row=dict(signal_i=i,signal_close_time=f.decision.iloc[i],hypothetical_entry_time=f.date.iloc[i+1] if i+1<len(f) else pd.NaT)
        for label,td in [('A',tdA),('B',tdB)]:
            reasons=set();ids=set()
            for j in [i-1,i]:
                localj=j-offset
                for t in td.itertuples():
                    if t.entry_i<=localj<t.exit_i:reasons.add('held_during_confirmation');ids.add(int(t.trade_id))
                    elif t.exit_i==localj:reasons.add('sold_candle_confirmation_reset');ids.add(int(t.trade_id))
            actual=td[(td.direction=='short')&(td.entry_time.eq(row['hypothetical_entry_time']))]
            row[label+'_actual_entry']=bool(len(actual));row[label+'_actual_trade_ids']=';'.join(str(int(v)) for v in actual.trade_id);row[label+'_blocked_by_occupancy']=('held_during_confirmation' in reasons);row[label+'_blocking_reasons']=';'.join(sorted(reasons));row[label+'_blocking_trade_ids']=';'.join(map(str,sorted(ids)))
        details.append(row)
    detail=pd.DataFrame(details);save(detail,'confirmed_candidate_conversion.csv')
    if len(detail):dif=dif.merge(detail.drop(columns=['signal_close_time']),on='signal_i',how='left')
    save(dif,'entry_signal_differences.csv')
    rows=[]
    for period in ['2020-2023']+[str(y) for y in range(2020,2024)]:
        pm=np.ones(len(f),bool) if period=='2020-2023' else f.date.dt.year.eq(int(period)).to_numpy()
        dm=np.ones(len(detail),bool) if period=='2020-2023' else pd.to_datetime(detail.signal_close_time,utc=True).dt.year.eq(int(period))
        dd=detail[dm]
        rows.append(dict(period=period,direction='short',added_technical=int((cm['addedtech']&pm).sum()),ridge_rejected_added=int((cm['addedtech']&~cm['fullB']&pm).sum()),added_single_full=int((cm['addedfull']&pm).sum()),added_two_confirm_candidate_bars=int((cm['added2']&pm).sum()),B_actual_openings_at_added_candidates=int(dd.B_actual_entry.sum()),B_candidates_blocked_by_hold=int(dd.B_blocked_by_occupancy.sum()),B_candidates_blocked_by_sold_candle=int(dd.B_blocking_reasons.str.contains('sold_candle',na=False).sum()),A_actual_openings_at_added_candidates=int(dd.A_actual_entry.sum())))
    save(rows,'candidate_flow.csv')
    return detail
def main():
    git=['git','-c','safe.directory='+BASE.as_posix()]
    # At initial run new branch HEAD must exactly match Step3. Reproduction after
    # delivery checks immutable base ancestry and all original artifact hashes.
    head=subprocess.check_output(git+['rev-parse','HEAD'],cwd=BASE,text=True).strip()
    ck('Step3 expected base is ancestor',subprocess.run(git+['merge-base','--is-ancestor',EXPECTED,'HEAD'],cwd=BASE).returncode==0)
    hashes={}
    for n,h in EXPECTED_HASHES.items():
        path=BASE/'inputs'/n;hashes[str(path.relative_to(BASE))]=sha(path);ck('expected hash '+n,sha(path)==h)
        if (STEP2/n).exists():ck('Step2 same hash '+n,sha(STEP2/n)==h)
    ck('unchanged Step1 Step2 Step3 artifacts before experiment',all(sha(folder/n)==v for folder in [BASE,STEP2,STEP3] for n,v in json.loads((folder/'artifact_manifest.json').read_text(encoding='utf-8')).items()))
    raw=pd.read_feather(BASE/'inputs/ohlcv_closed_pre2024.feather');raw=raw[(raw.date>=audit.START)&(raw.date+audit.BAR<=audit.END)].reset_index(drop=True)
    ck('strict closed-bar range 140255',len(raw)==140255 and raw.date.iloc[0]==audit.START and raw.date.iloc[-1]+audit.BAR<=audit.END)
    ck('unique contiguous 15m input',raw.date.is_unique and raw.date.diff().dropna().eq(audit.BAR).all())
    f=audit.original_features(raw);models=json.loads((BASE/'inputs/selected_models_30.json').read_text(encoding='utf-8'));arrays=audit.get_signals(f,models)
    ck('4h availability timestamps causal',bool((f.available_at.dropna()<=f.decision[f.available_at.notna()]).all()))
    off=int(np.flatnonzero(f.date>=audit.FORMAL)[0])-2;d=f.iloc[off:].reset_index(drop=True);da={s:{k:(v[off:].copy() if isinstance(v,np.ndarray) else v) for k,v in a.items()} for s,a in arrays.items()};gates={s:a['gate'] for s,a in da.items()}
    pA=plan_for(d,models,'A');save(pA,'monthly_plan_A.csv')
    print('GATE 1: baseline A independent replay',flush=True)
    a,qa=verify_executor(d,pA,gates,'A');tdA=diagnostic(a,d,da,'A');eqA=step2.eq_diag(d,tdA);annA,monA=performance(tdA,eqA)
    old=pd.read_csv(BASE/'trades_diagnostic.csv');oldEq=pd.read_csv(BASE/'equity_curve.csv')
    for c in ['entry_time','exit_bar_open_time','direction','exit_reason','status']:
        ck('A Step1 exact '+c,tdA[c].fillna('NA').astype(str).tolist()==old[c].fillna('NA').astype(str).tolist())
    for c in ['entry_price','actual_exit_price','net_realized_pnl']:ck('A Step1 fills PNL '+c,np.allclose(tdA[c],old[c],rtol=0,atol=1e-9,equal_nan=True))
    ck('A Step1 full directional curve',np.allclose(eqA.loc[eqA.time>=audit.FORMAL,['long_pnl','short_pnl','combined_pnl']],oldEq[['long_pnl','short_pnl','combined_pnl']],rtol=0,atol=1e-8))
    ck('A Step1 710 closed zero open',len(tdA)==710 and tdA.status.eq('closed').all())
    ck('A Step1 6954.3495 total',abs(annA.query("period=='2020-2023' and direction=='combined'").net_profit_marked.iloc[0]-6954.3495)<1e-3)
    ck('A Step1 2023 -94.9795',abs(annA.query("period=='2023' and direction=='combined'").net_profit_marked.iloc[0]+94.9795)<1e-3)
    print('GATE 1 PASSED. GATE 2: signal differences',flush=True)
    cm=candidate_masks(f,arrays)
    pB=plan_for(d,models,'B');expected=pA.copy();expected.loc[expected.side.eq('short'),'C2']=-np.inf
    ck('B changes only short plan C2',pB.equals(expected));ck('all original exits remain momentum short and sign long',pB.loc[pB.side.eq('short'),'exit_kind'].eq('momentum').all() and pB.loc[pB.side.eq('short'),'exit_value'].eq(.38).all() and pB.loc[pB.side.eq('long'),'exit_kind'].eq('sign').all())
    save(pB,'monthly_plan_B.csv')
    print('GATE 2 PASSED. GATE 3/4: full continuous B replay and independent validation',flush=True)
    b,qb=verify_executor(d,pB,gates,'B');tdB=diagnostic(b,d,da,'B');eqB=step2.eq_diag(d,tdB);annB,monB=performance(tdB,eqB)
    for name,td,eq,q in [('A',tdA,eqA,qa),('B',tdB,eqB,qb)]:
        ck(name+' diagnostic curve matches replay',np.allclose(1+eq.combined_pnl.to_numpy()/1000,q,rtol=0,atol=1e-11))
        ck(name+' actual entry no earlier than formal',bool(td.entry_time.ge(audit.FORMAL).all()))
        ck(name+' no future exit',bool(pd.to_datetime(td.exit_bar_open_time,utc=True).dropna().le(audit.END).all()))
        ck(name+' PNL and last curve reconcile',np.isclose(td.net_realized_pnl.sum()+td.censored_unrealized_net_pnl.sum(),eq.combined_pnl.iloc[-1],atol=1e-8))
        save(td,'trades_A_baseline.csv' if name=='A' else 'trades_B_short_S_bypass.csv');save(eq[eq.time>=audit.FORMAL],'equity_curve_'+name+'.csv')
    m,ao,bo=matches(tdA,tdB);detail=candidate_flow(f,cm,tdA,tdB,off)
    comp=compare(annA,annB,['period','direction']);save(comp,'comparison_period_direction.csv');save(comp[comp.period.isin([str(y) for y in range(2020,2024)])],'annual_comparison.csv');save(compare(monA,monB,['month','direction']),'monthly_comparison.csv')
    exits=[]
    for variant,td in [('A',tdA),('B',tdB)]:
        for (side,reason),g in td.groupby(['direction','exit_group']):exits.append(dict(variant=variant,direction=side,exit_group=reason,n=len(g),net_pnl=g.net_realized_pnl.sum(),mean_holding_hours=g.holding_duration_hours.mean(),mean_MFE_pct=g.MFE_pct.mean(),mean_MAE_pct=g.MAE_pct.mean(),mean_giveback_pct=g.giveback_pct.mean()))
    xe=pd.DataFrame(exits);save(compare(xe[xe.variant.eq('A')].drop(columns='variant'),xe[xe.variant.eq('B')].drop(columns='variant'),['direction','exit_group']),'exit_reason_comparison.csv')
    provenance=[]
    for variant,plan in [('A',pA),('B',pB)]:
        for side in ['long','short']:
            row=plan[plan.side.eq(side)].iloc[0];model=next(z for z in models if z['case']=='unfiltered_'+side)
            for n,v in zip(['C1','C2','C3','C4','exit_value','early_loss_pct','stop_pct'],model['core_parameters']):provenance.append(dict(variant=variant,direction=side,parameter=n,registered_value=v,execution_value=row[n],source='original frozen model core_parameters; Step2 plan_for(A)',meaning='bypass sentinel, not optimized threshold' if variant=='B' and side=='short' and n=='C2' else 'frozen'))
    save(provenance,'parameter_provenance.csv')
    ck('matched plus uniques partitions A B',len(m)+len(ao)==len(tdA) and len(m)+len(bo)==len(tdB))
    ck('newly confirmed candidate conversion accounted',len(detail)==24 and bool((detail.B_actual_entry|detail.B_blocking_reasons.ne('')).all()))
    ck('all original artifacts unchanged after experiment',all(sha(folder/n)==v for folder in [BASE,STEP2,STEP3] for n,v in json.loads((folder/'artifact_manifest.json').read_text(encoding='utf-8')).items()))
    save(CHECKS,'verification.csv')
    meta=dict(task='V19_STEP4_SHORT_S_GATE_ABLATION_2020_2023',base_commit=EXPECTED,initial_head=head,input_hashes=hashes,shared_helper_hashes={str(p.relative_to(BASE)):sha(p) for p in [STEP2/'run_step2.py',STEP2/'run_audit.py',BASE/'inputs/audit_support.py']},bars=len(raw),first_open=str(raw.date.iloc[0]),last_open=str(raw.date.iloc[-1]),last_complete_close=str(f.decision.iloc[-1]),formal_start=str(audit.FORMAL),cutoff=str(audit.END),only_factor='B short explicit C2 comparison bypass via -inf; original 0.49 retained; Ridge S unchanged',fee_per_side=.0005,slippage_per_side=.0002,margin_budget=1000,leverage=2,compound=False,funding=False,annual_restarts=False,models_refit=False,threshold_search=False,bootstrap=False,step5_started=False,verification_count=len(CHECKS),A_trades=len(tdA),B_trades=len(tdB),A_open=int(tdA.status.ne('closed').sum()),B_open=int(tdB.status.ne('closed').sum()),matched=len(m),A_only=len(ao),B_only=len(bo),statistical_status='Retrospective historically selected model; 2023 not independent OOS',python=platform.python_version())
    (P/'diagnostic_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FINAL RESULT',comp[comp.period.eq('2020-2023')][['direction','net_profit_marked_A','net_profit_marked_B','B_minus_A_net_profit_marked','max_drawdown_currency_A','max_drawdown_currency_B']].to_string(index=False),flush=True)
    print('Trade matching',len(m),len(ao),len(bo),flush=True)

if __name__=='__main__':main()
