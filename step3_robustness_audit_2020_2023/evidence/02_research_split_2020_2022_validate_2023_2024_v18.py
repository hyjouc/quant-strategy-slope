"""Strict three-way split: establish2020-2022, choose frozen gate2023-2024, test2025."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from numba import set_num_threads
from sklearn.linear_model import Ridge
import research_trade_memory_v15 as m
import research_nested_memory_v17 as previous
from seven_parameter_replay import run
from audit_core_2021 import schedules_from_table

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'split_2020_2022_validate_2023_2024_test2025_v18'
CAPS=[.02,.04,.08]


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def anchor(minimum,case):
    set_num_threads(2)
    path=OUT/'anchors'/str(minimum)/(case+'.json')
    if path.exists():
        o=json.loads(path.read_text());assert o['training_end']=='2022-12-31 23:45UTC'
        print('REUSE',minimum,case,flush=True);return
    v=m.r.v
    def metric(a,complete):
        good=((a[complete,:,0]>0)&(a[complete,:,2]>=minimum)).all(axis=0)
        utility=a[:,:,0].sum(axis=0)-10*np.maximum(0,a[:,:,1]-32).sum(axis=0)
        violation=(np.maximum(0,-a[complete,:,0])/1000).sum(axis=0)
        violation+=(np.maximum(0,minimum-a[complete,:,2])/minimum).sum(axis=0)
        return good,utility,np.where(good,1e7+utility,-1e6*violation+utility)
    v.metric=metric
    full=pd.read_feather(v.b.TRAIN)
    train=full[(full.date>='2020-01-10 23:45')&(full.date<'2023-01-01')].reset_index(drop=True)
    key,sign=v.CASES[case]
    calc,cache,complete,years=v.evaluator(train,key,sign,{})
    p,_,meta,steps=v.search(calc,cache,complete,v.b.fresh.eng.opt.INITIAL.copy())
    result=calc([p])[:,0,:]
    dump(path,dict(minimum=minimum,case=case,parameters=p,training_end='2022-12-31 23:45UTC',
        first_training_bar=str(train.date.iloc[0]),last_training_bar=str(train.date.iloc[-1]),search=meta,
        training_qualified=meta['initial_training_qualified'],
        annual=[dict(year=int(year),net_pnl=float(z[0]),dd_pct=float(z[1]),entries=int(z[2])) for year,z in zip(years,result)],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    pd.DataFrame(steps).to_csv(path.with_suffix('.steps.csv'),index=False)
    print('ANCHOR',minimum,case,p,'qualified',meta['initial_training_qualified'],'evaluated',meta['evaluated'],flush=True)


def build_models(full,p,key,sign,shadow):
    clocks=full.date.astype('int64').to_numpy();cut=int(np.searchsorted(clocks,pd.Timestamp('2023-01-01',tz='UTC').value))
    plan=m.r.v.b.schedule(full,key,p,p);v,codes,ps=m.pack(full,plan);base=m.r.v.b.gate(full,key);zero=np.zeros(len(full),bool)
    curve,_,_,_,record=shadow(v,codes,ps,base[1] if sign==1 else zero,base[-1] if sign==-1 else zero)
    original,*_=m.schedule_curve(v,codes,ps,base[1] if sign==1 else zero,base[-1] if sign==-1 else zero)
    np.testing.assert_allclose(curve,original,atol=1e-12)
    x=m.feature(full,sign);entry=record[:,0].astype(int);exit_i=record[:,1].astype(int)
    valid=(entry>0)&np.isfinite(x[np.maximum(0,entry-1)]).all(axis=1)&(exit_i<cut)
    assert valid.sum()>=50
    raw=x[entry[valid]-1];mean=raw.mean(axis=0);scale=raw.std(axis=0);scale[scale<1e-8]=1.
    xx=np.clip((raw-mean)/scale,-5,5);allx=np.nan_to_num(np.clip((x-mean)/scale,-5,5),nan=0.,posinf=5.,neginf=-5.)
    gates={'none':np.ones(len(full),bool)};models=[]
    for cap in CAPS:
        y=np.clip(record[valid,3],-cap,cap)
        model=Ridge(alpha=100.).fit(xx,y);threshold=float(np.median(model.predict(xx)))
        name=str(cap)
        gates[name]=allx@model.coef_+float(model.intercept_)>=threshold
        models.append(dict(recipe=name,cap=cap,coef=model.coef_.tolist(),intercept=float(model.intercept_),threshold=threshold,
            mean=mean.tolist(),scale=scale.tolist(),training_trades=int(valid.sum()),
            last_training_label_exit=str(full.date.iloc[int(exit_i[valid].max())]),core_parameters=p))
    return gates,models


def main_report():
    set_num_threads(2)
    full=pd.read_feather(m.r.v.b.TRAIN)
    full=full[(full.date>='2020-01-10 23:45')&(full.date<'2026-01-01')].reset_index(drop=True)
    before=full[full.date<'2023-01-01'].reset_index(drop=True)
    vm=(full.date>='2023-01-01')&(full.date<'2025-01-01');tm=(full.date>='2025-01-01')
    validation=full[vm].reset_index(drop=True);test=full[tm].reset_index(drop=True)
    assert len(test)==35040 and len(validation)==70176
    assert full.date.diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all()
    # Keep generated research helpers in this new experiment directory.
    m.OUT=OUT
    shadow=m.shadow_engine();scalar=m.scalar_engine()
    params={};gates={};choices={};selected=[];scores=[];modelrows=[];audit=[];sources={}
    for minimum in [30,75]:
        params[minimum]={};gates[minimum]={};choices[minimum]={}
        for case,(key,sign) in m.r.v.CASES.items():
            path=OUT/'anchors'/str(minimum)/(case+'.json');o=json.loads(path.read_text())
            assert pd.Timestamp(o['last_training_bar'])<pd.Timestamp('2023-01-01',tz='UTC')
            p=o['parameters'];params[minimum][case]=p;sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            options,models=build_models(full,p,key,sign,shadow)
            # Establish model directly from data truncated at2023; all later gates must use these coefficients.
            _,prefixmodels=build_models(before,p,key,sign,shadow)
            for a,b in zip(models,prefixmodels):
                assert a==b
            n=int(np.searchsorted(full.date.astype('int64').to_numpy(),pd.Timestamp('2025-01-01',tz='UTC').value))
            prefix_options,prefixmodels=build_models(full.iloc[:n].copy(),p,key,sign,shadow)
            for name in options:np.testing.assert_array_equal(options[name][:n],prefix_options[name])
            modelrows.extend(dict(minimum=minimum,case=case,**z) for z in models)
            gates[minimum][case]=options
            plan=m.r.v.b.schedule(validation,key,p,p);v,codes,ps=m.pack(validation,plan);base=m.r.v.b.gate(validation,key);zero=np.zeros(len(validation),bool)
            results={}
            for name,mask in options.items():
                gate=mask[vm.to_numpy()]
                curve,en,_,_=m.schedule_curve(v,codes,ps,base[1]&gate if sign==1 else zero,base[-1]&gate if sign==-1 else zero)
                results[name]=dict(recipe=name,entries=int(en.sum()),**previous.metrics(curve,validation))
            baseline=results['none'];pool=[]
            for name,z in results.items():
                if name=='none':stable=z['score']
                else:
                    i=CAPS.index(float(name));stable=min(results[str(cap)]['score'] for cap in CAPS[max(0,i-1):min(len(CAPS),i+2)])
                qualified=bool(name!='none' and z['net_pnl']>baseline['net_pnl'] and z['dd_pct']<=baseline['dd_pct'] and stable>baseline['score'])
                z.update(stable_score=stable,qualified=qualified)
                scores.append(dict(minimum=minimum,case=case,**z))
                if qualified:pool.append(z)
            winner=max(pool,key=lambda z:(z['stable_score'],-abs(float(z['recipe'])-.04))) if pool else baseline
            choices[minimum][case]=winner['recipe']
            selected.append(dict(minimum=minimum,case=case,selected=winner['recipe'],validation_net=winner['net_pnl'],validation_dd=winner['dd_pct'],baseline_net=baseline['net_pnl'],baseline_dd=baseline['dd_pct'],stable_score=winner['stable_score'],training_qualified=o['training_qualified']))
            audit.append(dict(minimum=minimum,case=case,anchor_cutoff=True,model_exact_from2020_2022=True,model_and_gate_prefix=True,core_parameters_frozen=True))
            print('VALIDATION CHOICE',selected[-1],flush=True)
    pd.DataFrame(selected).to_csv(OUT/'selected_rules.csv',index=False)
    dump(OUT/'validation_scores.json',scores);dump(OUT/'frozen_models.json',modelrows)
    dump(OUT/'locked_selection.json',dict(chosen=choices,training_end='2022-12-31 23:45UTC',selection_end='2024-12-31 23:45UTC',test_start='2025-01-01UTC',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),selection_sha256=hashlib.sha256((OUT/'selected_rules.csv').read_bytes()).hexdigest()))
    summary=[];monthly=[];annual=[];verification=[];validation_combined=[]
    for minimum in [30,75]:
        for key,(lc,sc) in m.r.v.ALIASES.items():
            lp=params[minimum][lc];sp=params[minimum][sc]
            # Report combined validation descriptively; these combinations never choose recipes.
            for stage,d,mask in [('validation',validation,vm.to_numpy()),('test2025',test,tm.to_numpy())]:
                plan=m.r.v.b.schedule(d,key,lp,sp);v,codes,ps=m.pack(d,plan);base=m.r.v.b.gate(d,key)
                chosen_gates={1:base[1]&gates[minimum][lc][choices[minimum][lc]][mask],-1:base[-1]&gates[minimum][sc][choices[minimum][sc]][mask]}
                for mode,gate in [('original',base),('selected',chosen_gates)]:
                    curve,en,_,_=m.schedule_curve(v,codes,ps,gate[1],gate[-1])
                    es,xs=schedules_from_table(plan);risks={s:{z.month:(z.early_loss_pct,z.stop_pct/100) for z in plan[plan.side==side].itertuples()} for s,side in [(1,'long'),(-1,'short')]}
                    trades,q=run(d,es,xs,None,early_exit=True,entry_gate=gate,fixed_account_ratio=10,risk_schedule=risks)
                    it,iq=scalar(d,plan,key,allow_open=True,entry_gate=gate)
                    np.testing.assert_allclose(curve,q[:,1],atol=1e-12);np.testing.assert_allclose(curve,iq,atol=1e-12)
                    closed=trades[trades.status.eq('closed')]
                    assert it[['entry_i','exit_i','side','exit_reason']].values.tolist()==closed[['entry_i','exit_i','side','exit_reason']].values.tolist()
                    np.testing.assert_allclose(it.pnl,closed.simple_2x_pnl_initial,atol=1e-12)
                    pnl=1000*trades.simple_2x_pnl_initial;trades['fixed1000_net_pnl']=pnl
                    np.testing.assert_allclose(pnl.sum(),1000*(curve[-1]-1),atol=1e-7)
                    profit=pnl[trades.status.eq('closed')];loss=-profit[profit<0].sum()
                    row=dict(minimum=minimum,strategy=key,mode=mode,net_pnl=float(1000*(curve[-1]-1)),dd_pct=m.dd(curve),pf=float(profit[profit>0].sum()/loss) if loss>0 else None,
                        entries=int(en.sum()),closed_trades=len(closed),end_open=int(trades.status.ne('closed').sum()),
                        long_pnl=float(pnl[trades.side.eq('long')].sum()),short_pnl=float(pnl[trades.side.eq('short')].sum()),median_holding_hours=float(trades.holding_hours_bar_start.median()),
                        win_rate=float(profit.gt(0).mean()),long_recipe=choices[minimum][lc],short_recipe=choices[minimum][sc])
                    if stage=='validation':validation_combined.append(row)
                    else:
                        summary.append(row);stem=f'{key}_{minimum}_{mode}'
                        trades.to_csv(OUT/(stem+'_trades.csv.gz'),index=False,compression='gzip')
                        pd.DataFrame(dict(date=d.date,net_pnl=1000*(curve-1))).to_csv(OUT/(stem+'_equity.csv.gz'),index=False,compression='gzip')
                        plan.to_csv(OUT/(stem+'_parameters.csv'),index=False)
                        last=0.
                        for month,ix in d.groupby(d.date.dt.strftime('%Y-%m')).groups.items():
                            end=float(1000*(curve[list(ix)[-1]]-1));monthly.append(dict(minimum=minimum,strategy=key,mode=mode,month=month,net_pnl=end-last));last=end
                        for n in [len(d)//3,2*len(d)//3]:
                            prefix,*_=m.schedule_curve(v[:n],codes[:n],ps,gate[1][:n],gate[-1][:n]);np.testing.assert_allclose(prefix,curve[:n],atol=0)
                    verification.append(dict(stage=stage,minimum=minimum,strategy=key,mode=mode,three_engines_match=True,orders_and_fees=True,no_test_labels=True))
                    print(stage,row,flush=True)
    pd.DataFrame(summary).to_csv(OUT/'summary_2025.csv',index=False);pd.DataFrame(monthly).to_csv(OUT/'monthly_2025.csv',index=False)
    pd.DataFrame(validation_combined).to_csv(OUT/'validation_combined.csv',index=False)
    pd.DataFrame(verification).to_csv(OUT/'ledger_verification.csv',index=False);pd.DataFrame(audit).to_csv(OUT/'model_cutoff_audit.csv',index=False)
    sources[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    sources[str(m.r.v.b.TRAIN)]=hashlib.sha256(Path(m.r.v.b.TRAIN).read_bytes()).hexdigest()
    dump(OUT/'source_hashes.json',sources)
    dump(OUT/'final_verification.json',dict(anchor_cases=8,all_anchor_training_cutoff=True,model_build2020_2022_only=True,models_frozen_through2025=True,selection_only2023_2024=True,core_parameters_frozen=True,independent_ledgers=24,test_ledger_prefix=True,model_gate_and_training_label_prefix=True,full_2025_bars=35040,validation_bars=70176,formal_strategy_unchanged=True,historical_research_not_untouched_OOS=True))


def orchestrate():
    OUT.mkdir(exist_ok=True)
    dump(OUT/'predeclared_method.json',dict(establish='core parameters and frozen Ridge models2020-2022 only, common original959bar warmup',validation='2023-2024; independent direction selection only; all models frozen, no learning in validation',test='2025full year startflat; no model or coreparam updates; pre2025open positions excluded from test',choices=['none']+CAPS,selection='higher2yearnet and no worse2yearDD; adjacentcapworstscore>none; score=net-10*worstannualDD; higheststable thencloserto.04; otherwise retain original, never disable direction',anchors='same3starts multi-round per-axis search,integerconfirmation2,L8,R.01,prior96barstd;perdirection annualprofit and30/75entries,DD>32softpenalty;fallback explicit',accounting='1000margin2xnoncompound,fee.05%slip.02%per side; nofunding/liquidation',partial_period_policy='validation continuously2023-2024;test separatelyflat2025; yearendMTMnotforcedexit',formal_strategy_unchanged=True,historical_research=True))
    def task(pair):
        minimum,case=pair;log=OUT/f'anchor_{minimum}_{case}.log'
        with log.open('w',encoding='utf-8') as handle:
            result=subprocess.run([sys.executable,'-u',__file__,'--anchor',str(minimum),case],stdout=handle,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(str(log))
        return log.read_text(encoding='utf-8')
    pairs=[(minimum,case) for minimum in [30,75] for case in m.r.v.CASES]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(task,pairs):print(result,flush=True)
    main_report()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--anchor',nargs=2);p.add_argument('--report',action='store_true');args=p.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.anchor:anchor(int(args.anchor[0]),args.anchor[1])
    elif args.report:main_report()
    else:orchestrate()
