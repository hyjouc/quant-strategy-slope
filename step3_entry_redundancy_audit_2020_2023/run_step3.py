"""Frozen entry mechanism audit: signal diagnostics only; never calls replay.run.

Reuses Step 1 feature function and scores. No strategy, model or threshold fitting.
Bootstrap uses common moving calendar-day block weights, retaining every signal in
each sampled day/block, rather than treating overlapping signals as independent.
"""
from pathlib import Path
import sys,json,hashlib,itertools,platform
P=Path(__file__).resolve().parent; BASE=P.parent
sys.path.insert(0,str(BASE.parent.parent/'.research_deps'));sys.path.insert(0,str(BASE))
import run_audit as frozen
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import rankdata

SEED=20261108;NBOOT=10000;HS=[1,2,4,8]
CHECKS=[];COMPARISONS=[];OUTCOMES=[]
NAMES=['M','S','A','T'];FN=['M','S','A','EMA4h','vol96','deltaM']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ck(name,ok):
    CHECKS.append(dict(check=name,passed=bool(ok)));assert bool(ok),name
    print('PASS',name,flush=True)
def save(rows,name):
    d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    d.to_csv(P/name,index=False,encoding='utf-8-sig');return d
def residual(x,z):
    if np.ndim(z)==1:z=z[:,None]
    design=np.column_stack([np.ones(len(x)),z])
    return x-design@np.linalg.lstsq(design,x,rcond=None)[0]
def corr(x,y):
    return np.corrcoef(x,y)[0,1] if len(x)>2 and np.std(x)>0 and np.std(y)>0 else np.nan
def partial(x,y,z):
    valid=np.isfinite(x)&np.isfinite(y)&np.isfinite(z).all(axis=1)
    if valid.sum()<10:return np.nan
    return corr(residual(x[valid],z[valid]),residual(y[valid],z[valid]))
def periods(f,eligible):
    yield '2020-2023',eligible
    yield '2020-2022',eligible&(f.date.dt.year.to_numpy()<=2022)
    for y in range(2020,2024):yield str(y),eligible&(f.date.dt.year.to_numpy()==y)
    for y in range(2020,2024):
        for q in range(1,5):yield f'{y}Q{q}',eligible&(f.date.dt.year.to_numpy()==y)&(f.date.dt.quarter.to_numpy()==q)
def summarize(v):
    v=np.asarray(v);v=v[np.isfinite(v)]
    return dict(n=len(v),mean=float(v.mean()) if len(v) else np.nan,median=float(np.median(v)) if len(v) else np.nan,p10=float(np.quantile(v,.1)) if len(v) else np.nan,p90=float(np.quantile(v,.9)) if len(v) else np.nan)
def correlation_rows(x,names,side,per,scope):
    good=np.isfinite(x).all(axis=1);x=x[good];out=[]
    for method in ['pearson','spearman']:
        z=x if method=='pearson' else np.column_stack([rankdata(x[:,j]) for j in range(x.shape[1])])
        cm=np.corrcoef(z,rowvar=False) if len(z)>2 else np.full((len(names),len(names)),np.nan)
        precision=np.linalg.pinv(cm) if np.isfinite(cm).all() else np.full_like(cm,np.nan)
        for i,j in itertools.combinations(range(len(names)),2):
            controls=[k for k in range(len(names)) if k not in [i,j]]
            pc=-precision[i,j]/np.sqrt(precision[i,i]*precision[j,j]) if precision[i,i]>0 and precision[j,j]>0 else np.nan
            out.append(dict(direction=side,period=per,scope=scope,method=method,feature_1=names[i],feature_2=names[j],n=len(x),correlation=cm[i,j],partial_correlation=pc,controls=','.join(names[k] for k in controls)))
    return out
def forwards(f,s):
    # signal i closes at date[i]+15m; hypothetical next open is i+1.
    # h hours contains bars i+1 through i+4h, inclusive; mark at their last close.
    ep=f.open.shift(-1).to_numpy();res={}
    for h in HS:
        k=4*h
        end=f.close.shift(-k).to_numpy()
        future_high=f.high.rolling(k,min_periods=k).max().shift(-k).to_numpy()
        future_low=f.low.rolling(k,min_periods=k).min().shift(-k).to_numpy()
        res[h]={'return':100*s*(end/ep-1),'MFE':np.maximum(0,100*s*((future_high if s==1 else future_low)/ep-1)),'MAE':np.minimum(0,100*s*((future_low if s==1 else future_high)/ep-1))}
        ck(f'{s} horizon {h} terminal missing',bool(np.isnan(res[h]['return'][-k:]).all()))
    return res
def add_outcomes(f,eligible,side,layer,group,mask,fw):
    for per,pm in periods(f,eligible):
        if 'Q' in per:continue
        for h in HS:
            valid=mask&pm&np.isfinite(fw[h]['return'])
            z=fw[h]
            row=dict(direction=side,period=per,layer=layer,group=group,horizon_hours=h,signal_count=int((mask&pm).sum()),excluded_incomplete_horizon=int((mask&pm&~np.isfinite(z['return'])).sum()))
            for name in ['return','MFE','MAE']:
                st=summarize(z[name][valid]);row.update({name+'_'+k:v for k,v in st.items()})
            row['positive_rate']=float((z['return'][valid]>0).mean()) if valid.any() else np.nan
            OUTCOMES.append(row)
def compare(label,side,mask1,mask0,fw,eligible):
    # Difference = group1 minus group0; all horizons and three diagnostics are predeclared.
    for h in HS:
        for metric in ['return','MFE','MAE']:
            COMPARISONS.append(dict(comparison=label,direction=side,horizon_hours=h,metric=metric,mask1=mask1&eligible,mask0=mask0&eligible,values=fw[h][metric],paired=False))

def bootstrap(f,eligible):
    day=((f.decision.dt.floor('D')-pd.Timestamp('2020-01-01',tz='UTC')).dt.days).to_numpy();ndays=int(day.max()+1)
    # Common resamples across comparisons. Continuous non-circular 28/56-day blocks;
    # last drawn block truncated to preserve the exact original calendar length.
    rows=[]
    for weeks in [4,8]:
        length=7*weeks;rng=np.random.default_rng(SEED+weeks);nblocks=int(np.ceil(ndays/length))
        starts=rng.integers(0,ndays-length+1,size=(NBOOT,nblocks));weights=np.zeros((NBOOT,ndays),dtype=np.int16)
        for b in range(nblocks):
            width=min(length,ndays-b*length)
            for offset in range(width):np.add.at(weights,(np.arange(NBOOT),starts[:,b]+offset),1)
        ck(f'{weeks}w bootstrap exact day weight totals',bool((weights.sum(axis=1)==ndays).all()))
        save(pd.DataFrame({'replicate':np.arange(NBOOT),'block_start_day_offsets':[' '.join(map(str,row)) for row in starts]}),f'bootstrap_{weeks}w_draws.csv.gz')
        # Multiple daily summaries share a matrix multiplication, reducing memory and runtime.
        totals=[];valid_info=[]
        for c in COMPARISONS:
            y=c['values'];ms=[]
            for key in ['mask1','mask0']:
                m=c[key]&np.isfinite(y);ms.append(m)
                yy=np.zeros(m.sum()) if c.get('paired_zero_reference') and key=='mask0' else y[m]
                totals.extend([np.bincount(day[m],weights=yy,minlength=ndays),np.bincount(day[m],minlength=ndays)])
            valid_info.append(ms)
        agg=weights.astype(float)@np.asarray(totals).T
        for ci,(c,ms) in enumerate(zip(COMPARISONS,valid_info)):
            g=agg[:,4*ci:4*ci+4];valid=(g[:,1]>0)&(g[:,3]>0)
            dist=g[valid,0]/g[valid,1]-g[valid,2]/g[valid,3]
            y=c['values'];y1=y[ms[0]];y0=np.zeros(ms[1].sum()) if c.get('paired_zero_reference') else y[ms[1]]
            delta=y1.mean()-y0.mean() if len(y1) and len(y0) else np.nan
            signs=[]
            for yr in range(2020,2024):
                ym=f.date.dt.year.to_numpy()==yr
                a=y[ms[0]&ym];b=np.zeros((ms[1]&ym).sum()) if c.get('paired_zero_reference') else y[ms[1]&ym]
                signs.append(float(a.mean()-b.mean()) if len(a) and len(b) else np.nan)
            row={k:v for k,v in c.items() if k not in ['mask1','mask0','values']}
            row.update(block_weeks=weeks,replicates=NBOOT,valid_replicates=int(valid.sum()),n_group1=len(y1),n_group0=len(y0),mean_group1=y1.mean() if len(y1) else np.nan,mean_group0=y0.mean() if len(y0) else np.nan,median_group1=np.median(y1) if len(y1) else np.nan,median_group0=np.median(y0) if len(y0) else np.nan,mean_difference=delta,ci95_low=np.quantile(dist,.025) if len(dist) else np.nan,ci95_high=np.quantile(dist,.975) if len(dist) else np.nan,bootstrap_median_difference=np.median(dist) if len(dist) else np.nan,yearly_positive_differences=sum(x>0 for x in signs),yearly_negative_differences=sum(x<0 for x in signs),yearly_same_sign_as_overall=sum(np.sign(x)==np.sign(delta) for x in signs if np.isfinite(x)))
            row.update({f'difference_{y}':v for y,v in zip(range(2020,2024),signs)});rows.append(row)
        print('Bootstrap finished',weeks,'weeks',len(COMPARISONS),'comparisons',flush=True)
    return save(rows,'bootstrap_stability.csv')

def main():
    manifest=json.loads((BASE/'artifact_manifest.json').read_text(encoding='utf-8'))
    ck('all Step1 input/result artifact hashes',all(sha(BASE/n)==v for n,v in manifest.items()))
    source=BASE/'inputs/ohlcv_closed_pre2024.feather'
    dates=pd.read_feather(source,columns=['date']).date
    ck('date-column-only range audit',bool(((dates>=frozen.START)&(dates+frozen.BAR<=frozen.END)).all()))
    raw=pd.read_feather(source);raw=raw[(raw.date>=frozen.START)&(raw.date+frozen.BAR<=frozen.END)].reset_index(drop=True)
    ck('complete contiguous input 140255 bars',len(raw)==140255 and raw.date.is_unique and raw.date.diff().dropna().eq(frozen.BAR).all())
    f=frozen.original_features(raw);models=json.loads((BASE/'inputs/selected_models_30.json').read_text(encoding='utf-8'));signals=frozen.get_signals(f,models)
    eligible=(f.decision>=frozen.FORMAL).to_numpy()&np.isfinite(f[['momentum4','slope_scaled','model_accel_scaled','trend8']]).all(axis=1).to_numpy()
    ck('formal eligible count matches Step1',eligible.sum()==139297)
    ck('4h features available only after completed 4h close',bool((f.available_at[f.available_at.notna()]<=f.decision[f.available_at.notna()]).all()))
    prefix=frozen.original_features(raw.iloc[:35000]);pg=frozen.get_signals(prefix,models)
    ck('causal feature prefix',np.allclose(prefix[['slope','model_accel','fast','slow']],f.iloc[:35000][['slope','model_accel','fast','slow']],atol=1e-12,rtol=0,equal_nan=True))
    for s in [1,-1]:ck('causal scoring prefix '+str(s),np.allclose(pg[s]['score'],signals[s]['score'][:35000],atol=1e-12,rtol=0))
    correlations=[];patterns=[];conditionals=[];counts=[];distributions=[];ridgecor=[];contributions=[];coefficients=[];independence=[];confrows=[];events=[];provenance=[];perbar=[]
    oldfilters=pd.read_csv(BASE/'signal_filter_summary.csv');oldfilters=oldfilters[oldfilters.period.eq('2020-2023')]
    forward_cache={s:forwards(f,s) for s in [1,-1]}
    for side,s in frozen.SIGN.items():
        sig=signals[s];bits=sig['bits'];tech=sig['tech'];gate=sig['gate'];full=sig['full'];code=bits.astype(int)@np.array([8,4,2,1]);fw=forward_cache[s]
        ind=s*f[['momentum4','slope_scaled','model_accel_scaled','trend8']].to_numpy()
        ff=frozen.feature(f,s)
        perbar.append(pd.DataFrame(dict(open_time=f.date,signal_close_time=f.decision,hypothetical_next_open_time=f.date.shift(-1),hypothetical_next_open_price=f.open.shift(-1),direction=side,eligible=eligible,M=ind[:,0],S=ind[:,1],A=ind[:,2],T=ind[:,3],M_pass=bits[:,0],S_pass=bits[:,1],A_pass=bits[:,2],T_pass=bits[:,3],quality_prediction=sig['score'],quality_pass=gate,pattern=code)))
        old=oldfilters[oldfilters.direction.eq(side)].iloc[0]
        ck(side+' Step1 frozen technical/quality counts',int((tech&eligible).sum())==old.technical_signal_bars and int((tech&gate&eligible).sum())==old.quality_passed_bars)
        for per,pm in periods(f,eligible):
            print('Correlation and counts',side,per,flush=True)
            scopes={'all':pm,'M_S_pass':pm&bits[:,0]&bits[:,1],'all_technical_pass':pm&tech}
            for name,mask in scopes.items():correlations.extend(correlation_rows(ind[mask],NAMES,side,per,name))
            for j in range(4):
                condmask=pm&np.delete(bits,j,axis=1).all(axis=1)
                correlations.extend(correlation_rows(ind[condmask],NAMES,side,per,'other_three_pass_for_'+NAMES[j]))
            for c in range(16):patterns.append(dict(direction=side,period=per,pattern=format(c,'04b'),n=int((pm&(code==c)).sum()),rate=float((pm&(code==c)).sum()/pm.sum())))
            for j,name in enumerate(NAMES):
                for k,other in enumerate(NAMES):
                    den=int((pm&bits[:,k]).sum());conditionals.append(dict(direction=side,period=per,event=name,condition=other,condition_n=den,joint_n=int((pm&bits[:,k]&bits[:,j]).sum()),conditional_probability=float((pm&bits[:,k]&bits[:,j]).sum()/den) if den else np.nan,unconditional_pass_rate=float((pm&bits[:,j]).sum()/pm.sum()),unconditional_reject_rate=float((pm&~bits[:,j]).sum()/pm.sum()),four_pass_rate=float((pm&tech).sum()/pm.sum())))
            for j,name in enumerate(NAMES):
                removed=np.delete(bits,j,axis=1).all(axis=1);added=removed&~bits[:,j]
                month=f.date.dt.strftime('%Y-%m');same=month.eq(month.shift(1))&month.eq((f.date+frozen.BAR).dt.strftime('%Y-%m'))
                base2=full&np.r_[False,full[:-1]]&same.to_numpy();cf=removed&gate;cf2=cf&np.r_[False,cf[:-1]]&same.to_numpy()
                for layer,add in [('technical',added),('pipeline',added&gate)]:
                    counts.append(dict(direction=side,period=per,removed_condition=name,layer=layer,original_signal_bars=int((pm&(tech if layer=='technical' else full)).sum()),added_signal_bars=int((pm&add).sum()),added_rate_all=float((pm&add).sum()/pm.sum()),ridge_rejected_added=int((pm&added&~gate).sum()),base_confirmed_candidate_bars=int((pm&base2).sum()),new_confirmed_candidate_bars=int((pm&cf2&~base2).sum()),new_confirmed_with_failed_current_removed_condition=int((pm&cf2&~base2&added).sum())))
                    for k,n in enumerate(NAMES):
                        st=summarize(ind[pm&add,k]);distributions.append(dict(direction=side,period=per,layer=layer,removed_condition=name,feature=n,**st))
            for scope,mask in [('all',pm),('all_technical_pass',pm&tech)]:ridgecor.extend(correlation_rows(ff[mask],FN,side,per,scope))
        for regime,rm in [('EMA_up',f.fast.to_numpy()>=f.slow.to_numpy()),('EMA_down',f.fast.to_numpy()<f.slow.to_numpy())]:
            correlations.extend(correlation_rows(ind[eligible&rm],NAMES,side,'2020-2023',regime))
        for layer,base in [('technical',tech),('pipeline',full)]:
            add_outcomes(f,eligible,side,layer,'baseline',base,fw)
            for j,name in enumerate(NAMES):
                added=np.delete(bits,j,axis=1).all(axis=1)&~bits[:,j]
                if layer=='pipeline':added&=gate
                add_outcomes(f,eligible,side,layer,'remove_'+name,added,fw)
                compare(layer+'_remove_'+name+'_minus_baseline',side,added,base,fw,eligible)
        add_outcomes(f,eligible,side,'ridge','quality_pass',tech&gate,fw);add_outcomes(f,eligible,side,'ridge','quality_reject',tech&~gate,fw)
        compare('ridge_pass_minus_reject',side,tech&gate,tech&~gate,fw,eligible)
        # Six feature decomposition for all three registered models; the third is
        # contextual filtered-long, never substituted into baseline entry decisions.
        for model in [m for m in models if m['case'].endswith('_'+side)]:
            z=np.nan_to_num(np.clip((ff-np.array(model['mean']))/np.array(model['scale']),-5,5),nan=0,posinf=5,neginf=-5)
            cc=z*np.array(model['coef']);score=cc.sum(axis=1)+model['intercept']
            ck(model['case']+' contribution exact score reconstruction',np.allclose(score,z@np.array(model['coef'])+model['intercept'],atol=1e-15))
            save(pd.DataFrame({'signal_close_time':f.decision,'eligible':eligible,'technical_pass':tech,'score':score,'threshold':model['threshold'],**{FN[j]:cc[:,j] for j in range(6)}}),f"contributions_{model['case']}.csv.gz")
            for j,n in enumerate(FN):coefficients.append(dict(model=model['case'],direction=side,feature=n,coef=model['coef'][j],mean=model['mean'][j],scale=model['scale'][j],intercept=model['intercept'],threshold=model['threshold'],cap=model['cap']))
            for per,pm in periods(f,eligible):
                if 'Q' in per:continue
                for scope,mask in [('all',pm),('all_technical_pass',pm&tech)]:
                    for j,n in enumerate(FN):
                        st=summarize(cc[mask,j]);absst=summarize(abs(cc[mask,j]));contributions.append(dict(model=model['case'],direction=side,period=per,scope=scope,feature=n,**st,mean_abs=absst['mean'],median_abs=absst['median'],p90_abs=absst['p90'],p95_abs=np.quantile(abs(cc[mask,j]),.95) if mask.any() else np.nan))
            if model['case'].startswith('unfiltered'):
                core=cc[:,:3].sum(axis=1);context=cc[:,3]+cc[:,4];delta=cc[:,5]
                parts={'MSA_contribution_nonnegative':core>=0,'EMA_contribution_nonnegative':cc[:,3]>=0,'vol_contribution_nonnegative':cc[:,4]>=0,'EMA_and_vol_contribution_nonnegative':(cc[:,3]>=0)&(cc[:,4]>=0),'deltaM_contribution_nonnegative':delta>=0}
                for label,m in parts.items():
                    add_outcomes(f,eligible,side,'ridge_source',label+'_yes',tech&m,fw);add_outcomes(f,eligible,side,'ridge_source',label+'_no',tech&~m,fw)
                    compare(label+'_yes_minus_no',side,tech&m,tech&~m,fw,eligible)
                # Descriptive linear partial correlations; nuisance projections are
                # not predictive training and do not alter the frozen Ridge model.
                for per,pm in periods(f,eligible):
                    if 'Q' in per:continue
                    for h in HS:
                        y=fw[h]['return'];mask=pm&tech&np.isfinite(y)&np.isfinite(ff).all(axis=1)
                        for label,xx,controls in [('Ridge_beyond_EMA_vol',score,[3,4]),('MSA_beyond_EMA_vol_delta',core,[3,4,5]),('EMA_beyond_MSA_vol_delta',cc[:,3],[0,1,2,4,5]),('vol_beyond_MSA_EMA_delta',cc[:,4],[0,1,2,3,5]),('deltaM_beyond_MSA_EMA_vol',delta,[0,1,2,3,4])]:
                            independence.append(dict(direction=side,period=per,horizon_hours=h,question=label,n=int(mask.sum()),raw_outcome_correlation=corr(xx[mask],y[mask]),partial_outcome_correlation=partial(xx[mask],y[mask],ff[mask][:,controls]),controls=','.join(FN[j] for j in controls),interpretation='descriptive_linear_projection_not_causal_not_refit'))
        # Run-start episodes: one first signal per uninterrupted full-signal run.
        # This is signal-only, independent of actual position availability.
        month=f.date.dt.strftime('%Y-%m').to_numpy();prev=np.r_[False,full[:-1]];newmonth=np.r_[True,month[1:]!=month[:-1]]
        starts=np.flatnonzero(full&(~prev|newmonth)&eligible)
        success=np.zeros(len(f),dtype=bool);failed=np.zeros(len(f),dtype=bool)
        for i in starts:
            reason='confirmed';j=i+1
            if i+2>=len(f):reason='censored_missing_next_open'
            elif month[i]!=month[j] or month[j]!=f.date.iloc[j].__add__(frozen.BAR).strftime('%Y-%m'):reason='month_reset'
            elif not full[j]:reason='second_condition_failed'
            if reason=='confirmed':success[i]=True
            elif reason=='second_condition_failed':failed[i]=True
            events.append(dict(direction=side,first_i=i,first_signal_close_time=f.decision.iloc[i],second_confirmation_close_time=f.decision.iloc[j] if reason=='confirmed' else pd.NaT,hypothetical_entry_time=f.date.iloc[i+2] if reason=='confirmed' else pd.NaT,first_next_open_time=f.date.iloc[i+1] if i+1<len(f) else pd.NaT,reason=reason,wait_minutes=15 if reason=='confirmed' else np.nan,confirmation_price_cost_pct=100*s*(f.open.iloc[i+2]/f.close.iloc[i]-1) if reason=='confirmed' else np.nan))
        add_outcomes(f,eligible,side,'confirmation','first_success',success,fw);add_outcomes(f,eligible,side,'confirmation','first_failed',failed,fw)
        compare('confirmation_first_success_minus_failed',side,success,failed,fw,eligible)
        secondmask=np.r_[False,success[:-1]]
        add_outcomes(f,eligible,side,'confirmation','second_after_success',secondmask,fw)
        failed_secondmask=np.r_[False,failed[:-1]]
        add_outcomes(f,eligible,side,'confirmation','common_clock_failed',failed_secondmask,fw)
        compare('confirmation_common_clock_success_minus_failed',side,secondmask,failed_secondmask,fw,eligible)
        for h in HS:
            for metric in ['return','MFE','MAE']:
                y=fw[h][metric];after=np.r_[y[1:],np.nan];valid=success&np.isfinite(y)&np.isfinite(after)
                # Paired difference per same successful episode; same calendar block.
                COMPARISONS.append(dict(comparison='confirmation_paired_second_minus_first',direction=side,horizon_hours=h,metric=metric,mask1=valid,mask0=valid,values=after-y,paired=True))
                # For paired bootstrap we encode group0 as zero for identical mask.
                COMPARISONS[-1]['paired_zero_reference']=True
        for per,pm in periods(f,eligible):
            es=[e for e in events if e['direction']==side and pm[e['first_i']]]
            confirmed=[e for e in es if e['reason']=='confirmed'];cost=np.array([e['confirmation_price_cost_pct'] for e in confirmed])
            confrows.append(dict(direction=side,period=per,first_events=len(es),confirmed_events=len(confirmed),second_failed=sum(e['reason']=='second_condition_failed' for e in es),month_reset=sum(e['reason']=='month_reset' for e in es),censored=sum(e['reason'].startswith('censored') for e in es),success_rate=len(confirmed)/len(es) if es else np.nan,mean_wait_minutes=15 if confirmed else np.nan,mean_cost_pct=float(cost.mean()) if len(cost) else np.nan,median_cost_pct=float(np.median(cost)) if len(cost) else np.nan,p90_cost_pct=float(np.quantile(cost,.9)) if len(cost) else np.nan))
    save(perbar[0].pipe(lambda _:pd.concat(perbar,ignore_index=True)),'per_bar_features.csv.gz')
    cor=save(correlations,'indicator_correlation.csv');save(patterns,'indicator_pass_patterns.csv');save(conditionals,'indicator_conditional_pass.csv');cnt=save(counts,'marginal_signal_counts.csv');save(distributions,'marginal_feature_distributions.csv')
    outcomes=save(OUTCOMES,'marginal_signal_outcomes.csv');save(outcomes[outcomes.layer.eq('ridge')],'ridge_filter_effect.csv');save(outcomes[outcomes.layer.eq('ridge_source')],'ridge_source_outcomes.csv')
    save(ridgecor,'ridge_feature_correlation.csv');save(contributions,'ridge_contribution.csv');save(coefficients,'ridge_coefficients.csv');save(independence,'ridge_independent_information.csv');save(confrows,'two_confirmation_analysis.csv');save(events,'confirmation_events.csv.gz')
    save(outcomes[outcomes.layer.eq('confirmation')],'confirmation_outcomes.csv')
    save(cor[cor.period.isin([str(y) for y in range(2020,2024)])],'yearly_indicator_analysis.csv');save(cor[cor.period.str.contains('Q')],'quarterly_indicator_analysis.csv')
    bs=bootstrap(f,eligible)
    # Provenance states what can and cannot be established from the supplied frozen records.
    reg=json.loads((BASE/'inputs/original_registration.json').read_text(encoding='utf-8'))
    for case,p in reg['parameters'].items():
        for name,value in zip(reg['parameter_order'],p):provenance.append(dict(component=case,parameter=name,value=value,source='inputs/original_registration.json',selection_period='2020-2022 historical parameter search',independent_oos=False))
    for m in models:
        provenance.append(dict(component=m['case'],parameter='Ridge cap/recipe',value=m['cap'],source='inputs/selected_models_30.json',selection_period='2023-2024 historically used for final recipe selection; no post2023 data used here',independent_oos=False))
        provenance.append(dict(component=m['case'],parameter='Ridge training',value=m['training_trades'],source='inputs/selected_models_30.json',selection_period='2020-2022; last training label exit '+m['last_training_label_exit'],independent_oos=False))
    save(provenance,'parameter_provenance.csv')
    ck('original artifacts unchanged after audit',all(sha(BASE/n)==v for n,v in manifest.items()))
    save(CHECKS,'verification.csv')
    meta=dict(task='V19_STEP3_ENTRY_REDUNDANCY_AUDIT_2020_2023',base_commit='c2cedb8524ebc7c6574c03d41cb0decdbaf086a5',seed=SEED,bootstrap_replicates=NBOOT,block_weeks=[4,8],bootstrap_method='non-circular moving calendar-day blocks; common draws; same-day signals sampled together; truncated final block',bars=len(raw),eligible_bars=int(eligible.sum()),first_open=str(raw.date.iloc[0]),last_open=str(raw.date.iloc[-1]),last_close=str(f.decision.iloc[-1]),formal_start=str(frozen.FORMAL),cutoff=str(frozen.END),source_sha256={n:sha(BASE/'inputs'/n) for n in ['ohlcv_closed_pre2024.feather','selected_models_30.json','ETH_KalmanMomentum_V18C3_TwoConfirm.py']},models=[m['case'] for m in models],filtered_long_role='contextual decomposition only; not baseline gate',forward_entry='next open, no fees/slippage/leverage or position constraints; signal diagnostics only',future_label='close after 4*h complete held bars; MFE/MAE high-low over those same bars',full_strategy_backtests=0,models_refit=0,parameters_optimized=0,step4_started=False,statistical_status='exploratory retrospective, not independent OOS; bootstrap does not remove selection bias or multiplicity',verification_count=len(CHECKS),python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,bootstrap_comparisons=len(COMPARISONS))
    (P/'diagnostic_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Audit computations complete',flush=True)

if __name__=='__main__':main()
