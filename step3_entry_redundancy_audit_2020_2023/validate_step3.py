"""Independent alignment, partition and block-resampling checks (no strategy run)."""
from pathlib import Path
import sys,json,hashlib,ast
P=Path(__file__).resolve().parent;B=P.parent
sys.path.insert(0,str(B.parent.parent/'.research_deps'));sys.path.insert(0,str(P));import run_step3 as audit
import numpy as np,pandas as pd

def main():
    checks=[]
    def ck(name,ok):
        assert bool(ok),name
        checks.append(dict(check=name,passed=True));print('PASS',name,flush=True)
    raw=pd.read_feather(B/'inputs/ohlcv_closed_pre2024.feather');f=audit.frozen.original_features(raw)
    models=json.loads((B/'inputs/selected_models_30.json').read_text(encoding='utf-8'));signals=audit.frozen.get_signals(f,models)
    eligible=(f.decision>=audit.frozen.FORMAL).to_numpy()
    ck('last held horizon can only end before cutoff',f.decision.iloc[-1]<=audit.frozen.END and f.decision.iloc[-1]==pd.Timestamp('2023-12-31 23:45',tz='UTC'))
    fw={s:audit.forwards(f,s) for s in [1,-1]}
    for s in [1,-1]:
        for h in audit.HS:
            for i in [1000,12345,len(f)-4*h-1]:
                held=raw.iloc[i+1:i+4*h+1];ep=raw.open.iloc[i+1]
                ck(f'next-open return hand check {s}/{h}/{i}',np.isclose(fw[s][h]['return'][i],100*s*(held.close.iloc[-1]/ep-1),atol=1e-12))
                mfe=max(0,100*s*((held.high.max() if s==1 else held.low.min())/ep-1));mae=min(0,100*s*((held.low.min() if s==1 else held.high.max())/ep-1))
                ck(f'complete future-bar extremes hand check {s}/{h}/{i}',np.isclose(fw[s][h]['MFE'][i],mfe) and np.isclose(fw[s][h]['MAE'][i],mae))
                ck(f'future bar count and exact time {s}/{h}/{i}',len(held)==4*h and held.date.iloc[-1]+audit.frozen.BAR==f.decision.iloc[i]+pd.Timedelta(hours=h))
    ct=pd.read_csv(P/'marginal_signal_counts.csv');pat=pd.read_csv(P/'indicator_pass_patterns.csv',dtype={'pattern':str});bs=pd.read_csv(P/'bootstrap_stability.csv');events=pd.read_csv(P/'confirmation_events.csv.gz')
    for side,s in audit.frozen.SIGN.items():
        sig=signals[s];bits=sig['bits'];tech=sig['tech'];gate=sig['gate']
        ck(side+' 16 pattern partition',pat[(pat.direction==side)&(pat.period=='2020-2023')].n.sum()==eligible.sum())
        for j,n in enumerate(audit.NAMES):
            added=np.delete(bits,j,axis=1).all(axis=1)&~bits[:,j]&eligible
            a=ct[(ct.direction==side)&(ct.period=='2020-2023')&(ct.removed_condition==n)]
            ck(side+' disjoint added '+n,not (added&tech).any())
            ck(side+' technical counts '+n,int(a[a.layer=='technical'].added_signal_bars.iloc[0])==added.sum())
            ck(side+' pipeline counts '+n,int(a[a.layer=='pipeline'].added_signal_bars.iloc[0])==(added&gate).sum())
        ev=events[events.direction==side]
        ck(side+' event reason partition',len(ev)==ev.reason.value_counts().sum())
        for r in ev[ev.reason=='confirmed'].itertuples():
            i=int(r.first_i)
            assert sig['full'][i] and sig['full'][i+1]
            assert pd.Timestamp(r.second_confirmation_close_time)==f.decision.iloc[i+1]
            assert pd.Timestamp(r.hypothetical_entry_time)==f.date.iloc[i+2]
            assert np.isclose(r.confirmation_price_cost_pct,100*s*(f.open.iloc[i+2]/f.close.iloc[i]-1))
        ck(side+' all successful events exact two closes next open and cost',True)
    # Recompute all primary marginal/quality outcome rows from masks and raw bars.
    outcome=pd.read_csv(P/'marginal_signal_outcomes.csv')
    for r in outcome[outcome.layer.isin(['technical','pipeline','ridge'])].itertuples():
        sig=signals[audit.frozen.SIGN[r.direction]];mask=eligible.copy()
        if r.period=='2020-2022':mask&=f.date.dt.year.to_numpy()<=2022
        elif r.period!='2020-2023':mask&=f.date.dt.year.to_numpy()==int(r.period)
        if r.layer=='ridge':mask&=sig['tech']&(sig['gate'] if r.group=='quality_pass' else ~sig['gate'])
        else:
            if r.group=='baseline':mask&=sig['tech']
            else:
                j=audit.NAMES.index(r.group.split('_')[1]);mask&=np.delete(sig['bits'],j,axis=1).all(axis=1)&~sig['bits'][:,j]
            if r.layer=='pipeline':mask&=sig['gate']
        z=fw[audit.frozen.SIGN[r.direction]][r.horizon_hours];valid=mask&np.isfinite(z['return'])
        assert r.return_n==valid.sum()
        for metric in ['return','MFE','MAE']:
            v=z[metric][valid];expected=getattr(r,metric+'_mean')
            assert (not len(v) and pd.isna(expected)) or np.isclose(v.mean(),expected,atol=1e-12),('outcome mean',r)
            expected=getattr(r,metric+'_median');assert (not len(v) and pd.isna(expected)) or np.isclose(np.median(v),expected,atol=1e-12)
    ck('all technical pipeline and Ridge outcome counts means medians recomputed',True)
    # Precision-matrix partial correlation agrees with the residual method.
    x=f.loc[eligible,['momentum4','slope_scaled','model_accel_scaled','trend8']].to_numpy();cm=np.corrcoef(x,rowvar=False);inv=np.linalg.inv(cm)
    for i,j in [(0,1),(1,3),(2,3)]:
        controls=[k for k in range(4) if k not in [i,j]];pc=-inv[i,j]/np.sqrt(inv[i,i]*inv[j,j]);rp=audit.partial(x[:,i],x[:,j],x[:,controls]);ck(f'precision vs residual partial {i}/{j}',np.isclose(pc,rp,atol=1e-10))
    # Independent bootstrap CI reconstruction via cumulative daily sums and saved
    # block starts, avoiding the runner's day-weight-matrix implementation.
    day=(f.decision.dt.floor('D')-pd.Timestamp('2020-01-01',tz='UTC')).dt.days.to_numpy();ndays=int(day.max()+1)
    for weeks in [4,8]:
        draws=pd.read_csv(P/f'bootstrap_{weeks}w_draws.csv.gz');starts=np.array([list(map(int,x.split())) for x in draws.block_start_day_offsets]);length=7*weeks
        ck(f'{weeks}w 10000 complete block draws',len(draws)==10000 and starts.min()>=0 and starts.max()<=ndays-length)
        for label,paired in [('pipeline_remove_A_minus_baseline',False),('confirmation_paired_second_minus_first',True),('confirmation_common_clock_success_minus_failed',False)]:
            y=fw[1][4]['return'];sig=signals[1]
            if label.startswith('pipeline'):
                m1=eligible&np.delete(sig['bits'],2,axis=1).all(axis=1)&~sig['bits'][:,2]&sig['gate'];m0=eligible&sig['full'];z1=y;z0=y
            elif not paired:
                m1=np.zeros(len(f),bool);m0=np.zeros(len(f),bool)
                for reason,mask in [('confirmed',m1),('second_condition_failed',m0)]:
                    ix=events[(events.direction=='long')&(events.reason==reason)].first_i.to_numpy(int)+1;mask[ix]=True
                m1&=eligible;m0&=eligible;z1=y;z0=y
            else:
                ev=events[(events.direction=='long')&(events.reason=='confirmed')];m1=np.zeros(len(f),bool);m1[ev.first_i.to_numpy(int)]=True
                z1=np.r_[y[1:],np.nan]-y;z0=np.zeros(len(y));m1&=np.isfinite(z1);m0=m1.copy()
            outputs=[]
            for mask,z in [(m1,z1),(m0,z0)]:
                mask=mask&np.isfinite(z)
                sums=np.bincount(day[mask],weights=z[mask],minlength=ndays);counts=np.bincount(day[mask],minlength=ndays)
                ps=np.r_[0,np.cumsum(sums)];pc=np.r_[0,np.cumsum(counts)];ss=np.zeros(10000);nn=np.zeros(10000)
                for k in range(starts.shape[1]):
                    width=min(length,ndays-k*length);ix=starts[:,k];ss+=ps[ix+width]-ps[ix];nn+=pc[ix+width]-pc[ix]
                outputs.append(np.divide(ss,nn,out=np.full(10000,np.nan),where=nn>0))
            diff=outputs[0]-outputs[1];diff=diff[np.isfinite(diff)]
            ref=bs[(bs.direction=='long')&(bs.horizon_hours==4)&(bs.metric=='return')&(bs.block_weeks==weeks)&(bs.comparison==label)].iloc[0]
            ck(f'{weeks}w independent CI reconstruction {label}',np.allclose(np.quantile(diff,[.025,.975]),[ref.ci95_low,ref.ci95_high],atol=1e-10,rtol=0))
    ck('no full strategy replay call',not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['run','execute','run_variant'] for n in ast.walk(ast.parse((P/'run_step3.py').read_text(encoding='utf-8')))))
    ck('408 comparisons both block lengths',len(bs)==816 and bs.groupby('block_weeks').size().eq(408).all())
    old=json.loads((B/'artifact_manifest.json').read_text(encoding='utf-8'));ck('all Step1 original artifacts unchanged',all(hashlib.sha256((B/n).read_bytes()).hexdigest()==h for n,h in old.items()))
    step2=B/'step2_short_exit_ablation_2020_2023';old2=json.loads((step2/'artifact_manifest.json').read_text(encoding='utf-8'));ck('all Step2 original artifacts unchanged',all(hashlib.sha256((step2/n).read_bytes()).hexdigest()==h for n,h in old2.items()))
    v=pd.concat([pd.read_csv(P/'verification.csv'),pd.DataFrame(checks)],ignore_index=True).drop_duplicates('check',keep='last');v.to_csv(P/'verification.csv',index=False,encoding='utf-8-sig')
    meta=json.loads((P/'diagnostic_metadata.json').read_text(encoding='utf-8'));meta['verification_count']=len(v);meta['independent_bootstrap_reconstruction']=True;meta['step1_step2_unchanged']=True;(P/'diagnostic_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Validation complete',len(v))

if __name__=='__main__':main()
