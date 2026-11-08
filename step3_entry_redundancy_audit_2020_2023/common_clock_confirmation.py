"""Remove mechanical confirmation-bar overlap: compare both groups at i+2 open.

Uses the already saved 10000 draws per block length. This adds a diagnostic
comparison, not a strategy arm or a fitted trading condition.
"""
from pathlib import Path
import sys,json
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P));import run_step3 as a
import numpy as np,pandas as pd

def main():
    raw=pd.read_feather(P.parent/'inputs/ohlcv_closed_pre2024.feather');f=a.frozen.original_features(raw);eligible=(f.decision>=a.frozen.FORMAL).to_numpy();ev=pd.read_csv(P/'confirmation_events.csv.gz')
    day=(f.decision.dt.floor('D')-pd.Timestamp('2020-01-01',tz='UTC')).dt.days.to_numpy();ndays=int(day.max()+1);out=[]
    old_o=pd.read_csv(P/'marginal_signal_outcomes.csv');old_bs=pd.read_csv(P/'bootstrap_stability.csv');new=[]
    for side,s in a.frozen.SIGN.items():
        fw=a.forwards(f,s);masks=[]
        for reason in ['confirmed','second_condition_failed']:
            m=np.zeros(len(f),bool);ix=ev[(ev.direction==side)&(ev.reason==reason)].first_i.to_numpy(int)+1;m[ix[ix<len(f)]]=True;masks.append(m)
        a.OUTCOMES.clear();a.add_outcomes(f,eligible,side,'confirmation','common_clock_failed',masks[1],fw);out.extend(a.OUTCOMES)
        for weeks in [4,8]:
            starts=np.array([list(map(int,x.split())) for x in pd.read_csv(P/f'bootstrap_{weeks}w_draws.csv.gz').block_start_day_offsets]);length=7*weeks
            for h in a.HS:
                for metric in ['return','MFE','MAE']:
                    y=fw[h][metric];reps=[];samples=[];validm=[]
                    for mask in masks:
                        m=mask&eligible&np.isfinite(y);validm.append(m);samples.append(y[m])
                        sums=np.bincount(day[m],weights=y[m],minlength=ndays);counts=np.bincount(day[m],minlength=ndays);ps=np.r_[0,np.cumsum(sums)];pc=np.r_[0,np.cumsum(counts)];ss=np.zeros(10000);nn=np.zeros(10000)
                        for k in range(starts.shape[1]):
                            ix=starts[:,k];width=min(length,ndays-k*length);ss+=ps[ix+width]-ps[ix];nn+=pc[ix+width]-pc[ix]
                        reps.append(np.divide(ss,nn,out=np.full(10000,np.nan),where=nn>0))
                    delta=reps[0]-reps[1];delta=delta[np.isfinite(delta)];ys=[]
                    for yr in range(2020,2024):
                        ym=f.date.dt.year.to_numpy()==yr;ys.append(y[validm[0]&ym].mean()-y[validm[1]&ym].mean())
                    diff=samples[0].mean()-samples[1].mean()
                    row=dict(comparison='confirmation_common_clock_success_minus_failed',direction=side,horizon_hours=h,metric=metric,paired=False,block_weeks=weeks,replicates=10000,valid_replicates=len(delta),n_group1=len(samples[0]),n_group0=len(samples[1]),mean_group1=samples[0].mean(),mean_group0=samples[1].mean(),median_group1=np.median(samples[0]),median_group0=np.median(samples[1]),mean_difference=diff,ci95_low=np.quantile(delta,.025),ci95_high=np.quantile(delta,.975),bootstrap_median_difference=np.median(delta),yearly_positive_differences=sum(v>0 for v in ys),yearly_negative_differences=sum(v<0 for v in ys),yearly_same_sign_as_overall=sum(np.sign(v)==np.sign(diff) for v in ys));row.update({f'difference_{y}':v for y,v in zip(range(2020,2024),ys)});new.append(row)
    old_o=old_o[old_o.group.ne('common_clock_failed')];o=pd.concat([old_o,pd.DataFrame(out)],ignore_index=True);o.to_csv(P/'marginal_signal_outcomes.csv',index=False,encoding='utf-8-sig');o[o.layer.eq('confirmation')].to_csv(P/'confirmation_outcomes.csv',index=False,encoding='utf-8-sig')
    old_bs=old_bs[old_bs.comparison.ne('confirmation_common_clock_success_minus_failed')];bs=pd.concat([old_bs,pd.DataFrame(new)],ignore_index=True);bs.to_csv(P/'bootstrap_stability.csv',index=False,encoding='utf-8-sig')
    meta=json.loads((P/'diagnostic_metadata.json').read_text(encoding='utf-8'));meta['bootstrap_comparisons']=int(len(bs)/2);meta['confirmation_common_clock_comparison_added']=True;(P/'diagnostic_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(new).query("metric=='return' and block_weeks==8")[['direction','horizon_hours','mean_difference','ci95_low','ci95_high']].to_string(index=False))

if __name__=='__main__':main()
