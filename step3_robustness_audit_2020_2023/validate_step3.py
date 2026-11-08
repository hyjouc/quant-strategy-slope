"""Independent arithmetic/boundary checks of Step3 delivered diagnostic tables."""
from pathlib import Path
import sys,json,hashlib,math
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent.parent.parent/'.research_deps'))
import numpy as np,pandas as pd

def main():
    checks=[]
    def ck(n,ok):
        assert bool(ok),n
        checks.append(dict(check=n,passed=True));print('PASS',n,flush=True)
    for folder in [P.parent,P.parent/'step2_short_exit_ablation_2020_2023']:
        m=json.loads((folder/'artifact_manifest.json').read_text(encoding='utf-8'))
        ck(folder.name+' all manifest artifacts unchanged',all(hashlib.sha256((folder/n).read_bytes()).hexdigest()==h for n,h in m.items()))
    c=pd.read_csv(P/'common_normal_short_ranked.csv');d=pd.read_csv(P/'concentration_deletions.csv');bs=pd.read_csv(P/'bootstrap_summary.csv');b=pd.read_csv(P/'bootstrap_paired_replicates.csv.gz');daily=pd.read_csv(P/'daily_marked_pnl.csv')
    ck('common positive negative equal counts',((c.comparison=='A_better').sum(),(c.comparison=='B_better').sum(),(c.comparison=='equal').sum())==(58,108,72))
    ck('common PNL advantage equals paired components',np.isclose(c.A_minus_B.sum(),c.net_realized_pnl_A.sum()-c.net_realized_pnl_B.sum(),atol=1e-8))
    for r in d.itertuples():
        ck('paired deletion '+r.deletion,np.isclose(r.remaining_common_A_minus_B,c.A_minus_B.iloc[r.removed_n:].sum(),atol=1e-8) and np.isclose(r.removed_A_minus_B,c.A_minus_B.iloc[:r.removed_n].sum(),atol=1e-8))
    ck('40000 total bootstrap draws',len(b)==40000)
    n=len(daily);cols=[v+'_'+s for v in ['A','B'] for s in ['long','short','combined']]
    x=daily[cols].to_numpy()
    for L in [28,56]:
        rng=np.random.default_rng(20261108+L);k=math.ceil(n/L)
        # Reconstruct the first draw by literal daily index concatenation, separate
        # from the cumulative-sum block-total implementation used in production.
        starts=rng.integers(0,n-L+1,size=(1,k))[0]
        idx=np.concatenate([np.arange(s,s+L) for s in starts])[:n]
        literal=x[idx].sum(axis=0);row=b[(b.block_days==L)&(b.replicate==0)].iloc[0]
        ck(f'block{L} independent literal paired first draw',np.allclose(literal,row[cols].to_numpy(float),rtol=0,atol=1e-8))
        ck(f'block{L} no sampled index outside allowed data',idx.min()>=0 and idx.max()<n and len(idx)==n)
        for side in ['long','short','combined']:
            g=b[b.block_days==L]['B_minus_A_'+side];z=bs[(bs.block_days==L)&(bs.direction==side)].iloc[0]
            ck(f'block{L} percentile and fraction '+side,np.isclose(z.percentile95_low,np.quantile(g,.025)) and np.isclose(z.percentile95_high,np.quantile(g,.975)) and np.isclose(z.fraction_B_minus_A_gt0,(g>0).mean()))
    w=pd.read_csv(P/'time_stability_all_windows.csv')
    ck('all444 windows retained',len(w)==444)
    expected={'annual':4,'quarterly':16,'monthly':48,'rolling6m':43,'rolling12m':37}
    for kind,count in expected.items():ck('window count '+kind,all(len(w[(w.window_type==kind)&(w.direction==side)])==count for side in ['long','short','combined']))
    for variant in ['A','B']:
        for side in ['long','short','combined']:
            # Additive non-overlapping windows must reconcile daily full horizon.
            expected_total=daily[variant+'_'+side].sum()
            for kind in ['annual','quarterly','monthly']:
                ck(variant+' '+side+' '+kind+' marked reconciliation',np.isclose(w.loc[(w.window_type==kind)&(w.direction==side),variant+'_net_profit'].sum(),expected_total,atol=1e-8))
    cost=pd.read_csv(P/'cost_stress_all.csv');ck('only three declared cost scenarios',set(cost.cost_multiplier)=={1.,1.5,2.} and len(cost)==18)
    ck('fees and slippage proportional only',np.allclose(cost.fee_per_side,cost.cost_multiplier*.0005) and np.allclose(cost.slippage_per_side,cost.cost_multiplier*.0002))
    for r in cost.itertuples():
        t=pd.read_csv(P/f'cost_{r.cost_multiplier:g}_{r.variant}_trades.csv');eq=pd.read_csv(P/f'cost_{r.cost_multiplier:g}_{r.variant}_equity.csv.gz')
        g=t if r.direction=='combined' else t[t.side==r.direction]
        curve=eq[r.direction+'_pnl'].to_numpy();levels=np.r_[0,curve]
        ck(f'cost{r.cost_multiplier} {r.variant} '+r.direction+' PNL/DD/trades',np.isclose(g.simple_2x_pnl_initial.sum()*1000,r.net_profit,atol=1e-8) and np.isclose((np.maximum.accumulate(levels)-levels).max(),r.max_drawdown,atol=1e-8) and len(g)==r.trades_closed)
    v=pd.read_csv(P/'verification.csv');v=pd.concat([v,pd.DataFrame(checks)],ignore_index=True).drop_duplicates('check',keep='last');ck('all main checks already passed',v.passed.all());v.to_csv(P/'verification.csv',index=False,encoding='utf-8-sig')
    meta=json.loads((P/'diagnostic_metadata.json').read_text(encoding='utf-8'));meta['independent_output_validation_checks']=len(checks);meta['verification_total']=len(v);(P/'diagnostic_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8');print('VALIDATED',len(checks),'checks; total',len(v))

if __name__=='__main__':main()
