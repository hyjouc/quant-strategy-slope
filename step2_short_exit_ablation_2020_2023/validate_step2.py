"""Validate executed normal exits, exact Step1 preservation, and output partitions."""
from pathlib import Path
import sys,json,hashlib,subprocess
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent.parent.parent/'.research_deps'))
import numpy as np,pandas as pd
sys.path.insert(0,str(P));import run_step2 as exp
a=exp.audit

def main():
    records=[]
    def ck(name,ok):
        assert bool(ok),name
        records.append(dict(check=name,passed=True));print('PASS',name,flush=True)
    raw=pd.read_feather(P/'ohlcv_closed_pre2024.feather');f=a.original_features(raw)
    models=json.loads((P/'selected_models_30.json').read_text(encoding='utf-8'))
    for version,file in [('A','trades_A_baseline.csv'),('B','trades_B_sign_exit.csv')]:
        td=pd.read_csv(P/file);normal=td[td.exit_group.eq('normal')]
        for r in normal.itertuples():
            s=a.SIGN[r.direction];ei=int(f.date.searchsorted(pd.Timestamp(r.entry_time)));xi=int(f.date.searchsorted(pd.Timestamp(r.exit_bar_open_time)))
            tail=f.iloc[xi-2:xi];p=next(m for m in models if m['case']=='unfiltered_'+r.direction)['core_parameters']
            assert len(tail)==2 and tail.date.iloc[0]>=pd.Timestamp(r.entry_time)
            assert (s*tail.slope<0).all(),(version,r.trade_id,'second slope confirmation')
            if version=='A' and s==-1:assert (s*tail.momentum4<-.38).all()
            assert (s*f.slope.iloc[max(0,ei-1):xi]>0).any(),'armed missing'
            assert pd.Timestamp(r.actual_exit_time)==tail.decision.iloc[-1]
            assert np.isclose(r.actual_exit_price,f.open.iloc[xi]*(1-s*.0002),atol=1e-9)
        ck(version+' all normal exits keep armed and exactly two completed reversal closes',True)
        ck(version+' normal exits use next open with registered slippage',True)
        ck(version+' confirms complete before each entry',bool((pd.to_datetime(td.second_confirmation_time,utc=True)==pd.to_datetime(td.entry_time,utc=True)).all()))
        ck(version+' no overlapping actual holdings',bool((pd.to_datetime(td.entry_time.iloc[1:],utc=True).reset_index(drop=True)>=pd.to_datetime(td.exit_bar_open_time.iloc[:-1],utc=True).reset_index(drop=True)).all()))
        ck(version+' no invented precise intrabar stop time',bool(td.loc[td.exit_time_precision.eq('within_15m_bar_unknown'),'actual_exit_time'].isna().all()))
        ck(version+' stop-touch extremes excluded',bool(td.loc[td.exit_time_precision.eq('within_15m_bar_unknown'),'mfe_is_lower_bound'].all()))
        eq=pd.read_csv(P/('equity_curve_'+version+'.csv'))
        ck(version+' final closed PNL equals equity',np.isclose(td.net_realized_pnl.sum(),eq.combined_pnl.iloc[-1],atol=1e-8))
    pa=pd.read_csv(P/'monthly_plan_A.csv');pb=pd.read_csv(P/'monthly_plan_B.csv');p=pa.copy();p.loc[p.side.eq('short'),'exit_kind']='sign'
    ck('all monthly schedule fields identical except declared short exit_kind',p.equals(pb))
    ck('both exit_value records preserve .38',bool(pa.loc[pa.side.eq('short'),'exit_value'].eq(.38).all() and pb.loc[pb.side.eq('short'),'exit_value'].eq(.38).all()))
    original=json.loads((P.parent/'artifact_manifest.json').read_text(encoding='utf-8'))
    ck('all Step1 42 artifacts unchanged on disk',all(hashlib.sha256((P.parent/name).read_bytes()).hexdigest()==digest for name,digest in original.items()))
    comp=pd.read_csv(P/'comparison_period_direction.csv');mon=pd.read_csv(P/'monthly_comparison.csv')
    ck('18 period direction comparisons',len(comp)==18)
    ck('144 month direction comparisons',len(mon)==144)
    for side in ['long','short','combined']:
        g=comp[(comp.period=='2020-2023')&(comp.direction==side)].iloc[0]
        for version in ['A','B']:
            ck(version+' monthly reconciliation '+side,np.isclose(mon.loc[mon.direction==side,'net_pnl_marked_'+version].sum(),g[version+'_net_profit_marked'],atol=1e-8))
    v=pd.read_csv(P/'verification.csv');v=pd.concat([v,pd.DataFrame(records)],ignore_index=True);v.drop_duplicates('check',keep='last').to_csv(P/'verification.csv',index=False,encoding='utf-8-sig')
    meta=json.loads((P/'diagnostic_metadata.json').read_text(encoding='utf-8'));meta['normal_exit_execution_checks']=len(records);meta['verification_count']=len(v.drop_duplicates('check'));meta['step1_all_42_artifacts_unchanged']=True;(P/'diagnostic_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Final validation passed',len(records))

if __name__=='__main__':main()
