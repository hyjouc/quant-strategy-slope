"""V18 chronology and quality pipeline, finer C3 mesh with physical band width."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from numba import set_num_threads
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
import research_split_2020_2022_validate_2023_2024_v18 as n
OUT=HERE/'v18_c3_span001'
v=n.m.r.v;eng=v.b.fresh.eng
ORIGINAL_AXIS=eng.axis_search;ORIGINAL_SEARCH=v.search

def region(vv,ok,score,current,min_width=.001):
    safe=ok if ok.any() else np.ones(len(vv),bool)
    tolerance=max(50.,.1*np.ptp(score[safe])) if ok.any() else .01*max(1000.,abs(score[safe].max()))
    mask=safe&(score>=score[safe].max()-tolerance)
    groups=[];i=0
    while i<len(vv):
        if not mask[i]:i+=1;continue
        j=i
        while j+1<len(vv) and mask[j+1] and vv[j+1]-vv[j]==1:j+=1
        if vv[j]-vv[i]>=round(min_width*1000):groups.append(list(range(i,j+1)))
        i=j+1
    return groups,tolerance

def axis(p,dim,calc):
    if dim!=2:return ORIGINAL_AXIS(p,dim,calc)
    current=int(round(p[2]*1000));ticks={int(round(x*1000)) for x in eng.opt.GRIDS[2]}
    ticks.update(current+k for k in range(-100,101));expansions=0
    for refinement in range(3):
        def assess():
            ps=[];vv=[]
            for tick in sorted(ticks):
                q=p.copy();q[2]=tick/1000
                if eng.opt.valid(q):ps.append(q);vv.append(tick)
            ok,scores=eng.metrics(calc(ps));return vv,ok,scores
        vv,ok,scores=assess();best=int(np.argmax(scores))
        while best in [0,len(vv)-1] and expansions<3:
            # Keep original .01 physical boundary expansion, not .001 crawl.
            extra=vv[best]+(-10 if best==0 else 10);q=p.copy();q[2]=extra/1000
            if not eng.opt.valid(q) or extra in ticks:break
            ticks.add(extra);expansions+=1;vv,ok,scores=assess();best=int(np.argmax(scores))
        if refinement<2:
            added={vv[best]+k for k in range(-100,101)}-ticks
            if added:ticks.update(added);continue
        break
    groups,tol=region(vv,ok,scores,current)
    band=[]
    if groups:
        band=max(groups,key=lambda g:(float(np.min(scores[g])),len(g)))
        middle=band[(len(band)-1)//2:len(band)//2+1]
        pick=min(middle,key=lambda i:abs(vv[i]-current))
    else:pick=int(np.argmax(scores))
    q=p.copy();q[2]=vv[pick]/1000
    return q,dict(stable_axis_band=bool(band),band_low=vv[band[0]]/1000 if band else None,
        band_high=vv[band[-1]]/1000 if band else None,feasible=bool(ok[pick]),expansions=expansions,candidates=len(vv),
        c3_step=.001,minimum_physical_width=.001)

def neighbors(p):
    result=[p.copy()]
    for dim in v.DIMS:
        for delta in [-.01,.01]:
            q=p.copy();q[dim]=round(q[dim]+delta,3 if dim==2 else 2)
            if eng.opt.valid(q):result.append(q)
    return result

def search(calc,cache,complete,anchor):
    p,stable,meta,steps=ORIGINAL_SEARCH(calc,cache,complete,anchor)
    audits=[];eligible=[]
    for r in stable:
        center=r['parameters'];tick=int(round(center[2]*1000));vv=[];qs=[]
        for x in range(tick-10,tick+11):
            q=center.copy();q[2]=x/1000
            if eng.opt.valid(q):vv.append(x);qs.append(q)
        good,u,score=v.metric(calc(qs),complete)
        groups,tol=region(vv,good,score,tick)
        # Require a qualified near-optimal physical band containing this center.
        bands=[g for g in groups if vv[g[0]]<=tick<=vv[g[-1]]]
        accepted=bool(good[vv.index(tick)] and bands)
        row=dict(parameters=center,accepted=accepted,minimum_width=.001,tolerance=float(tol),
            band_low=vv[bands[0][0]]/1000 if bands else None,band_high=vv[bands[0][-1]]/1000 if bands else None,
            stable_score=r['stable_score'])
        audits.append(row)
        if accepted:eligible.append(r)
    # stable already sorted by original V18 score and anchor-distance ties.
    chosen=eligible[0]['parameters'] if eligible else p
    meta.update(c3_broad_audits=audits,c3_broad_qualified=bool(eligible),
        initial_training_qualified=bool(eligible),fallback_original_ranking=not bool(eligible),
        c3_step=.001,c3_minimum_band_width=.001,final_neighbor_step=.01)
    return chosen,eligible,meta,steps

def main():
    set_num_threads(2);OUT.mkdir(exist_ok=True)
    protocol=dict(establish='2020-22 core and quality models',validation='2023-24 quality recipe only',test='2025 fixed',
        modifications=['C3 mesh .001, all other axes .01','C3 bands span>=.001, at least two adjacent .001 points',
                       'C3 final neighbors remain physical +/- .01; preserve three decimals of center',
                       'original final shortlist ranking followed by broad C3 eligibility, explicit fallback'],
        unchanged='original3starts,20rounds,30/75minimums,4direction cases,DDsoft penalty,Ridge caps,quality selection,all trading and accounting',
        min_band_rationale='user correction: actual band span .001, distinct from C3 mesh .001 and final neighborhood +/-.01',
        no2026=True,formal_strategy_changed=False,warning='historical exploratory variant, not identical optimizer nor untouched OOS')
    n.dump(OUT/'protocol.json',protocol)
    n.OUT=OUT;eng.axis_search=axis;v.neighbors=neighbors;v.search=search
    # No test outcomes read until every core fit has been saved.
    for minimum in [30,75]:
        for case in v.CASES:n.anchor(minimum,case)
    n.main_report()
    old=ROOT/'split_2020_2022_validate_2023_2024_test2025_v18'
    a=pd.read_csv(old/'summary_2025.csv');a['version']='original_V18'
    b=pd.read_csv(OUT/'summary_2025.csv');b['version']='fine_C3_broad_band'
    pd.concat([a,b]).to_csv(OUT/'comparison_2025.csv',index=False)
    n.dump(OUT/'source_binding.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(n.__file__),ROOT/'research_directional_independent_v5.py',ROOT/'research_fixedstep001_exit2_2021_2024.py',Path(v.b.TRAIN)]})
    print('PASS COMPLETE V18 C3 SPAN001',flush=True)

if __name__=='__main__':main()
