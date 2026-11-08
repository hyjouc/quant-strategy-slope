"""Independent directional coordinate searches; no joint outcome enters selection."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from numba import set_num_threads
import research_online_parameter_experience_v2 as b
import research_champion_challenger_v4 as v4
from seven_parameter_kernel import scan_seven

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'directional_independent_v5_20261019'
CASES = {'unfiltered_long': ('baseline', 1), 'unfiltered_short': ('baseline', -1),
         'filtered_long': ('symmetric', 1), 'filtered_short': ('symmetric', -1)}
ALIASES = {'baseline': ('unfiltered_long', 'unfiltered_short'),
           'short_only': ('unfiltered_long', 'filtered_short'),
           'symmetric': ('filtered_long', 'filtered_short')}
DIMS = [0, 1, 2, 4, 5, 6]
SCALE = b.SCALES[:7]


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    b.dump(path, obj)


def metric(a, complete):
    # Directional half of the former joint 150-entry minimum. DD remains soft.
    good = ((a[complete, :, 0] > 0) & (a[complete, :, 2] >= 75)).all(axis=0)
    utility = a[:, :, 0].sum(axis=0) - 10 * np.maximum(0, a[:, :, 1] - 32).sum(axis=0)
    violation = (np.maximum(0, -a[complete, :, 0]) / 1000).sum(axis=0)
    violation += (np.maximum(0, 75 - a[complete, :, 2]) / 75).sum(axis=0)
    return good, utility, np.where(good, 1e7 + utility, -1e6 * violation + utility)


def evaluator(history, key, sign, annual_cache):
    cutoff = history.date.iloc[-1] + pd.Timedelta(minutes=15)
    blocks = []
    for year, d in history.groupby(history.date.dt.year):
        blocks.append((int(year), d[b.fresh.eng.opt.FIELDS].to_numpy(float),
                       (d.date.dt.year * 12 + d.date.dt.month).to_numpy(np.int64),
                       b.gate(d, key)[sign], year < cutoff.year))
    complete = np.array([z[-1] for z in blocks])
    cache = {}

    def calc(ps):
        # The second half of this legacy-axis adapter is deliberately ignored.
        ps = [tuple(p[:7]) for p in ps]
        missing = list(dict.fromkeys(p for p in ps if p not in cache))
        if missing:
            measures = []
            for year, data, months, mask, full in blocks:
                ac = annual_cache.setdefault(year, {}) if full else {}
                todo = [p for p in missing if p not in ac]
                if todo:
                    a = scan_seven(data, months, np.array(todo, float), sign, mask)
                    a[:, 0] *= 10  # percent of 1000 -> currency
                    ac.update({p: a[i] for i, p in enumerate(todo)})
                measures.append(np.array([ac[p] for p in missing]))
            a = np.array(measures)
            cache.update({p: a[:, i, :] for i, p in enumerate(missing)})
        return np.stack([cache[p] for p in ps], axis=1)

    return calc, cache, complete, [z[0] for z in blocks]


def neighbors(p):
    out = [p.copy()]
    for dim in DIMS:
        for delta in [-.01, .01]:
            q = p.copy(); q[dim] = round(q[dim] + delta, 2)
            if b.fresh.eng.opt.valid(q):
                out.append(q)
    return out


def search(calc, cache, complete, anchor):
    eng = b.fresh.eng
    eng.metrics = lambda a: (metric(a, complete)[0], metric(a, complete)[2])
    starts = [anchor, [.1, -.5, .3, 2, .1, .5, 3], [.4, .5, .8, 2, .2, 1, 5]]
    steps = []; ends = []
    for si, start in enumerate(starts):
        p = start.copy() + start.copy(); seen = {tuple(p[:7])}; unchanged = 0
        reason = 'round_limit'
        order = DIMS if si == 0 else (DIMS[::-1] if si == 1 else DIMS[3:] + DIMS[:3])
        for iteration in range(1, 21):
            before = p[:7].copy()
            for dim in order:
                old = p[:7].copy(); p, meta = eng.axis_search(p, dim, calc)
                steps.append(dict(start=si, round=iteration, axis=dim, before=old,
                                  after=p[:7].copy(), **meta))
            unchanged = unchanged + 1 if p[:7] == before else 0
            if unchanged >= 2:
                reason = 'converged'; break
            if p[:7] != before and tuple(p[:7]) in seen:
                reason = 'cycle'; break
            seen.add(tuple(p[:7]))
        ends.append(dict(start=si, parameters=p[:7], rounds=iteration, reason=reason))
    ps = [list(p) for p in cache]
    a = calc(ps); good, utility, score = metric(a, complete)
    order = sorted(range(len(ps)), key=lambda i: (-score[i], np.mean(abs((np.array(ps[i])-anchor)/SCALE))))
    stable = []; audits = []
    for i in order[:32]:
        ns = neighbors(ps[i]); ng, nu, _ = metric(calc(ns), complete)
        stable_score = min(float(utility[i]), float(np.quantile(nu, .25)))
        fraction = float(ng[1:].mean())
        row = dict(parameters=ps[i], eligible=bool(good[i]), utility=float(utility[i]),
                   neighbor_good_fraction=fraction, stable_score=stable_score)
        audits.append(row)
        if good[i] and fraction >= .5:
            stable.append(row)
    stable.sort(key=lambda r: (-r['stable_score'], np.mean(abs((np.array(r['parameters'])-anchor)/SCALE))))
    # Initial search must provide a literal valid parameter even if training constraints fail.
    initial = stable[0]['parameters'] if stable else ps[order[0]]
    return initial, stable, dict(evaluated=len(cache), endpoints=ends, neighborhoods=audits,
                                initial_training_qualified=bool(stable)), steps


def run(case):
    set_num_threads(2)
    key, sign = CASES[case]; side = 'long' if sign == 1 else 'short'
    full = pd.read_feather(b.TRAIN)
    months = sorted(full.loc[full.date >= '2024-01-01', 'date'].dt.strftime('%Y-%m').unique())
    ac = {}; bank = {}; live = b.fresh.eng.opt.INITIAL.copy()
    for month in months:
        path = OUT / case / (month + '.json')
        if path.exists():
            obj = json.loads(path.read_text()); live = obj['parameters']; bank = obj['bank_after']; continue
        started = time.monotonic()
        cutoff = pd.Timestamp(month + '-01', tz='UTC')
        history = full[(full.date >= '2020-01-10 23:45') & (full.date < cutoff)]
        calc, cache, complete, years = evaluator(history, key, sign, ac)
        fitted, donors, meta, steps = search(calc, cache, complete, live)
        previous = live.copy()
        if month == '2024-01':
            live = fitted.copy()
        # Admit at most three distinct direction candidates; no opposite-side filtering.
        admitted = []
        for r in donors:
            p = r['parameters']
            if p == live or any(np.mean(abs((np.array(p)-q)/SCALE)) < .03 for q in admitted):
                continue
            admitted.append(p)
            if len(admitted) == 3: break
        for p in [live] + admitted: b.add(bank, side, p, month)
        before = {k: dict(born=r['born'], last_month=r['history'][-1]['month'] if r['history'] else None)
                  for k, r in bank.items()}
        observed = [(cutoff-pd.DateOffset(months=i)).strftime('%Y-%m') for i in [3,2,1]]
        current = bank[side + ':' + b.pid(live)]
        records = [r for r in bank.values() if r['parameters'] != live and r['born'] <= observed[0]
                   and all(any(z['month'] == m for z in r['history']) for m in observed)]
        checks = []; winner = None
        if current['born'] <= observed[0] and records:
            good, utility, _ = metric(calc([r['parameters'] for r in records]), complete)
            order = sorted(np.flatnonzero(good), key=lambda i: -utility[i])[:3]
            for i in order:
                ev = v4.paired_evidence(records[i], current, observed)
                ev.update(candidate_id=records[i]['id'], parameters=records[i]['parameters'])
                checks.append(ev)
                if ev['qualified'] and (winner is None or ev['empirical_mean_lower'] > winner['empirical_mean_lower']):
                    winner = ev
            if winner is not None: live = winner['parameters'].copy()
        # Assertions demonstrate that the opposite half cannot affect this evaluator.
        test = calc([live+live, live+[.9, -.9, .9, 2, .9, .9, 9.]])
        np.testing.assert_array_equal(test[:, 0], test[:, 1])
        selected = calc([live])[:, 0, :]
        d = full[full.date.dt.strftime('%Y-%m').eq(month)].reset_index(drop=True)
        assert d.date.iloc[0] == cutoff and history.date.max() < cutoff
        assert all(r['last_month'] is None or r['last_month'] < month for r in before.values())
        complete_month = d.decision.iloc[-1] == cutoff + pd.offsets.MonthBegin(1)
        records = list(bank.values())
        states, values = b.shadow_month(d[b.fresh.eng.opt.FIELDS].to_numpy(float),
            np.array([r['parameters'] for r in records]), np.array([r['state'] for r in records]),
            sign, b.gate(d, key)[sign])
        for i, r in enumerate(records):
            r['state'] = states[i].tolist()
            if complete_month:
                r['history'].append(dict(month=month, return_pct=float(values[i,0]),
                    dd_points=float(values[i,1]), entries=int(values[i,2]), exits=int(values[i,3])))
        obj = dict(month=month, case=case, previous=previous, parameters=live,
                   changed=month != '2024-01' and live != previous, initialized=month == '2024-01',
                   last_training_bar=str(history.date.max()), partial_month=not complete_month,
                   search=meta, checks=checks, bank_before=before, bank_after=bank,
                   selected_training=[dict(year=y,net=float(a[0]),dd=float(a[1]),entries=int(a[2]))
                                      for y,a in zip(years,selected)], opposite_invariance=True)
        dump(path,obj)
        pd.DataFrame(steps).to_csv(path.with_name(month+'_steps.csv'),index=False)
        print(case,month,'changed',obj['changed'],'searched',meta['evaluated'],'bank',len(bank),
              'seconds',round(time.monotonic()-started,1),flush=True)


def report():
    set_num_threads(2)
    full = pd.read_feather(b.TRAIN); d = full[full.date >= '2024-01-01'].reset_index(drop=True)
    annual=[]; summary=[]; monthly=[]; params=[]; verify=[]; stops=[]
    chains={}
    for case in CASES:
        chains[case]=[json.loads(p.read_text()) for p in sorted((OUT/case).glob('????-??.json'))]
        assert len(chains[case]) == d.date.dt.strftime('%Y-%m').nunique()
        for o in chains[case]:
            assert pd.Timestamp(o['last_training_bar']) < pd.Timestamp(o['month']+'-01',tz='UTC')
            assert o['opposite_invariance']
            params.append(dict(case=case,month=o['month'],changed=o['changed'],
                               parameters=json.dumps(o['parameters']),**dict(zip(b.fresh.eng.opt.NAMES,o['parameters']))))
            stops.extend(dict(case=case,month=o['month'],**z) for z in o['search']['endpoints'])
    for key,(lc,sc) in ALIASES.items():
        plan=pd.concat([b.schedule(d[d.date.dt.strftime('%Y-%m').eq(l['month'])],key,l['parameters'],s['parameters'])
                        for l,s in zip(chains[lc],chains[sc])],ignore_index=True)
        plan.to_csv(OUT/(key+'_schedule.csv'),index=False)
        r,t,e=b.replay(d,plan,key,True)
        summary.append(dict(strategy=key,mode='independent_v5',**r))
        t.to_csv(OUT/(key+'_trades.csv'),index=False);e.to_csv(OUT/(key+'_equity.csv'),index=False)
        old=0.
        for year,ix in d.groupby(d.date.dt.year).groups.items():
            z=e.iloc[list(ix)];net=float(z.net_pnl.iloc[-1]);curve=1000+z.net_pnl.to_numpy()-old
            dd=float(100*(1-curve/np.maximum.accumulate(np.r_[1000,curve])[1:]).max())
            annual.append(dict(strategy=key,mode='independent_v5',year=int(year),net_pnl=net-old,dd_pct=dd,
                               entries=int(t.entry_time.dt.year.eq(year).sum())))
            old=net
        old=0.
        for month,ix in d.groupby(d.date.dt.strftime('%Y-%m')).groups.items():
            net=float(e.net_pnl.iloc[list(ix)[-1]]);monthly.append(dict(strategy=key,month=month,net_pnl=net-old));old=net
        pre=d[d.date<'2025-07-01'].reset_index(drop=True)
        _,_,pe=b.replay(pre,plan[plan.month<'2025-07'],key,True)
        np.testing.assert_allclose(pe.equity,e.equity.iloc[:len(pre)],atol=1e-9)
        verify.append(dict(strategy=key,strict_cutoff=True,opposite_direction_invariant=True,
                           independent_fills_fees=True,equity_prefix=True,no_joint_selection=True))
        print('REPORT',key,flush=True)
    for name,rows in [('summary',summary),('continuous_years',annual),('monthly',monthly),
                      ('parameters',params),('search_stops',stops),('verification',verify)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
    comp=pd.concat([pd.DataFrame(annual),pd.read_csv(v4.OUT/'continuous_years.csv'),
                    pd.read_csv(v4.OUT/'annual_continuity_control_years.csv')],ignore_index=True)
    comp.to_csv(OUT/'comparison.csv',index=False)
    text=['# 多空完全独立优化V5','','多头只用多头交易评价，空头只用空头交易评价，合并回测不参与选参。',
          '2024年初用2020—2023独立搜索初始化；以后每月严格截止前一月，使用2020开始的扩展历史。',
          '每方向6个可调参数，R=.01、L=8、入场2根、正常退出2根固定。总计仍有12个方向参数，并非总自由度减半。',
          '3起点、逐轴0.01、多轮最多20轮、边界扩展；优先相邻稳定区间。初始无合格候选时显式保存回退结果，不停用交易。',
          '完整训练年各方向盈利且至少75次入场，原联合150次拆为方向75次；DD超过32%仍软扣分。方向门槛可能无法满足，照实记录。',
          '每月最多3个候选进入冻结影子库；观察出生后3个完整月，用V4相同方向配对规则晋升，取消联合收益准入。',
          '观察期3个月沿用上一轮2020—2023内部选择，不按本次2024—2026结果再调；本轮没有重做观察期内层选择。',
          '相同门控的方向共享同一独立优化结果：基准多头=上涨禁空多头，上涨禁空空头=对称过滤空头。',
          '合并后仍为一个共享持仓，不同时多空；因此合并实际收益不等于两个单方向影子收益之和。',
          '每笔1000元、2倍、不复利，费用单边.05%、滑点.02%，不含资金费率及强平。连续跨年账本；年度回撤重基1000。',
          '截至2026-10-18 13:00UTC；2024—2026已被观察，是历史复核，非全新样本。正式策略未覆盖。','',
          pd.DataFrame(annual).to_string(index=False),'',pd.DataFrame(summary).to_string(index=False)]
    (OUT/'结果.md').write_text('\n'.join(text),encoding='utf-8')
    dump(OUT/'manifest.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [Path(__file__),b.TRAIN,*OUT.glob('*/*.json')]})
    print(pd.DataFrame(annual).to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',choices=list(CASES));p.add_argument('--report',action='store_true')
    a=p.parse_args();OUT.mkdir(exist_ok=True)
    if a.report:report()
    else:run(a.case)
