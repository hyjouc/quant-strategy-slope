"""Validate delivered tables and generate deterministic report/figures; no new arm."""
from pathlib import Path
import json,hashlib,sys,platform
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
import run_step4 as r
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PIL import Image

def read(n):return pd.read_csv(P/(n+'.csv'))
def table(df):
    def fmt(x):
        if isinstance(x,(float,np.floating)):return f'{x:.6f}' if np.isfinite(x) else 'NA'
        return str(x)
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))+'\n'

def main():
    a,b=read('trades_A_baseline'),read('trades_B_short_S_bypass')
    ao,bo=read('A_only_trades'),read('B_only_trades')
    c,m=read('comparison_period_direction'),read('monthly_comparison')
    flow=read('candidate_flow');conv=read('confirmed_candidate_conversion');match=read('matched_trade_comparison')
    eq={v:read('equity_curve_'+v) for v in ['A','B']}
    checks=[]
    def ck(n,x):
        checks.append({'check':n,'passed':bool(x)});assert bool(x),n
        print('PASS',n,flush=True)
    pa,pb=read('monthly_plan_A'),read('monthly_plan_B');expected=pa.copy();expected.loc[expected.side.eq('short'),'C2']=-np.inf
    ck('delivered plans differ only short C2',expected.equals(pb))
    ck('Ridge and source input hashes unchanged',all(r.sha(r.BASE/n)==h for n,h in json.loads((P/'diagnostic_metadata.json').read_text())['input_hashes'].items()))
    raw=pd.read_feather(r.BASE/'inputs/ohlcv_closed_pre2024.feather')
    ck('source contains only allowed closed 2020-2023 bars',raw.date.ge(r.audit.START).all() and (raw.date+r.audit.BAR).le(r.audit.END).all())
    f=r.audit.original_features(raw);models=json.loads((r.BASE/'inputs/selected_models_30.json').read_text(encoding='utf-8'));arr=r.audit.get_signals(f,models)
    off=int(np.flatnonzero(f.date>=r.audit.FORMAL)[0])-2
    for v,t in [('A',a),('B',b)]:
        ok=True
        for z in t.itertuples():
            e=int(z.entry_i)+off;s=r.audit.SIGN[z.direction]
            bits=arr[s]['bits'][e-2:e]
            tech=bits.all() if v=='A' or s==1 else np.delete(bits,1,axis=1).all()
            ok &= bool(tech and arr[s]['gate'][e-2:e].all())
            ok &= bool(np.allclose([z.quality_first_confirmation,z.quality_second_confirmation],arr[s]['score'][e-2:e],atol=1e-12,rtol=0))
            ok &= bool(np.allclose([z.slope_scaled_first_confirmation,z.slope_scaled_second_confirmation],f.slope_scaled.iloc[e-2:e],atol=1e-12,rtol=0))
            ok &= bool(f.date.iloc[e-2].strftime('%Y-%m')==f.date.iloc[e].strftime('%Y-%m'))
        ck(v+' all actual entries original scores S features and two monthly confirmations',ok)
        times=pd.to_datetime(t.entry_time,utc=True);exits=pd.to_datetime(t.exit_bar_open_time,utc=True)
        ck(v+' continuous single position no overlaps',bool((times.iloc[1:].reset_index(drop=True)>=exits.iloc[:-1].reset_index(drop=True)).all()))
        ck(v+' all trade sources precede actual next open',bool((pd.to_datetime(t.entry_time,utc=True)==pd.to_datetime(f.date.iloc[(t.entry_i.astype(int)+off).to_numpy()].to_numpy(),utc=True)).all()))
        for side in ['long','short','combined']:
            row=c[(c.period=='2020-2023')&(c.direction==side)].iloc[0]
            sub=t if side=='combined' else t[t.direction==side]
            ck(v+' '+side+' monthly PNL equals terminal curve and trades',np.isclose(m[m.direction==side]['net_pnl_marked_'+v].sum(),row['net_profit_marked_'+v],atol=1e-8) and np.isclose(sub.net_realized_pnl.sum(),row['net_profit_marked_'+v],atol=1e-8))
            ck(v+' '+side+' monthly closed counts sum',m[m.direction==side]['closed_trades_'+v].sum()==len(sub))
        ck(v+' long plus short full curve equals combined',np.allclose(eq[v].long_pnl+eq[v].short_pnl,eq[v].combined_pnl,atol=1e-8))
    ck('common entries identical exits fills; PNL within 1e-9 USDT floating precision',all(match[x+'_A'].equals(match[x+'_B']) for x in ['exit_bar_open_time','exit_reason','actual_exit_price','entry_price']) and np.allclose(match.net_realized_pnl_A,match.net_realized_pnl_B,rtol=0,atol=1e-9))
    ck('trade partitions and unique PNL bridge',len(match)+len(ao)==len(a) and len(match)+len(bo)==len(b) and np.isclose(bo.net_realized_pnl.sum()-ao.net_realized_pnl.sum(),b.net_realized_pnl.sum()-a.net_realized_pnl.sum()))
    ck('24 candidates partition into 15 actual short entries and 9 held',len(conv)==24 and conv.B_actual_entry.sum()==15 and ((~conv.B_actual_entry)&conv.B_blocked_by_occupancy).sum()==9)
    ck('all 15 direct new shorts included in B-only ledger',set(conv.loc[conv.B_actual_entry,'hypothetical_entry_time'])==set(bo.loc[bo.direction=='short','entry_time']))
    ck('every lost A entry mapped to a B occupying trade',ao.other_variant_occupied_trade_ids.notna().all())
    ck('all previous artifact manifests unchanged',all(r.sha(folder/n)==h for folder in [r.BASE,r.STEP2,r.STEP3] for n,h in json.loads((folder/'artifact_manifest.json').read_text(encoding='utf-8')).items()))
    r.save(checks,'delivery_verification.csv')
    old=read('verification');old=old[~old.check.str.startswith('delivery:')]
    r.save(pd.concat([old,pd.DataFrame([dict(check='delivery:'+x['check'],passed=x['passed']) for x in checks])],ignore_index=True),'verification.csv')
    meta=json.loads((P/'diagnostic_metadata.json').read_text());meta['delivery_verification_count']=len(checks);meta['verification_count']=len(old)+len(checks);meta['versions']={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__}
    (P/'diagnostic_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white','svg.fonttype':'path'})
    figuremeta=[]
    def export(fig,name,desc,source):
        for ext in ['png','svg']:fig.savefig(P/'figures'/(name+'.'+ext),dpi=180,facecolor='white')
        with Image.open(P/'figures'/(name+'.png')) as im:dim=im.size;dpi=im.info.get('dpi')
        figuremeta.append(dict(name=name,description=desc,source=source,pixels=dim,dpi=dpi,transform='No smoothing; complete observations or explicit table aggregation',uncertainty='None; descriptive historical census, no bootstrap'))
        plt.close(fig)
    for side in ['combined','long','short']:
        fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
        for v,color,ls in [('A','#0072B2','-'),('B','#A64B00','--')]:ax.plot(pd.to_datetime(eq[v].time),1000+eq[v][side+'_pnl'],color=color,ls=ls,lw=1,label=v)
        ax.set(title=side.title()+' marked equity (1000 + cumulative PNL)',xlabel='UTC',ylabel='USDT');ax.legend();export(fig,side+'_equity','A/B '+side+' marked equity; baseline 1000 USDT','equity_curve_A/B.csv')
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    for v,color,ls in [('A','#0072B2','-'),('B','#A64B00','--')]:
        q=np.r_[0,eq[v].combined_pnl];dd=np.maximum.accumulate(q)-q
        ax.plot(pd.to_datetime(eq[v].time),-dd[1:],color=color,ls=ls,label=v,lw=1)
    ax.set(title='Combined drawdown below running marked peak',xlabel='UTC',ylabel='USDT (negative)');ax.legend();export(fig,'combined_drawdown','A/B all 15m marked drawdowns, identical worst drawdown','equity_curve_A/B.csv')
    fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,side in zip(axs,['long','short','combined']):
        g=c[(c.direction==side)&c.period.isin(['2020','2021','2022','2023'])];x=np.arange(4)
        ax.bar(x-.18,g.net_profit_marked_A,.36,label='A',color='#0072B2');ax.bar(x+.18,g.net_profit_marked_B,.36,label='B',color='#A64B00',hatch='//');ax.axhline(0,color='black',lw=.6);ax.set(title=side,xlabel='Year',ylabel='Marked PNL (USDT)',xticks=x,xticklabels=g.period);ax.legend()
    export(fig,'annual_returns','Yearly marked PNL per direction, continuous portfolio','annual_comparison.csv')
    fig,axs=plt.subplots(3,1,figsize=(12,8),layout='constrained',sharex=True)
    for ax,side in zip(axs,['long','short','combined']):
        g=m[m.direction==side];x=np.arange(len(g));ax.bar(x,g.B_minus_A_net_pnl_marked,color='#0072B2');ax.axhline(0,color='black',lw=.6);ax.set(title=side,ylabel='B - A (USDT)',xticks=x[::3],xticklabels=g.month.iloc[::3]);ax.tick_params(axis='x',rotation=60)
    export(fig,'monthly_differences','All 48 monthly PNL differences for each direction','monthly_comparison.csv')
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained');labels=['Added technical bars','Added Ridge-pass bars','Added 2-confirm bars','Actual B short trades'];values=[1419,40,24,15]
    ax.barh(labels,values,color='#0072B2');ax.invert_yaxis();ax.set(xlabel='Count (bars in first 3 stages; trades in last)',title='Signal-to-trade funnel: 1379 Ridge rejects; 9 held candidates')
    for i,n in enumerate(values):ax.text(n+10,i,str(n),va='center')
    ax.set_xlim(0,1650);export(fig,'candidate_funnel','Mixed-unit stages explicitly labeled; 24 signals are not trades','candidate_flow.csv')
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained',sharey=True,sharex=True)
    for ax,t,label in [(axs[0],ao,'A-only'),(axs[1],bo,'B-only')]:
        for side,marker,col in [('long','o','#0072B2'),('short','x','#A64B00')]:
            g=t[t.direction==side];ax.scatter(pd.to_datetime(g.entry_time),g.net_realized_pnl,marker=marker,color=col,label=f'{side} n={len(g)}')
        ax.axhline(0,color='black',lw=.6);ax.set(title=label+' actual trades',xlabel='Entry time (UTC)',ylabel='Net PNL (USDT)');ax.legend()
        ax.xaxis.set_major_locator(mdates.YearLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    export(fig,'unique_trade_distribution','Every A-only/B-only trade plotted; entry dates, no bins','A_only_trades.csv; B_only_trades.csv')
    (P/'figure_metadata.json').write_text(json.dumps(figuremeta,ensure_ascii=False,indent=2),encoding='utf-8')
    summary=c[c.period=='2020-2023'];annual=c[c.period.isin(['2020','2021','2022','2023'])]
    cols=['period','direction','net_profit_marked_A','net_profit_marked_B','B_minus_A_net_profit_marked','max_drawdown_currency_A','max_drawdown_currency_B','B_minus_A_max_drawdown_currency']
    mm=m[m.direction=='combined'];nonzero=mm[mm.B_minus_A_net_pnl_marked.abs()>1e-8].copy();nonzero['abs_delta']=nonzero.B_minus_A_net_pnl_marked.abs();nonzero=nonzero.sort_values('abs_delta',ascending=False)
    uniques=pd.DataFrame([dict(arm=label,direction=side,trades=len(g),net_pnl=g.net_realized_pnl.sum()) for label,t in [('A-only',ao),('B-only',bo)] for side in ['long','short','combined'] for g in [t if side=='combined' else t[t.direction==side]]])
    metrics=['net_profit_marked','max_drawdown_currency','return_drawdown_ratio','profit_factor','sharpe','sortino','win_rate','mean_win','mean_loss','trades_closed','mean_holding_hours','median_holding_hours','mean_MFE_pct','mean_MAE_pct','mean_giveback_pct']
    detailed=pd.DataFrame([dict(direction=z.direction,metric=k,A=z[k+'_A'],B=z[k+'_B'],B_minus_A=z['B_minus_A_'+k]) for _,z in summary.iterrows() for k in metrics])
    report='''# V19 Step 4：空头显式 S 门槛单因素消融

任务：`V19_STEP4_SHORT_S_GATE_ABLATION_2020_2023`。结论：**存在小幅历史改善，但证据不足以升级；保留 A，B 仅作为待审阅的简化候选。**

## 冻结范围与执行门禁

A 为原始 V18，不采用 Step 2 B。B 仅在独立月度计划把 short C2 从注册值 0.49 改为 `-inf`，这是绕过显式比较的标记；Ridge 六项输入（包括 S）、系数、标准化和门槛完全保留。原空头 momentum 0.38 退出、armed、两根确认、多头全部规则、止损、早退、单持仓与执行优先级均未更改。

每笔保证金预算 1000 USDT、2 倍杠杆、不复利；单边手续费 0.05%、滑点 0.02%，无资金费率、无年度重启。A/B 分别从头连续回放，没有把候选收益叠加到旧交易表。先复现 A，再核验候选，之后才执行 B；日志保存顺序。A 与 Step 1 的逐笔时间、方向、退出原因、成交价和完整资金曲线一致，710 笔平仓、0 未平仓，总净利润 6954.349451 USDT，2023 年 -94.979541 USDT。

A/B 均通过独立标量执行器的零滑点和原始成本订单顺序、入场价、退出价及完整资金曲线核验。共有交易的退出时间、原因、成交价和净利润完全相同。输入和前期产物哈希前后相同；只新增 Step 4 文件。

## 数据与统计口径

仅读取已验证的 `inputs/ohlcv_closed_pre2024.feather`，140255 根连续 15 分钟完整收盘K线；从 2020-01-01 00:00 UTC 开始预热，正式起点 2020-01-10 23:45 UTC。最后开盘 2023-12-31 23:30，最后完整收盘 23:45；截止 23:59:59 UTC。4 小时 EMA 使用原始实现的已收盘可用时间，已验证因果时间关系。不读取 2024 年及之后行情、指标或交易结果。

**这是历史已选模型的回溯实验，不是独立样本外验证。** 原模型历史选择范围涉及 2023—2024；2023 年也不能标为独立 OOS。这里保留冻结权重，并未读取该时期之后的行情或重训。

净利润/年度/月度收益采用连续组合的收盘盯市变化，跨年不重启；交易胜率、PF、均盈亏、持仓、MFE/MAE及回吐按退出所属区间统计，因此年内交易已实现利润与年内盯市变化可能不同。Sharpe 为固定 1000 USDT 分母的日盯市增量，按 sqrt(365) 年化；Sortino 下行分母为含非负日零值的负收益平方均值平方根，目标收益为零。收益回撤比=区间净利润/区间最大货币回撤；回撤包含区间前基准峰值。MFE/MAE 为完整持有K线的高低价诊断；止损触发的部分持有K线被排除，极值诊断属于下界而非精确盘中路径；回吐为原诊断定义的 MFE 减毛价格退出收益，单位百分点，均不是额外可成交收益。

## 总体与分方向绩效（差值均为 B−A）

'''+table(detailed)+'''
## 年度与合并区间

'''+table(c[cols])+'''
## 候选到成交

'''+table(flow)+'''
1419 根新增技术候选中，1379 根被原 Ridge 拒绝，仅剩 40 根新增完整单根信号，形成 24 根新增两根确认候选。24 是信号K线计数，不是独立事件或交易数；其中 15 根对应 B 实际空头开仓，9 根受已有持仓占用而未开仓。本轮无卖出当根重置阻挡的新增候选。逐根记录见 `confirmed_candidate_conversion.csv`，技术差异与原 S 值保留在 `entry_signal_differences.csv`。

## 实际交易集合与占位效应

共有 698 笔；A 独有 12 笔（全为空头），B 独有 16 笔（15 空头、1 多头）。所以总交易净增加 4 笔，不应把 B 独有 16 笔都称作净新增。相同入场时间及方向的 698 笔退出全部一致。

'''+table(uniques)+f'''
B 独有净利润 {bo.net_realized_pnl.sum():.6f} − A 独有净利润 {ao.net_realized_pnl.sum():.6f} = 组合净改善 {bo.net_realized_pnl.sum()-ao.net_realized_pnl.sum():.6f} USDT。A 独有交易在对应入场时均有 B 空头占位，映射见 `A_only_trades.csv`；这是路径占位证据，不是独立因果质量评分。B 的 15 笔不同空头与原路径 12 笔空头相替换，另释放一个原来没有的多头机会，而该多头最终固定止损 -79.705775 USDT。

## 退出结构

'''+table(read('exit_reason_comparison'))+'''
短空头正常退出 238→239、早退 101→102、固定止损 26→27；多头止损 32→33。规则没有变化，结构变化源于入场和占位路径。正常退出空头利润增加 292.418406 USDT，但空头早退和止损利润分别下降 60.132160 和 83.815280 USDT。共有入场没有退出差异，PnL 浮点误差上限为 1e-9 USDT（观测最大 1.78e-12），未改变成交。

## 月度集中度

'''+f'48 个月中 {len(nonzero)} 个月组合收益有非零差异；改善月 {(mm.B_minus_A_net_pnl_marked>1e-8).sum()}，恶化月 {(mm.B_minus_A_net_pnl_marked < -1e-8).sum()}，其余无变化。下表按绝对差异排列全部变化月（不是只展示改善月份）。\n\n'+table(nonzero[['month','net_pnl_marked_A','net_pnl_marked_B','B_minus_A_net_pnl_marked']])+'''
## Q1—Q8

1. **实际交易增加多少？** 710→714，净增 4 笔；空头 365→368（+3）、多头 345→346（+1）。15 个新增确认候选实际成为空头交易，但替代了 12 个 A 独有空头。
2. **新增交易是否贡献净利润？** B 独有减 A 独有后仅 +68.765190 USDT（约为 A 总净利润的 0.99%）；空头 +148.470965，多头 -79.705775。不能只加总 B 独有盈利。
3. **原空头结构是否改变？** 12 个原入场被不同时间的空头占位；共有入场退出完全一致。正常退出、早退和固定止损各净增 1 笔。
4. **是否挤占更优空头？** 有盈利的 A 交易被占位，不能声称所有替换更优；2022 空头净利润下降 52.749257 USDT。逐笔盈亏与占位编号完整交付，整体空头收益增加并不能保证每个替换都改善。
5. **是否影响多头？** 是。规则相同但路径变化新增一个多头，亏损 79.705775 USDT；原多头均保留。
6. **收益与最大回撤？** 组合净利润 6954.349452→7023.114642，+68.765190；最大回撤 815.535637→815.535637 USDT，未改善。空头回撤 858.608635→832.390445，改善 26.218190；多头回撤不变。
7. **是否集中于少数年份/月？** 是，2020 +120.040520、2021 +81.179701、2022 -132.455032、2023 为零。所有真实变化集中在前述少数月份；2023 无新增完整信号，也没有交易变化，不能算作独立支持证据。
8. **能否保留简化候选？** 可留待负责人评审，但不足以支持策略升级。有效不同空头只有 15 笔，净增仅 3 笔，改善幅度小、年度不一致且多头损失抵消部分空头收益。本轮无 bootstrap、参数搜索或显著性宣称。

## 图表与复现

图表为内部研究展示，无平滑、截尾或统计推断误差条；PNG 与 SVG 同时交付，数据来源和变换见 `figure_metadata.json`。资金曲线为 1000 + 累计盯市净利润，方向曲线各自使用 1000 展示基准，不代表三套并行资金。A/B 高度重叠时 B 虚线覆盖 A 实线。

'''
    for z in figuremeta:report+=f"![{z['description']}](figures/{z['name']}.png)\n\n"
    report+='运行方法见 `README.md`。全部输入哈希见 `diagnostic_metadata.json`；全部交付文件哈希见 `artifact_manifest.json`。完成 Step 4 后停止，未启动 Step 5，未部署或升级冻结策略。\n'
    (P/'step4_report.md').write_text(report,encoding='utf-8')
    (P/'README.md').write_text('''# Step 4 reproducibility

Run from this directory using Python 3.14.3 with numpy, pandas, pyarrow, matplotlib and Pillow. The workspace fallback `.research_deps` is loaded by the runner. Installed versions are recorded in diagnostic_metadata.json.

```powershell
python run_step4.py *> run.log
if ($LASTEXITCODE -ne 0) { throw 'Replay failed; stop' }
python finalize_step4.py *> delivery_validation.log
if ($LASTEXITCODE -ne 0) { throw 'Validation failed; stop' }
python manifest_step4.py
```

Requires unchanged Step 1 inputs, Step 2 helpers and Step 3 artifacts in the parent repository at the declared base ancestry. Only A and B are run. Initial baseline branch/head/remote preflight was performed before branch creation; metadata initial_head records the exact Step 3 SHA. Hash checks protect all original artifacts. Finalization independently reconstructs actual-entry masks/scores, validates table accounting, and renders eight figure sets from the delivered tables. It does not train, search, or replay other experiments.

Manifest excludes itself, caches and .pyc. Commit/remote receipt is outside the hashed experiment to avoid self-referential hashes. Read task_request.md for the full registered protocol and step4_report.md for findings and limitations.
''',encoding='utf-8')
    print('REPORT AND FIGURES COMPLETE',flush=True)

if __name__=='__main__':main()
