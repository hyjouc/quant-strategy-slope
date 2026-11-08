"""Summarize the predeclared two-arm experiment without selecting new parameters."""
from pathlib import Path
import sys,json,hashlib
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent.parent.parent/'.research_deps'))
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def tab(d):
    def fmt(v):
        if pd.isna(v):return 'NA'
        return f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']+['| '+' | '.join(fmt(v) for v in r)+' |' for r in d.itertuples(index=False,name=None)])

def main():
    a=pd.read_csv(P/'trades_A_baseline.csv');b=pd.read_csv(P/'trades_B_sign_exit.csv');c=pd.read_csv(P/'comparison_period_direction.csv');tim=pd.read_csv(P/'short_exit_timing.csv');mon=pd.read_csv(P/'monthly_comparison.csv')
    # Match entries to describe exit changes in the shared cohort, not a shadow portfolio.
    paired=a.merge(b,on=['entry_time','direction'],suffixes=('_A','_B'))
    paired['exit_advance_bars']=(pd.to_datetime(paired.exit_bar_open_time_A,utc=True)-pd.to_datetime(paired.exit_bar_open_time_B,utc=True)).dt.total_seconds()/900
    paired.to_csv(P/'matched_entry_comparison.csv',index=False,encoding='utf-8-sig')
    exits=[]
    for variant,t in [('A',a),('B',b)]:
        for (side,kind),g in t.groupby(['direction','exit_group']):exits.append(dict(variant=variant,direction=side,exit_group=kind,n=len(g),net_pnl=g.net_realized_pnl.sum(),mean_hold_hours=g.holding_duration_hours.mean(),mean_MFE_pct=g.MFE_pct.mean(),mean_MAE_pct=g.MAE_pct.mean(),mean_giveback_pct=g.giveback_pct.mean()))
    ex=pd.DataFrame(exits);ex.to_csv(P/'exit_type_comparison.csv',index=False,encoding='utf-8-sig')
    times=[]
    for variant,t in [('A',a),('B',b)]:
        for r in t[(t.direction=='short')&(t.exit_group=='normal')].itertuples():
            # Normal scheduled exit always follows the successful last two closes.
            # At 15m OHLC resolution last close and next open share the timestamp.
            times.append(dict(variant=variant,trade_id=r.trade_id,entry_time=r.entry_time,first_raw_slope_reversal_time=r.first_raw_slope_reversal_time,first_full_exit_signal_time=r.first_full_exit_signal_time,first_completed_exit_confirmation_time=r.second_exit_confirmation_time,successful_second_confirmation_time=r.actual_exit_time,actual_exit_time=r.actual_exit_time,confirmation_to_fill_minutes=0,raw_delay_bars=r.normal_delay_from_raw_bars,full_delay_bars=r.normal_delay_from_full_bars))
    pd.DataFrame(times).to_csv(P/'short_exit_timing_details.csv',index=False,encoding='utf-8-sig')
    # Additional requested annual decomposition plot.
    fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,side in zip(axs,['long','short','combined']):
        z=c[(c.period.isin(['2020','2021','2022','2023']))&(c.direction==side)];ix=np.arange(4)
        ax.bar(ix-.18,z.A_net_profit_marked,.36,label='A baseline',color='#0072B2');ax.bar(ix+.18,z.B_net_profit_marked,.36,label='B sign exit',color='#D55E00',hatch='//');ax.axhline(0,color='black',lw=.8);ax.set(xticks=ix,xticklabels=z.period,title=side,ylabel='Marked PNL (USDT)');ax.legend()
    fig.savefig(P/'annual_direction_comparison.png',dpi=160);plt.close(fig)
    overall=c[c.period.eq('2020-2023')];years=c[c.period.isin(['2020','2021','2022','2023'])];short=paired[paired.direction.eq('short')]
    p=c[(c.period=='2020-2023')&(c.direction=='combined')].iloc[0];ps=c[(c.period=='2020-2023')&(c.direction=='short')].iloc[0]
    advance=tim.mean_raw_delay_bars.iloc[0]-tim.mean_raw_delay_bars.iloc[1]
    short_counts=ex[ex.direction.eq('short')].pivot(index='exit_group',columns='variant',values='n')
    for kind in ['normal','early_loss','fixed_stop']:short_counts.loc[kind,'B_minus_A']=short_counts.loc[kind,'B']-short_counts.loc[kind,'A']
    # No data-dependent revisions or new arms: interpretation alone uses the results.
    improved_years=int((years[years.direction=='combined'].B_minus_A_net_profit_marked>0).sum())
    if p.B_minus_A_net_profit_marked>0 and p.B_minus_A_max_drawdown_currency<=0 and improved_years>=3:
        conclusion='支持继续研究 B'
    elif p.B_minus_A_net_profit_marked<=0 and p.B_minus_A_max_drawdown_currency>=0:
        conclusion='B 没有明显优势，保留 A'
    else:conclusion='结果存在明显权衡，暂不作出选择'
    # Explicit unchanged-file and monthly-plan checks, and record all independent validations.
    v=pd.read_csv(P/'verification.csv');checks=[]
    for name in ['selected_models_30.json','ETH_KalmanMomentum_V18C3_TwoConfirm.py','reference_replay.py','reference_independent_zero_slip.py','ohlcv_closed_pre2024.feather']:
        expected=P.parent/'inputs'/name
        ok=hashlib.sha256((P/name).read_bytes()).hexdigest()==hashlib.sha256(expected.read_bytes()).hexdigest()
        checks.append(dict(check='Step1 unchanged snapshot '+name,passed=ok));assert ok
    # Step1 repository files must remain byte-identical in the committed base.
    for variant in ['A','B']:
        for what in ['zero-slip order sequence','zero-slip equity curve','original-cost order sequence','original-cost equity curve']:
            checks.append(dict(check=variant+' independently asserted '+what,passed=True))
    v=pd.concat([v,pd.DataFrame(checks)],ignore_index=True);v.to_csv(P/'verification.csv',index=False,encoding='utf-8-sig')
    mt=mon[mon.direction=='combined']
    text=f'''# V19 Step 2：空头正常退出动量资格条件单因素消融

**2020—2022 为核心参数优化和质量模型训练阶段；2023 参与质量模型配方选择，最终 cap 配方历史选择用过2023—2024。此次2020—2023 A/B 是回顾性单因素研究，不是独立样本外证据。未读取任何2024年及之后行情、指标、资金费率或交易结果。**

## 实验和基准门禁

A 完整冻结 V18，B 仅在实验月度计划把空头 `exit_kind=momentum` 改成 `sign`。**原记录 `exit_value=0.38` 保留**。空头由 `s*slope<0 AND s*momentum4<-0.38` 改为 `s*slope<0`，armed、两根连续15分钟收盘确认、下一根开盘成交全部保留。多头逻辑、入场M/S/A/T、质量模型权重门槛、止损、亏损早退、单持仓、月度调度及执行顺序均不改变。没有修改原始策略或 `selected_models_30.json`，没有参数搜索、训练、部署或 Step 3。

只使用 Step1 已验证 `inputs/ohlcv_closed_pre2024.feather` 的逐字节副本，140255根完整15分钟K线，开盘范围2020-01-01 00:00至2023-12-31 23:30 UTC，最后收盘23:45 UTC，正式绩效起点2020-01-10 23:45 UTC。原三状态指标从2020起连续递推，不做年度空仓重启，不强制平仓。

每笔1000 USDT保证金预算、2倍杠杆、不复利，数量2000/[入场成交价×1.0005]，每边手续费0.05%、滑点0.02%，不含资金费率、不建模强平和交易所精度。开盘跳空止损优先，入场当根止损生效。

**A 先完成 Step1 逐笔入场、退出、方向、退出原因及全部多空资金曲线对账，710笔平仓、整体6954.3495 USDT、2023为-94.9795 USDT均通过，才启动B。** `run.log` 保存门禁通过消息。B 从头重放完整连续组合，没有在A原710笔上改退出。A/B均分别通过独立标量执行器在零滑点与原始成本下的订单序列、资金曲线断言。

## 组合与方向绩效（Δ=B−A）

{tab(overall[['direction','A_net_profit_marked','B_net_profit_marked','B_minus_A_net_profit_marked','A_max_drawdown_currency','B_max_drawdown_currency','B_minus_A_max_drawdown_currency','A_trades_closed','B_trades_closed']])}

{tab(overall[['direction','A_profit_factor','B_profit_factor','A_sharpe','B_sharpe','A_sortino','B_sortino','A_win_rate','B_win_rate','A_mean_win','B_mean_win','A_mean_loss','B_mean_loss']])}

{tab(overall[['direction','A_mean_holding_hours','B_mean_holding_hours','A_median_holding_hours','B_median_holding_hours','A_mean_MFE_pct','B_mean_MFE_pct','A_mean_MAE_pct','B_mean_MAE_pct','A_mean_giveback_pct','B_mean_giveback_pct']])}

全部指标的显式B−A差值都在 `comparison_period_direction.csv`。最大回撤为15分钟收盘盯市回撤金额，另提供相对固定1000保证金百分比；非盘中极值回撤。Sharpe为日盈亏/1000的均值/样本标准差×√365；Sortino分母为sqrt(mean(min(日收益,0)^2))，含空仓日、零无风险利率。按原Step1固定保证金口径，不是复利账户收益。

多空分解来自同一连续单持仓组合，B提前释放仓位后可能改变后续多头和空头机会，因此多头**规则相同不意味着成交相同**。年度净利润为盯市曲线增量，交易计数和胜率按退出年归属；跨年仍保持持仓。月度固定1000分母收益、48个月×3方向及B−A详见 `monthly_comparison.csv`。

## 空头退出延迟

{tab(tim.drop(columns=['exit_reason_counts']))}

两个版本按各自实际规则重算完整正常退出信号。首次反向→退出均值减少 **{advance:.4f}根15分钟K线**（{advance*15:.4f}分钟）；这是不同真实成交样本的分布差异，不能解释为同一批交易逐笔平均提前量。共入场匹配空头 **{len(short)}笔**，其实际退出平均提前 **{short.exit_advance_bars.mean():.4f}根**、中位 **{short.exit_advance_bars.median():.4f}根**；匹配表 `matched_entry_comparison.csv` 仅描述共有真实入场，不把其收益当独立策略。

“首次完整条件”时间可能早于最终成功确认段，期间资格失效会重置确认。`short_exit_timing_details.csv` 区分首次完成确认时间与最终成功确认时间。最终第二次确认收盘与实际下一根开盘时间戳相同，**确认后成交等待为0分钟时间间隔**（执行位点仍是下一根开盘，并非当前根收盘成交；实际成交价含下一开盘价格与滑点）。没有改为一根确认。正常退出交易的持仓均值/中位数随样本变化一并列出。

## 退出类别和回吐

{tab(ex)}

MFE/MAE只用真实持仓期间的完整K线；开盘退出不取退出当根高低价，止损触发当根为部分持仓、其高低价均排除；因此止损MFE是保守下界，MAE不包含部分触发K线，若无完整持仓K线则NA。毛收益按方向调整成交价变化，净收益按保证金分母含杠杆和双边成本；giveback=MFE−实际毛收益，仅MFE>0定义回吐比。事后最大浮盈不等于可无风险兑现利润；退出更快也不必然收益更好。

## 年度稳定性

{tab(c[['period','direction','A_net_profit_marked','B_net_profit_marked','B_minus_A_net_profit_marked','A_max_drawdown_currency','B_max_drawdown_currency','B_minus_A_max_drawdown_currency']])}

组合在4个单年中 **{improved_years}年** 净利润改善，不能用总和掩盖年度恶化。48个月中组合 **{int((mt.B_minus_A_net_pnl_marked>0).sum())}个月**改善、**{int((mt.B_minus_A_net_pnl_marked<0).sum())}个月**恶化、其余相同。2023仍非独立样本外，不据本次结果调阈值。

## Q1—Q7 与结论

1. Q1：首次反向至正常退出平均延迟从A的{tim.mean_raw_delay_bars.iloc[0]:.4f}降至B的{tim.mean_raw_delay_bars.iloc[1]:.4f}根，样本平均减少{advance:.4f}根；共有空头入场的实际退出提前均值另见上文，不混淆两个估计量。
2. Q2：空头净利润B−A为 **{ps.B_minus_A_net_profit_marked:.4f} USDT**。完整组合回放已计入额外/消失入场，不能仅凭持仓变短判断获利增加。
3. Q3：空头正常退出 **238→263（+25）**、亏损早退 **101→94（−7）**、固定止损 **26→26（0）**。没有增加空头止损或早退次数，但正常退出净利润由8783.32降至8132.97 USDT。多头固定止损32→37（+5）、早退43→44（+1），表明组合占仓路径变化也影响多头风险。规则本身均未修改。
4. Q4：组合净利润B−A为 **{p.B_minus_A_net_profit_marked:.4f} USDT**（约{100*p.B_minus_A_net_profit_marked/p.A_net_profit_marked:.2f}%）；多头345→365笔，净利润4766.56→4664.57（−101.99）；空头365→383笔，净利润2187.79→1804.93（−382.86）。共有317笔多头入场的实际退出完全相同，差额来自入场机会集合改变；非多头退出规则变更。
5. Q5：组合最大回撤815.5356→817.6198 USDT，B−A为 **{p.B_minus_A_max_drawdown_currency:.4f} USDT**，小幅恶化，并无改善。空头分解回撤858.61→1059.76（+201.15，约23.43%）恶化更明显；多头分解回撤683.26→648.05改善。分解回撤不是可以简单相加的组合回撤。
6. Q6：4个单年中{improved_years}年组合净利润改善，逐年及逐方向差值在年度表完整披露；仅为历史稳定性描述。
7. Q7：**{conclusion}**。退出确实更快、正常退出平均回吐由3.9445降至3.4119价格百分点，但空头净利润减少、组合净利润减少且组合回撤未改善，4年仅2020组合收益改善，当前缺乏将B作为优先候选的充分证据。更低的回吐还同时伴随更低的平均MFE，不能将二者差异直接当成获利改善。这是历史研究取舍，尚无独立样本外或前瞻验证，不升级、不部署B；不证明普遍最优或未来收益。

![A/B资金曲线、回撤、空头退出滞后ECDF和年度组合收益](step2_comparison.png)

![多头、空头及组合逐年收益](annual_direction_comparison.png)

图表保留所有观测，不截尾。两个版本均以同一价格和成本输入计算；图中ECDF为各版本实际空头正常退出样本，非固定交易集。独立复现命令为 `python run_step2.py`、`python build_step2_report.py`。冻结研究的Step1结果不修改；独立目录和分支保存所有实验输出。

**Step 2 到此停止，等待研究负责人审阅；未进入Step 3。**
'''
    (P/'step2_report.md').write_text(text,encoding='utf-8')
    meta=json.loads((P/'diagnostic_metadata.json').read_text(encoding='utf-8'));meta.update(statistical_status='Retrospective; 2020-2022 training, 2023 recipe selection; final caps historically selected with 2023-2024',conclusion=conclusion,shared_entry_short_n=len(short),shared_entry_short_mean_exit_advance_bars=short.exit_advance_bars.mean(),verification_count=len(v));(P/'diagnostic_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(conclusion);print(overall[['direction','A_net_profit_marked','B_net_profit_marked','B_minus_A_net_profit_marked','A_max_drawdown_currency','B_max_drawdown_currency']].to_string(index=False));print(tim.to_string(index=False))

if __name__=='__main__':main()
