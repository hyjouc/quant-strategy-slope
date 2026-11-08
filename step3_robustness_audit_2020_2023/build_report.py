"""Build a transparent retrospective robustness report and diagnostic figures."""
from pathlib import Path
import sys,json,hashlib
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent.parent.parent/'.research_deps'))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def table(d):
    def f(v):
        if pd.isna(v):return 'NA'
        return f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']+['| '+' | '.join(f(v) for v in r)+' |' for r in d.itertuples(index=False,name=None)])

def main():
    read=lambda n:pd.read_csv(P/n)
    concentration=read('concentration_deletions.csv');cs=read('concentration_summary.csv');rank=read('common_normal_short_ranked.csv')
    windows=read('time_stability_all_windows.csv');st=read('time_stability_summary.csv');boot=read('bootstrap_summary.csv');cost=read('cost_stress_comparison.csv');prov=read('parameter_provenance.csv')
    fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    axs[0,0].plot(rank['rank'],rank.cum_A_minus_B,color='#0072B2');axs[0,0].axhline(0,color='black',lw=.8);axs[0,0].axhline(rank.A_minus_B.sum(),color='#A64B00',linestyle='--',label='Full238 net advantage');axs[0,0].set(xlabel='Trades ranked by A minus B (descending)',ylabel='Cumulative A minus B (USDT)',title='Matched normal short advantage concentration');axs[0,0].legend()
    for kind,c,style in [('rolling6m','#0072B2','-'),('rolling12m','#A64B00','--')]:
        z=windows[(windows.window_type==kind)&(windows.direction=='combined')];axs[0,1].plot(pd.to_datetime(z.end_exclusive_utc,utc=True)-pd.Timedelta(seconds=1),z.B_minus_A_net_profit,label=kind,color=c,linestyle=style)
    axs[0,1].axhline(0,color='black',lw=.8);axs[0,1].set(title='All rolling portfolio windows',ylabel='B minus A marked PNL (USDT)',xlabel='Window end (UTC)');axs[0,1].legend()
    z=boot[boot.direction=='combined'];ix=np.arange(len(z));axs[1,0].errorbar(ix,z.bootstrap_median,yerr=[z.bootstrap_median-z.percentile95_low,z.percentile95_high-z.bootstrap_median],fmt='o',capsize=6,color='#0072B2');axs[1,0].axhline(0,color='black',lw=.8);axs[1,0].set(xticks=ix,xticklabels=['4-week blocks','8-week blocks'],ylabel='B minus A total PNL (USDT)',title='Paired MBB median and95% percentile interval')
    for side,c,marker in [('long','#0072B2','o'),('short','#A64B00','s'),('combined','#333333','^')]:
        z=cost[cost.direction==side];axs[1,1].plot(z.cost_multiplier,z.B_minus_A_net_profit,label=side,color=c,marker=marker)
    axs[1,1].axhline(0,color='black',lw=.8);axs[1,1].set(xlabel='Total fee and slippage multiplier',ylabel='B minus A PNL (USDT)',title='Predeclared cost scenarios',xticks=[1,1.5,2]);axs[1,1].legend()
    fig.savefig(P/'robustness_overview.png',dpi=160);plt.close(fig)
    # Full monthly differences: no selective omission of windows.
    z=windows[(windows.window_type=='monthly')&(windows.direction=='combined')]
    fig,ax=plt.subplots(figsize=(14,4),layout='constrained');ix=np.arange(len(z));ax.bar(ix,z.B_minus_A_net_profit,color=np.where(z.B_minus_A_net_profit>=0,'#0072B2','#A64B00'));ax.axhline(0,color='black',lw=.8);ax.set(xticks=ix[::3],xticklabels=z.window.iloc[::3],xlabel='All UTC months',ylabel='B minus A PNL (USDT)',title='Every monthly portfolio comparison; positive favors B');ax.tick_params(axis='x',rotation=45);fig.savefig(P/'all_monthly_differences.png',dpi=160);plt.close(fig)
    allcost=read('cost_stress_all.csv')
    # Frozen actual ledgers repriced as a second, accounting-only diagnostic.
    fixed=[]
    for v,file in [('A','trades_A_baseline.csv'),('B','trades_B_sign_exit.csv')]:
        t=pd.read_csv(P.parent/'step2_short_exit_ablation_2020_2023'/file)
        signs=np.where(t.direction=='long',1,-1)
        for m in [1,1.5,2]:
            ep=t.entry_price/(1+signs*.0002)*(1+signs*.0002*m);xp=t.actual_exit_price/(1-signs*.0002)*(1-signs*.0002*m);fee=.0005*m;units=2000/(ep*(1+fee));pnl=signs*units*(xp-ep)-units*(ep+xp)*fee
            if m==1:assert np.allclose(pnl,t.net_realized_pnl,atol=1e-8)
            fixed.append(dict(variant=v,cost_multiplier=m,trades=len(t),net_pnl=pnl.sum(),definition='fixed actual base ledgers repriced; raw exit unchanged even for stops; accounting diagnostic only'))
    fd=pd.DataFrame(fixed);fd.to_csv(P/'fixed_ledger_cost_sensitivity.csv',index=False,encoding='utf-8-sig')
    yearly=windows[(windows.window_type=='annual')&(windows.direction=='combined')]
    status='有一定证据但存在明显局限'
    text=f'''# Step 3：样本内优化偏差与策略稳健性审计

**结论等级：{status}。** A 的总体历史净收益较高，年度及重叠长窗口中有一定时间一致性，且成本压力下相对优势仍在；但优势高度集中，月/季并不普遍占优，配对时间区块Bootstrap的95%收益差区间跨零，完整历史搜索记录缺失，不能据此证明0.38%退出阈值具有独立、可泛化的效果。不把“继续保留A”包装为已证实该条件有效；本轮不选择新参数或升级策略。

**2020—2022用于核心参数优化和质量模型拟合，2023参与配方选择，最终多头cap0.02、空头cap0.08配方选择明确使用2023—2024。2020—2023是已参与历史选择的回顾性样本，不是独立样本外验证。Bootstrap不能修复历史参数搜索的选择偏差。**

## 研究边界与数据

本轮只做归因、时间稳定性和成本敏感性；未拟合/训练模型、未搜索阈值、未改变原始V18、未部署或进入下一步。市场数据仅为Step1 `inputs/ohlcv_closed_pre2024.feather`：140255根已完整收盘的15分钟K线，开盘2020-01-01 00:00至2023-12-31 23:30 UTC，最后完整收盘23:45 UTC，正式起点2020-01-10 23:45 UTC。不读取2024及之后任何行情、指标或交易结果，不使用资金费率，组合持续单持仓、不年度重启。

Step2 A/B完整交易和逐根盯市曲线保留不动。A平仓710笔，组合净利润6954.3495 USDT；B平仓748笔，6469.4992 USDT。原始A−B=484.8502 USDT。以下收益差除集中度/剔除表明确标为 **A−B** 外，均为 **B−A**，正值有利B。

## 1. 参数来源与可追溯性

已检查可用Git历史、冻结登记、模型元数据、最终搜索与质量选择源码、锁定选择元数据及来源审计摘要。Git仅保留此次迁移后的Step1/Step2提交，未保留优化发生时的完整提交、原始搜索日志和所有候选记录；不能用近期文件时间戳冒充历史预注册。`evidence/available_git_history.txt`是实际可用日志。

逐项审计表为 `parameter_provenance.csv`（{len(prov)}行），每个参数含值、历史选择区间、选择依据、实际尝试次数、记录完整性及证据。无法确认项统一标 **UNKNOWN**。摘要如下：

| 参数组 | 冻结值 | 可确认的最终选择/拟合区间 | 依据与记录局限 |
| --- | --- | --- | --- |
| 空头动量退出 | 0.38% | 最终核心搜索2020—2022 | 联合六维方向坐标搜索；0.38单项实际尝试次数UNKNOWN，不是单独的因果验证 |
| 入场M/S/A | 多0.69/0.21/0.340；空0.76/0.49/0.505 | 最终核心搜索2020—2022 | 逐年盈利、最低30交易、DD软惩罚、近优平台/邻域规则；完整候选轨迹缺失 |
| T条件、窗口 | 方向调整T>0，8根 | 初始设计选择期UNKNOWN；最终搜索固定 | 不能推断这些固定值未曾被更早研究选择 |
| 固定止损/亏损早退 | 多3.87%/1.49%；空4.07%/1.43% | 最终核心搜索2020—2022 | 与入场/退出参数联合搜索，实际试验总数UNKNOWN |
| Kalman | q=1e-7、R=.01、dt=.25h；原始斜率反向阈值0 | 初始设计选择期UNKNOWN；最终搜索固定 | 固定不等于从未优化，缺原始设计试验记录 |
| 两根确认及armed | 入场/正常退出/早退均2根；原始armed | 初始选定依据/期UNKNOWN；最终固定 | 源码证明执行值，不证明从未试过一根/其他机制 |
| Ridge权重、尺度、截距、预测门槛 | 逐项精确值见CSV/模型快照；alpha100 | 拟合2020—2022 | cap内标签裁剪，训练预测中位数门槛；所用最终模型与历史frozen_models逐项完全一致，未重新拟合 |
| 最终cap配方 | 多0.02、空0.08 | **2023—2024** | 锁定元数据selection_end=2024-12-31 23:45UTC；源码验证区间到2025前；4候选none/.02/.04/.08按相对收益、回撤和相邻cap最差评分选择 |

最终搜索程序声明3起点、每起点最多20轮、6个活动参数轴、2个最低交易数档（30/75）×4方向案例，C3细网格0.001，其余连续轴0.01；它们是**规定的搜索范围/上限，不是实际尝试次数**。来源摘要记载8个anchor及最终参数，原始 `anchors/30/unfiltered_short.json`、long对应JSON和`.steps.csv`不存在，故每个参数实际评估次数、全生命周期重复次数均UNKNOWN。未重新运行历史优化来填补这些缺口。

质量脚本每方向/档指定3个cap拟合并与none组成4配方候选，可确认的是设计候选数，不是历史总尝试数。原始训练样本台账和完整环境不足，无法重审所有标签生成细节；本轮只核实冻结模型元数据与历史模型逐字段一致、末次训练标签退出时间在2022年底之前。原始完整验证候选评分未保留，故配方选择不能称完整可复核搜索日志。

特别确认：**2024用于选择最终模型/cap配方，未证明2024用于重拟合模型权重。** 源码及标签元数据支持权重拟合在2020—2022；最终“用哪组权重”通过2023—2024选择。因此既不能说2024从未参与，也不能笼统声称权重在2024重新训练。源脚本仅作为文本读取，不导入、不执行；`evidence/`只保存源码、配置、模型元数据和方法片段，不读取历史2024—2026行情或绩效文件。更早设计选择和累计试验次数UNKNOWN，无法量化或校正完整选择偏差。

## 2. 共同正常退出空头与收益集中度

按相同入场时间+方向精确匹配，且A/B都属于正常退出，确为238笔；其中A占优58笔、B占优108笔、近似相同72笔（盈亏差绝对值≤1e-8 USDT）。这不是全部真实组合，也不是所有空头交易。A/B实际组合分别还有其他交易机会及多头贡献。

{table(cs)}

238笔中58笔A占优总差2173.4486 USDT，被108笔B占优的−1537.6517抵消，剩635.7969。共同正常退出A净利润8783.3193，与所有A正常空头相同；共同B正常空头8147.5223，**不等于B全部263笔正常空头的8132.9666**。差异来自额外交易集合，不能用共同样本直接替代全组合。

以下按 `A净盈亏−B净盈亏` 从大到小排序，top百分比使用ceil取整；对238总样本和58笔正贡献样本都列明分母。排名目的是审计“优势贡献”，不是单独按A绝对盈利排序。全部逐笔及累计值见 `common_normal_short_ranked.csv`，不截尾、不winsorize。

{table(concentration[['deletion','removed_n','removed_A_minus_B','share_of_common_net_advantage','share_of_58_positive_advantage','remaining_common_A_minus_B','diagnostic_ledger_A_minus_B']])}

前5/10/20笔A−B优势贡献分别885.4598/1425.5028/1820.2057 USDT，占58笔正贡献约40.74%/65.59%/83.75%，占共同样本净优势139.27%/224.21%/286.29%。百分比超过100%是因为其他交易抵消，不能截成100%。前5%（12笔）、10%（24笔）、20%（48笔）交易贡献也完整披露。**剔除前5笔后共同样本A−B转为−249.6628，完整已发生交易台账同时扣掉这5笔配对交易后的A−B为−400.6095。** 原始正式结果仍为A−B=484.8502，绝不以删后结果取代。

剔除只在双方已发生交易台账中移除同一对入场，不重新释放仓位、不重建资金曲线、不模拟后续替代交易，故 `diagnostic_ledger_*` 只是会计敏感性，**不是可实现的新策略收益**。不把删后PF、回撤或Sharpe伪装为重新回放结果，也不根据删后结果修改策略。少数大收益可来自趋势策略正常厚尾结构；集中本身不证明无效，但削弱“普遍持续优于B”的推断。

## 3. 全时间窗口稳定性

使用同一已发生真实组合曲线，不在窗口起点重开/平仓。每窗口净利润=结束盯市−开始前最后盯市，DD含该窗口初始基准；PF和交易笔数按窗口内退出归属，跨窗口持仓会导致平仓盈亏和盯市利润不同。Sharpe用每日盯市盈亏/1000，零无风险利率、样本标准差、√365年化。无亏损窗口PF为NA，不编造无穷或0。

覆盖全部4年度、16季度、48个月、43个滚动6月、37个滚动12月窗口×3方向，共444行，均在 `time_stability_all_windows.csv`；没有只取有利区间。滚动窗每月移动，完全重叠并非独立重复试验。2020首次窗口始于正式起点，2020年1月为部分月，含该月的首个6/12月窗口亦应理解为扣除统一预热后的窗口，不补齐虚构的空仓历史。

{table(yearly[['window','A_net_profit','B_net_profit','B_minus_A_net_profit','A_profit_factor','B_profit_factor','A_sharpe','B_sharpe','A_max_drawdown','B_max_drawdown','A_trades_closed','B_trades_closed']])}

{table(st)}

组合A在3/4年、27/43滚动6月、30/37滚动12月收益高于B，但季度8比8、月度23比23且2月相同，不存在每月普遍占优；空头月度A高于B19/48，B高于A27/48。长窗较一致支持**一定时间证据**，但高度重叠、同一批大交易反复出现会放大表面重复性，不能用30/37当30个独立成功实验。

## 4. 配对时间区块Bootstrap

基准为正式起点至2023-12-31最后完整收盘的**1452个UTC日**末次盯市盈亏序列，包含空仓日；第一天为正式起点后的部分日，最后一天截至23:45。数值先用USDT每日增量存储，收益率是其/1000，固定不复利。A/B以及多空分解始终抽取同一日历索引区块。

使用noncircular overlapping moving-block bootstrap，所有合法连续28天/56天起点等概率有放回抽取，分别代表4周/8周连续区块（可从任意UTC日开始，并非只周一）。每次抽ceil(N/L)块，最后一块截到刚好1452日；不跨首尾循环、不中心化。每种长度20000次，种子20261108+L。以抽样总B−A USDT分布的2.5%/97.5%分位数形成百分位95%区间，另输出B−A>0的经验比例；非逐笔IID抽样。原始40000个配对重复合计及方向分解保存在 `bootstrap_paired_replicates.csv.gz`，区块均同索引且经线性一致性核验。

{table(boot[['block_days','replicates','direction','observed_B_minus_A','bootstrap_median','percentile95_low','percentile95_high','fraction_B_minus_A_gt0']])}

组合28天区块95%区间约[−1695.63,471.63] USDT，B−A>0比例15.815%；56天约[−1674.72,399.66]，比例13.22%。**两者均跨零，不能给出A优于B的95%稳健结论。** 经验比例不是p值、不是“B真正更优的后验概率”，也不是未来胜率。原始估计−484.85和抽样均值/中位数不完全一致，是非循环边界权重、有限样本和区块截尾造成的，未为贴近原始估计重新调方法。

局限：区块内保留部分依赖，区块拼接破坏更长期关系和市场状态顺序；原样本高度非平稳，仅4年；不能再现新市场路径中的自适应交易执行，只对已发生的每日曲线抽样。4/8周是预指定两个敏感性尺度，不能看哪种显著就挑哪种。**Bootstrap条件于已经选好的策略/模型，既不重新执行历史参数选择，也不能恢复其缺失试验次数，所以无法修复数据窥探、多次搜索、cap选择或研究者选择偏差。** 本轮不能给出偏差校正后的净优势或deflated Sharpe结论。

## 5. 交易成本压力

将“1.5倍、2倍”明确解释为**总成本分别为基准的1.5×/2×**，手续费和滑点每边同时乘倍数：1×为0.05%/0.02%；1.5×为0.075%/0.03%；2×为0.10%/0.04%。未按“额外加1.5×即2.5×总成本”解释。仅三个预指定成本情景，不搜索参数或择优成本。

每情景对A/B做完整持续单持仓重放，指标、M/S/A/T和Ridge门槛/分数/入场资格、正常/早退/止损逻辑与优先级全部不变，只有成交滑点、手续费及预留手续费数量发生变化。**成本改变成交入场价，因此相对入场价的止损价与早退盈亏条件可能随之改变，后续成交集合可能变化**；这是原逻辑在新执行成本下的内生结果，不是修改信号公式。本轮每个成本情景均由独立标量执行器核对订单、成交价与资金曲线，并保存交易与逐根资金曲线；1×曲线与Step2再次对账。

{table(allcost)}

{table(cost[['cost_multiplier','direction','B_minus_A_net_profit','B_minus_A_max_drawdown','B_minus_A_profit_factor','B_minus_A_sharpe','B_minus_A_trades_closed']])}

组合A−B净收益差在1×/1.5×/2×分别为484.8502/567.2428/665.6870 USDT，利润排序保持。但回撤不一致：1.5×下A组合DD为917.0290、B为863.2480，**B回撤更低53.7810 USDT**；2×下A为1007.0244、B为1081.6928，A回撤较低。不能据成本压力把A描述成在全部指标上占优，且压力下两版本绝对利润都明显下降。

另附 `fixed_ledger_cost_sensitivity.csv`：锁定原始已发生交易时点、原始无滑点价格，再改变双边成本重算，1×精确重现原始净收益。此项只隔离会计成本变化，连原止损的无滑点退出价也固定，**不声称是新成本下完整合法执行路径**；正式压力结果以上述全回放为准。两种口径同时披露，避免将固定交易扣成本与完整重放混淆。均未使用资金费率。

## 6. 最终证据分级

选择：**{status}**。

- 已测得：A在原样本总体、3/4单年及多数滚动长窗收益较高；预指定成本压力的相对关系可直接核查。238共同正常空头的优势高度集中，前5笔已超过净优势；短窗口并不普遍占优。
- 尚不能证明：0.38%阈值具有普遍、独立、可泛化的退出效果；A未来将优于B。两种区块Bootstrap95%区间跨零，且不会纠正选择偏差。
- 证据缺口：完整原始搜索次数/轨迹、初始机制设计历史、所有模型候选验证记录，以及未参与选择的独立证据。未知项保留UNKNOWN，不补造试验数量。

这一分级强于只复述总利润，但**不等于“稳定性证据较强”**，也不将过去样本重新命名为样本外。若研究负责人采用更严格的“必须独立验证”标准，则目前最多只能认定历史样本内优势；本报告的“一定证据”仅指条件于已冻结配置的时间和成本敏感性。无需由本轮决定新参数、继续优化或部署。

![集中度、全滚动窗口、配对区块区间与成本压力](robustness_overview.png)

![全部48个月的组合收益差](all_monthly_differences.png)

图表保留全部窗口与交易，不截去反向贡献；ECDF/区间表所用原始数值可在CSV复查。来源、参数表、所有正式结果与剔除诊断分别保留。初次脚本有日历时区端点错误、第二次独立成本核验截获诊断代码的手续费字符串替换错误；均已修复并保留失败日志，未改任何交易规则、模型或阈值，最终以成功 `run.log` 为准。

复现：`python run_step3.py` → `python build_report.py` → `python validate_step3.py`。分析目录须保留在当前仓库结构，源证据路径见 `evidence_index.csv`。本轮结束，等待研究负责人审阅；不自行进入下一阶段。
'''
    (P/'step3_report.md').write_text(text,encoding='utf-8')
    meta=json.loads((P/'diagnostic_metadata.json').read_text(encoding='utf-8'));meta['evidence_classification']=status;meta['classification_scope']='conditional retrospective temporal/cost evidence, not independent validation';(P/'diagnostic_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(status);print(cost[cost.direction=='combined'][['cost_multiplier','net_profit_A','net_profit_B','B_minus_A_net_profit']].to_string(index=False))

if __name__=='__main__':main()
