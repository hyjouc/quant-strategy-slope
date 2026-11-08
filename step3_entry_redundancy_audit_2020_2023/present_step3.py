"""Descriptive tables and plots; never changes signals or trading parameters."""
from pathlib import Path
import sys,json
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parent.parent.parent/'.research_deps'))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
C=['#0072B2','#A64B00'];SIDES=['long','short'];N=['M','S','A','T']

def read(n):return pd.read_csv(P/n)
def table(d):
    def val(x):return 'NA' if pd.isna(x) else f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x)
    return '\n'.join(['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']+['| '+' | '.join(val(x) for x in r)+' |' for r in d.itertuples(index=False,name=None)])
def export(fig,name):fig.savefig(P/'figures'/name,dpi=160);plt.close(fig)

def main():
    cr=read('indicator_correlation.csv');pc=read('indicator_conditional_pass.csv');pt=read('indicator_pass_patterns.csv');mc=read('marginal_signal_counts.csv');o=read('marginal_signal_outcomes.csv');bs=read('bootstrap_stability.csv');rc=read('ridge_contribution.csv');tc=read('two_confirmation_analysis.csv');ri=read('ridge_independent_information.csv')
    totals=pt.groupby(['direction','period']).n.sum()
    pc['joint_rate_all']=[r.joint_n/totals.loc[(r.direction,r.period)] for r in pc.itertuples()]
    pc.to_csv(P/'indicator_conditional_pass.csv',index=False,encoding='utf-8-sig')
    for source,names,dest in [(cr,N,'indicator_correlation_matrices.csv'),(read('ridge_feature_correlation.csv'),['M','S','A','EMA4h','vol96','deltaM'],'ridge_correlation_matrices.csv')]:
        rows=[]
        for key,g in source.groupby(['direction','period','scope','method']):
            for metric in ['correlation','partial_correlation']:
                mat=pd.DataFrame(np.eye(len(names)),index=names,columns=names)
                for r in g.itertuples():mat.loc[r.feature_1,r.feature_2]=mat.loc[r.feature_2,r.feature_1]=getattr(r,metric)
                for n in names:rows.append(dict(zip(['direction','period','scope','method'],key))|{'metric':metric,'row_feature':n}|mat.loc[n].to_dict())
        pd.DataFrame(rows).to_csv(P/dest,index=False,encoding='utf-8-sig')
    prov=read('parameter_provenance.csv');shared=[dict(component='common frozen mechanism',parameter=n,value=v,source='inputs/ETH_KalmanMomentum_V18C3_TwoConfirm.py and original_registration.json',selection_period='historically selected; original search not rerun',independent_oos=False) for n,v in [('q',1e-7),('R',.01),('momentum_window',4),('normalization_window',96),('trend_window',8),('entry_confirmation',2),('exit_confirmation',2),('early_confirmation',2)]]
    pd.concat([prov,pd.DataFrame(shared)],ignore_index=True).drop_duplicates(['component','parameter'],keep='last').to_csv(P/'parameter_provenance.csv',index=False,encoding='utf-8-sig')
    cg=cr[(cr.period=='2020-2023')&(cr.scope=='all')&(cr.method=='pearson')]
    fig,axs=plt.subplots(1,2,figsize=(10,4.6),layout='constrained')
    for ax,side in zip(axs,SIDES):
        m=np.eye(4)
        for r in cg[cg.direction==side].itertuples():i=N.index(r.feature_1);j=N.index(r.feature_2);m[i,j]=m[j,i]=r.correlation
        im=ax.imshow(m,cmap='RdBu_r',vmin=-1,vmax=1)
        for i in range(4):
            for j in range(4):ax.text(j,i,f'{m[i,j]:.2f}',ha='center',va='center',color='white' if abs(m[i,j])>.6 else 'black')
        ax.set(xticks=range(4),yticks=range(4),xticklabels=N,yticklabels=N,title=side)
    fig.colorbar(im,ax=list(axs),label='Pearson r');fig.suptitle('Direction-adjusted indicators; all eligible closed bars');export(fig,'indicator_correlation.png')
    fig,axs=plt.subplots(1,2,figsize=(14,4.5),layout='constrained',sharey=True)
    for ax,side,c in zip(axs,SIDES,C):
        z=pt[(pt.direction==side)&(pt.period=='2020-2023')];ax.bar(range(16),z.rate*100,color=c);ax.set(xticks=range(16),xticklabels=z.pattern.astype(str).str.zfill(4),title=side,ylabel='Percent of eligible bars',xlabel='M S A T pass pattern');ax.tick_params(axis='x',rotation=70)
    export(fig,'pass_patterns.png')
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained',sharey=True)
    for ax,side in zip(axs,SIDES):
        z=mc[(mc.direction==side)&(mc.period=='2020-2023')];x=np.arange(4)
        for k,layer in enumerate(['technical','pipeline']):
            g=z[z.layer==layer].set_index('removed_condition').reindex(N);ax.bar(x+(k-.5)*.36,g.added_signal_bars,.36,label=layer,color=C[k],hatch='//' if k else None)
        ax.set(xticks=x,xticklabels=N,title=side,xlabel='Condition removed in candidate mask only',ylabel='Added single-bar signals');ax.legend()
    export(fig,'marginal_counts.png')
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained',sharey=True,sharex=True)
    for ax,side in zip(axs,SIDES):
        z=bs[(bs.direction==side)&(bs.block_weeks==8)&(bs.horizon_hours==4)&(bs.metric=='return')&bs.comparison.str.startswith('pipeline_remove')].copy()
        z['condition']=z.comparison.str.split('_').str[2];z=z.set_index('condition').reindex(N)
        for i,r in enumerate(z.itertuples()):
            ax.plot([r.ci95_low,r.ci95_high],[i,i],color=C[0],lw=2);ax.plot(r.mean_difference,i,'o',color=C[0])
        ax.axvline(0,color='black',ls='--');ax.set(yticks=range(4),yticklabels=N,title=side,xlabel='Added minus baseline: 4h return (pp); 8-week CI')
    export(fig,'marginal_outcomes.png')
    fig,axs=plt.subplots(1,3,figsize=(15,4),layout='constrained',sharey=True)
    for ax,model in zip(axs,['unfiltered_long','unfiltered_short','filtered_long']):
        z=rc[(rc.model==model)&(rc.period=='2020-2023')&(rc.scope=='all_technical_pass')];ax.bar(z.feature,z.mean_abs,color=C[0] if model.endswith('long') else C[1]);ax.set(title=model,ylabel='Mean absolute beta * clipped z');ax.tick_params(axis='x',rotation=45)
    export(fig,'ridge_contributions.png')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained',sharey=True)
    for ax,side in zip(axs,SIDES):
        z=o[(o.direction==side)&(o.layer=='confirmation')&(o.period=='2020-2023')]
        for group,ls,c in [('first_success','-',C[0]),('first_failed','--',C[1]),('second_after_success',':','#333333')]:
            g=z[z.group==group].sort_values('horizon_hours');ax.plot(g.horizon_hours,g['return_mean'],label=group,color=c,ls=ls,marker='o')
        cost=tc[(tc.direction==side)&(tc.period=='2020-2023')].mean_cost_pct.iloc[0]
        ax.set(title=f'{side}; confirmation mean price cost {cost:.3f}%',xlabel='Hours from each hypothetical next open',ylabel='Mean directional return (%)');ax.axhline(0,color='black',lw=.7);ax.legend(fontsize=8)
    fig.suptitle('Retrospective grouping: first-window includes the confirmation bar')
    export(fig,'two_confirmation.png')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained',sharey=True)
    for ax,side in zip(axs,SIDES):
        z=bs[(bs.direction==side)&bs.comparison.eq('confirmation_common_clock_success_minus_failed')&bs.metric.eq('return')&bs.block_weeks.eq(8)].sort_values('horizon_hours')
        ax.errorbar(z.horizon_hours,z.mean_difference,yerr=np.vstack([z.mean_difference-z.ci95_low,z.ci95_high-z.mean_difference]),fmt='o-',capsize=4,color=C[0] if side=='long' else C[1]);ax.axhline(0,color='black',ls='--');ax.set(title=side,xlabel='Hours after common executable open',ylabel='Success minus failed return (pp); 8-week 95% CI')
    fig.suptitle('Confirmation outcome observed before both groups enter at i+2 open')
    export(fig,'confirmation_common_clock.png')
    # Compact question-oriented evidence tables; all detailed periods remain in CSV.
    passes=pc[(pc.period=='2020-2023')&(((pc.event=='A')&(pc.condition=='T'))|((pc.event=='T')&(pc.condition=='A'))|((pc.event=='S')&(pc.condition=='M'))|((pc.event=='M')&(pc.condition=='S')))]
    counts=mc[(mc.period=='2020-2023')][['direction','removed_condition','layer','original_signal_bars','added_signal_bars','ridge_rejected_added','new_confirmed_candidate_bars']]
    primary=bs[(bs.metric=='return')&(bs.horizon_hours==4)&bs.comparison.str.contains('remove')][['direction','comparison','block_weeks','n_group1','n_group0','mean_group1','mean_group0','mean_difference','ci95_low','ci95_high','yearly_same_sign_as_overall']]
    ridgebs=bs[(bs.metric=='return')&bs.comparison.eq('ridge_pass_minus_reject')][['direction','horizon_hours','block_weeks','mean_difference','ci95_low','ci95_high','yearly_same_sign_as_overall']]
    confbs=bs[(bs.metric=='return')&(bs.horizon_hours.isin([1,4,8]))&bs.comparison.str.startswith('confirmation')][['direction','comparison','horizon_hours','block_weeks','n_group1','mean_difference','ci95_low','ci95_high','yearly_same_sign_as_overall']]
    # Results and recommendations are authored after the compute stage, never fed back to it.
    txt=f'''# V19 Step 3：入场条件冗余与独立信息贡献审计

任务编号：`V19_STEP3_ENTRY_REDUNDANCY_AUDIT_2020_2023`。

**本次全部结论属于探索性的历史样本内机制诊断。2020—2022 已用于参数优化和模型训练，2023参与配方选择；最终cap配方历史上使用过2023—2024选择。2023不是独立样本外。本次没有读取2024年及以后任何行情、指标、资金曲线或交易结果。Bootstrap不能消除参数搜索偏差或多重比较选择偏差。**

## 范围与冻结

从 Step2 `c2cedb8524ebc7c6574c03d41cb0decdbaf086a5` 建立独立分支；只新增本目录，Step1和Step2已提交文件保持不变。此前未提交的退出稳健性目录保留在工作区且不纳入本研究或提交。未修改V18、任何阈值、模型、确认、退出、早退或止损；没有训练、优化或完整策略消融回测。

复用Step1 `run_audit.original_features` 与 `get_signals`，直接从冻结源码提取原指标函数。先按日期列核对数据边界，再加载已验证 `inputs/ohlcv_closed_pre2024.feather` 并立即截断，之后计算指标。原始140255根15分钟完整K线，开盘范围2020-01-01 00:00至2023-12-31 23:30 UTC，最后收盘23:45 UTC；正式信号统计起点2020-01-10 23:45 UTC，139297根有效M/S/A/T记录。4小时EMA按16根完整15分钟K线聚合，以已完成4小时收盘时间向后匹配；指标和评分通过历史前缀重算核验。

正式起点首根的4小时EMA尚未达到原始60根最小历史，评分按冻结 `nan_to_num` 将缺失标准化特征置0，未补未来信息。相关性、偏相关性按各分析所需特征完整样本计算，有效n单列。M=4根动量，S=原始三状态斜率/此前96根标准差，A=同法标准化加速度，T=最近8根原始斜率回归趋势；均乘方向s。阈值多头0.69/0.21/0.340/0，空头0.76/0.49/0.505/0，全部严格大于。Ridge维持原门槛、均值、尺度和±5截断。

信号级诊断不受实际仓位占用限制，不加杠杆，不扣手续费/滑点；不是1000USDT账户收益，也不累加重叠信号。**信号i收盘后，以第i+1根开盘作为假设入场；h小时终点为i+4h根收盘，MFE/MAE使用i+1至i+4h完整K线高低价。** 截止不足的标签为NA，各表披露有效n和排除数；绝不取信号当前收盘冒充成交。

## M/S/A/T相关性与通过集合

{table(cg[['direction','feature_1','feature_2','n','correlation','partial_correlation']])}

上表偏相关控制其余两项指标。CSV还提供Spearman及秩偏相关、逐年逐季度，以及M/S已通过、四项全通过、对每项控制其余三项已通过的条件样本相关。季度只是日历分组，不是重新选择市场阈值；另列冻结4小时EMA上/下环境。条件筛选本身会造成范围限制与碰撞偏差，不把条件相关当因果独立性。

{table(passes[['direction','event','condition','condition_n','conditional_probability','unconditional_pass_rate']])}

`indicator_pass_patterns.csv` 列出所有16组合（固定M,S,A,T位序，含数量和比例），直接涵盖两两共通过、三项通过另一项拒绝及四项全通过；`indicator_conditional_pass.csv` 列出全部有序条件概率及单项通过/拒绝比例。对角event=condition的概率1只用于核对，不作信息价值证据。

![M/S/A/T相关性](figures/indicator_correlation.png)
![全部16种通过组合](figures/pass_patterns.png)

## 单项条件边际贡献

technical层仅删除技术掩码的一项：其余三项通过且该项失败；pipeline层将同一批新增单根候选再经过**原封不动的Ridge**。M/S/A依然留在质量模型特征中，所以pipeline测量显式门槛在当前流水线内的边际作用，不代表移除整个信息源。这里只构建诊断掩码，不删除真实条件、不运行策略。

{table(counts)}

`new_confirmed_candidate_bars` 额外提供保留月初重置和两根确认后相对于原流水线新增的信号K线数量（不是交易笔数，可为现根本已全通过而前根新增引发）。该字段不使用实际单持仓约束，不能称为最终新增成交。单根新增的年度、季度分布及指标均值/中位数/P10/P90见 `marginal_signal_counts.csv` 和 `marginal_feature_distributions.csv`。

以下主表为4小时价格收益的“新增−基准”差异；完整1/2/4/8小时收益、MFE、MAE及正方向比例见 `marginal_signal_outcomes.csv`，Bootstrap文件覆盖全部四期限三项诊断。负差异表示新纳入信号在这个历史均值上较差。

{table(primary)}

![单项放松后新增信号数](figures/marginal_counts.png)
![保留Ridge后的新增信号4小时表现差异与8周区块95%区间](figures/marginal_outcomes.png)

## Ridge重复过滤与信息来源

三个登记模型全部按原系数分解：unfiltered_long、unfiltered_short为当前基准两方向模型；filtered_long仅作登记模型的背景贡献审计，不替换基准模型、不将其当第三个实际交易方向。`ridge_coefficients.csv` 列标准化系数、原均值/尺度、截距和门槛；每个模型逐根的 `contributions_*.csv.gz` 记录 beta_j×clip(z_j,±5)，严格核对六项之和+截距=预测分数。`ridge_contribution.csv` 列全部有效K线和四项技术已通过后的逐年贡献均值、中位数、绝对均值、绝对中位数/P90/P95。

![三个模型六项特征的实际绝对贡献](figures/ridge_contributions.png)

六特征原始值的Pearson/Spearman、逐年变化和全技术条件通过后相关详见 `ridge_feature_correlation.csv`。同时将原评分拆成M/S/A、EMA、vol、deltaM贡献。按**贡献是否非负**固定分组，没有优化阈值；`ridge_source_outcomes.csv` 比较单项贡献符号、EMA与vol同时非负，以及相反集合的固定未来表现。这些分组是描述性归因，不能因相关特征的单个系数或贡献大就推断因果作用。

四项技术全过后，Ridge通过−拒绝的前瞻价格差异：

{table(ridgebs)}

`ridge_filter_effect.csv` 提供两组独立n、均值、中位数、MFE/MAE、正方向比例和逐年结果。`ridge_independent_information.csv` 另报告评分与未来收益控制EMA/vol后的线性偏相关、M/S/A贡献控制EMA/vol/deltaM后的偏相关，以及EMA、vol、deltaM分别控制其他五特征后的偏相关。只做解释性线性投影，不重新拟合Ridge，不训练或调节任何预测门槛；控制相关特征后的微小残差可能不稳定，不能据此宣称“真正独立”预测能力。

## 两根确认：等待、被排除事件与后续质量

一个事件定义为同一UTC月内连续完整信号段的第一根；不逐根重复计入同一连续段，不考虑真实仓位占用或卖出后的确认重新开始。第二根完整信号和月度一致性均满足，且其后开盘在允许数据内才记为确认成功；否则分为条件失败、月度重置、边界截断。条件失败不自动等于低质量，必须看未来表现。

{table(tc[tc.period=='2020-2023'])}

第一根假设入场为i+1开盘，第二根确认后的假设入场为i+2开盘；成功事件第一根收盘至实际假设入场等待15分钟。**第二根收盘与下一根开盘共享同一时刻，执行价格仍取下一开盘，而非前根收盘。** 等待价格成本定义100×s×[open(i+2)/close(i)−1]。`confirmation_events.csv.gz` 保留全部时刻、原因和成本。

第一根成功与第一根失败是不同事件集；“第二根−第一根”的比较使用**同一成功事件配对**，两个前瞻窗口有15分钟错位且高度重叠，不能当独立样本。配对Bootstrap先在每个事件上算差，再整体抽取该事件时间区块。其group1均值/中位数为事件配对差值、group0为同批事件的零基准，并非两组原始收益；原始收益见 `confirmation_outcomes.csv`。

**首次信号后的成功/失败比较存在机械前视分组：收益窗口包含了用于决定是否确认的那根K线，不能将其巨大差异解释为可交易的独立预测增益。** 因此另补充 `confirmation_common_clock_success_minus_failed`：成功与失败事件均从i+2开盘（确认结果已经可观察的同一时刻）开始比较未来1/2/4/8小时，去掉确认K线本身。`common_clock_confirmation.py` 用已保存的同一组4周/8周抽样追加这一核验，最终主脚本也包含同样定义，未增加策略版本或调参。

{table(confbs)}

![首次完整信号成功/失败与第二次确认的未来表现](figures/two_confirmation.png)

上图首次窗口按事后确认结果分组，包含确认K线，不能作独立预测证据。去掉确认K线、同一执行时刻比较如下：

![同一执行时刻的确认成功与失败组未来收益差](figures/confirmation_common_clock.png)

## 时间区块Bootstrap与解释限制

4周和8周分别进行10000次连续时间区块抽样，固定种子20261108+周数；以2020-01-01起整段日历为轴，从全部合法起点抽取28/56日连续区块，最后一段截短至原日历长度，同日全部信号一起抽取，不做逐根IID抽样。所有比较复用相同区块抽样，以保留跨组共同市场冲击。`bootstrap_4w_draws.csv.gz`/`bootstrap_8w_draws.csv.gz` 保存全部区块起点，可复现每次抽样。

95%区间是**均值差**Bootstrap分布的2.5%/97.5%分位，不是单个信号的预测区间，也不是中位数差的置信区间；两组样本n、均值、中位数以及各年度差异方向一致性都列于 `bootstrap_stability.csv`。稀疏组在某次抽样中缺样的重复会被排除，有效重复数单列。这里没有调整成百上千比较的家族错误率，不能从最窄/最有利区间事后宣布条件无用；不把重叠信号价格收益累加为策略净利润。不同区块长度之间的敏感性同样是研究结果。

## Q1—Q8及下一轮研究取舍

{(P/'conclusions.md').read_text(encoding='utf-8') if (P/'conclusions.md').exists() else '结论整理中；交付前补齐。'}

## 复现、边界和交付

先运行 `python run_step3.py`，再运行 `python present_step3.py` 和 `python validate_step3.py`。依赖numpy/pandas/pyarrow/scipy/matplotlib/Pillow，原工作区 `.research_deps` 可复用；代码不调用交易重放入口。逐年/季度相关矩阵以长表保存，避免对角和对称重复造成庞大矩阵文件。输入路径与哈希、时限、随机种子及验证结果见 `diagnostic_metadata.json` 和 `verification.csv`。首次执行因误将EMA缺失行剔除导致预热样本计数断言失败，未产出分析结果；已按原冻结缺失处理修正，失败日志保留 `run_attempt1.log`，成功运行以 `run.log` 为准。

本轮到此停止，等待研究负责人审阅，不开展完整策略消融或下一阶段。
'''
    (P/'step3_report.md').write_text(txt,encoding='utf-8')
    pics={f.name:{'size_pixels':Image.open(f).size,'dpi':Image.open(f).info.get('dpi'),'source':'Corresponding CSV tables; no trimming or smoothing','destination':'Internal exploratory research report, not journal-specific'} for f in (P/'figures').glob('*.png')};(P/'figure_metadata.json').write_text(json.dumps(pics,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Report and seven figures generated',flush=True)

if __name__=='__main__':main()
