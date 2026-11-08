"""Seven-parameter annual replay. Risk parameters lock at entry across month boundaries."""
import numpy as np
import pandas as pd
FEE=.0005
SLIP=.0002 # Original research friction; audit_reference explicitly sets zero.
MODES=['simple_1x','simple_2x','compound_1x','compound_2x']
LEV=np.array([1.,2.,1.,2.])
def stop_fill(sign,entry,threshold,op,hi,lo):
    if threshold is None:return None
    level=entry*(1-sign*threshold)
    gap=(op<=level if sign==1 else op>=level)
    touch=(lo<=level if sign==1 else hi>=level)
    if not touch and not gap:return None
    raw=op if gap else level
    return raw*(1-sign*SLIP),gap,level


def run(d,entries,exits,threshold,early_exit=False,policy=None,allowed=(1,-1),entry_gate=None,fixed_account_ratio=1.,indicator_bank=None,indicator_schedule=None,risk_schedule=None,protection=None,require_slope_trend=True,entry_trend_field=None,bar_hours=.25):
    assert fixed_account_ratio>=1
    cash=np.ones(4);q=[];pos=None;pending=0;sell=False;counts={1:0,-1:0};month=None;trades=[]
    rows=list(d.itertuples(index=False))
    bank=None if indicator_bank is None else {r:{c:f[c].to_numpy() for c in f.columns if c in ['slope','slope_scaled','model_accel_scaled'] or c.startswith('trend')} for r,f in indicator_bank.items()}
    if bank is not None:
        assert indicator_schedule is not None and policy is None
        for f in indicator_bank.values():
            pd.testing.assert_series_equal(f.date.reset_index(drop=True),d.date.reset_index(drop=True))
    def indicator(c,i,r=None):return getattr(rows[i],c) if bank is None else bank[r][c][i]
    blocked={1:False,-1:False};reset_counts={1:0,-1:0};blocked_until={1:-1,-1:-1}
    if protection:
        assert protection.get('be_trigger_pct') is None or protection['be_trigger_pct']>0
        if protection.get('trail_activation_pct') is not None:
            assert protection['trail_activation_pct']>=protection['trail_distance_pct']>0
    def protective_fill(b,open_only=False):
        if not protection:
            return stop_fill(pos['sign'],pos['price'],pos['stop_fraction'],b.open,b.open if open_only else b.high,b.open if open_only else b.low)
        s=pos['sign'];level=pos['active_stop'];gap=b.open<=level if s==1 else b.open>=level
        touch=False if open_only else (b.low<=level if s==1 else b.high>=level)
        if gap or touch:return (b.open if gap else level)*(1-s*SLIP),gap,level
        return None
    def stop_reason(gap):
        stem=pos['stop_source'] if protection else 'stop'
        return stem+('_gap' if gap else '_touch')
    def close(i,fill,reason,timing,level=np.nan):
        nonlocal cash,pos
        b=rows[i];s=pos['sign'];ep=pos['price'];before=pos['start']
        cash+=s*pos['units']*(fill-ep)-pos['units']*fill*FEE
        r=dict(side='long' if s==1 else 'short',entry_i=pos['i'],exit_i=i,
            entry_time=rows[pos['i']].date,exit_bar_time=b.date,exit_time_precision=timing,
            entry_price=ep,exit_price=fill,exit_reason=reason,stop_level=level,
            exit_kind=pos['cfg'][0],exit_value=pos['cfg'][1],exit_confirmation=pos['c4'],early_loss_pct=pos['early_loss_pct'],stop_pct=100*pos['stop_fraction'],
            holding_hours_bar_start=(i-pos['i'])*bar_hours,status='closed')
        if bank is not None:r['indicator_R']=pos['indicator_R']
        if protection:r.update(protection_set_i=pos['stop_set_i'],protection_source=pos['stop_source'])
        for k,mode in enumerate(MODES):
            r[mode+'_pnl_initial']=cash[k]-before[k]
            r[mode+'_return_pct']=100*(cash[k]/before[k]-1)
            r[mode+'_equity_before']=before[k];r[mode+'_equity_after']=cash[k]
        trades.append(r);pos=None
        if policy and policy[0]=='cooldown' and r['simple_2x_pnl_initial']<0:
            # Same-side only, measured from exit candle end. Both sides remain enabled.
            blocked_until[s]=i+int(policy[1])
        if policy and ((policy[0]=='reset' and reason.startswith('stop')) or
                       (policy[0]=='risk_reset' and (reason.startswith('stop') or reason=='dual_loss_any'))):
            blocked[s]=True;reset_counts[s]=0
    for i,b in enumerate(rows):
        m=b.date.strftime('%Y-%m')
        if m!=month:month=m;pending=0;counts={1:0,-1:0}
        sold=False
        # Resting stop gap and scheduled market exit both execute at open;
        # label gap-stop priority. Later intrabar extremes cannot override exit.
        if pos is not None:
            sf=protective_fill(b,True)
            if sf is not None:
                close(i,sf[0],stop_reason(True),'open',sf[2]);sold=True;sell=False
            elif sell:
                close(i,b.open*(1-pos['sign']*SLIP),pos.get('pending_reason','core_'+pos['cfg'][0]),'open');sold=True
        if pending:
            assert pos is None and not sold
            if fixed_account_ratio>1:
                assert (cash[:2]+fixed_account_ratio-1>=1).all(), 'Insufficient actual account cash for fixed trade margin'
            s=pending;ep=b.open*(1+s*SLIP);start=cash.copy()
            sizes=np.array([1.,1.,cash[2],cash[3]])*LEV
            units=sizes/(ep*(1+FEE));cash-=units*ep*FEE
            p=entries[s][month]
            ir=None if bank is None else indicator_schedule[s][month]
            pos=dict(sign=s,i=i,price=ep,start=start,units=units,cfg=exits[s][month],c4=int(p[3]),
                     armed=s*indicator('slope',i-1,ir)>0,count=0,risk_count=0,indicator_R=ir)
            pos['early_loss_pct'],pos['stop_fraction']=(.5,threshold) if risk_schedule is None else risk_schedule[s][month]
            if protection:
                assert pos['stop_fraction'] is not None
                pos.update(active_stop=ep*(1-s*pos['stop_fraction']),stop_source='stop',stop_set_i=i-1)
            pos['best_close_return']=0.;pos['structure_count']=0
            if policy and policy[0]=='structure':
                history=rows[max(0,i-int(policy[1])):i]
                pos['structure_level']=min(v.low for v in history) if s==1 else max(v.high for v in history)
        pending=0;sell=False
        # Stop is active immediately after entry, including the entry candle.
        if pos is not None:
            sf=protective_fill(b)
            if sf is not None:
                close(i,sf[0],stop_reason(sf[1]),'open' if sf[1] else 'within_15m_bar_unknown',sf[2]);sold=True
        if pos is not None:
            s=pos['sign'];ir=pos['indicator_R'];slope=indicator('slope',i,ir)
            pos['armed']=pos['armed'] or s*slope>0
            cond=s*slope<0
            if pos['cfg'][0]=='momentum':cond=cond and s*b.momentum4 < -pos['cfg'][1]
            pos['count']=pos['count']+1 if pos['armed'] and cond else 0
            sell=pos['count']>=pos['c4']
            pos['pending_reason']='core_'+pos['cfg'][0]
            if early_exit:
                risk=100*s*(b.close/pos['price']-1)<=-pos['early_loss_pct'] and s*b.momentum4<0 and s*indicator('model_accel_scaled',i,ir)<0
                pos['risk_count']=pos['risk_count']+1 if risk else 0
                if not sell and pos['risk_count']>=2:
                    sell=True;pos['pending_reason']='dual_loss_any'
            if policy and policy[0]=='trail':
                # Close-only trail: no assumed ordering of intrabar high/low.
                price_return=100*s*(b.close/pos['price']-1)
                pos['best_close_return']=max(pos['best_close_return'],price_return)
                if not sell and pos['best_close_return']>=policy[1] and pos['best_close_return']-price_return>=policy[2]:
                    sell=True;pos['pending_reason']='profit_giveback'
            if policy and policy[0]=='structure':
                failed=b.close<pos['structure_level'] if s==1 else b.close>pos['structure_level']
                pos['structure_count']=pos['structure_count']+1 if failed else 0
                if not sell and pos['structure_count']>=2:
                    sell=True;pos['pending_reason']='structure_failure'
            if policy and policy[0]=='fast_loss':
                failed=100*s*(b.close/pos['price']-1)<=-policy[1] and s*b.momentum4<0
                pos['structure_count']=pos['structure_count']+1 if failed else 0
                if not sell and pos['structure_count']>=2:
                    sell=True;pos['pending_reason']='loss_momentum'
            if policy and policy[0]=='stagnation':
                price_return=100*s*(b.close/pos['price']-1)
                pos['best_close_return']=max(pos['best_close_return'],price_return)
                if not sell and i-pos['i']+1>=policy[1] and pos['best_close_return']<.5 and price_return<0:
                    sell=True;pos['pending_reason']='stagnation'
            if protection:
                # Only completed closes can tighten the resting stop for later bars.
                price_return=100*s*(b.close/pos['price']-1)
                pos['best_close_return']=max(pos['best_close_return'],price_return)
                candidates=[]
                be=protection.get('be_trigger_pct')
                if be is not None and pos['best_close_return']>=be:
                    level=pos['price']*(1+s*FEE)/(1-s*FEE)/(1-s*SLIP)
                    candidates.append((level,'breakeven'))
                activation=protection.get('trail_activation_pct')
                if activation is not None and pos['best_close_return']>=activation:
                    level=pos['price']*(1+s*(pos['best_close_return']-protection['trail_distance_pct'])/100)
                    candidates.append((level,'trailing'))
                for level,source in candidates:
                    if s*(level-pos['active_stop'])>1e-12:
                        pos.update(active_stop=level,stop_source=source,stop_set_i=i)
        for s in allowed:
            c1,c2,c3,_,length=entries[s][month]
            ir=None if bank is None else indicator_schedule[s][month]
            cond=s*b.momentum4>c1 and s*indicator('slope_scaled',i,ir)>c2 and s*indicator('model_accel_scaled',i,ir)>c3
            if require_slope_trend:cond=cond and s*indicator(entry_trend_field or 'trend'+str(int(length)),i,ir)>0
            if entry_gate is not None:cond=cond and bool(entry_gate[s][i])
            if blocked[s] and not sold:
                reset_counts[s]=reset_counts[s]+1 if not cond else 0
                if reset_counts[s]>=2:blocked[s]=False;reset_counts[s]=0
            counts[s]=counts[s]+1 if pos is None and not sold and cond and not blocked[s] and i>blocked_until[s] else 0
            if counts[s]>=2:pending=s
        eq=cash.copy() if pos is None else cash+pos['sign']*pos['units']*(b.close-pos['price'])
        # Fixed-trade cumulative PNL is normalized by one trade margin, not total account capital.
        assert (eq[:2]+fixed_account_ratio-1>0).all() and (eq[2:]>0).all(), 'Insolvency: actual account or compound capital depleted'
        q.append(eq.copy())
    if pos is not None:
        b=rows[-1];r=dict(side='long' if pos['sign']==1 else 'short',entry_i=pos['i'],exit_i=len(d),entry_time=rows[pos['i']].date,
            exit_bar_time=pd.NaT,exit_time_precision='unclosed',entry_price=pos['price'],exit_price=np.nan,
            exit_reason='year_end_mark',stop_level=np.nan,exit_kind=pos['cfg'][0],exit_value=pos['cfg'][1],
            exit_confirmation=pos['c4'],early_loss_pct=pos['early_loss_pct'],stop_pct=100*pos['stop_fraction'],holding_hours_bar_start=(len(d)-pos['i'])*bar_hours,status='open_marked_at_end')
        if bank is not None:r['indicator_R']=pos['indicator_R']
        for k,mode in enumerate(MODES):
            r[mode+'_pnl_initial']=q[-1][k]-pos['start'][k]
            r[mode+'_return_pct']=100*(q[-1][k]/pos['start'][k]-1)
            r[mode+'_equity_before']=pos['start'][k];r[mode+'_equity_after']=q[-1][k]
        trades.append(r)
    return pd.DataFrame(trades),np.asarray(q)
