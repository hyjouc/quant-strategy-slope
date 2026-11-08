"""Independent scalar order reconstruction with seven-parameter exit risk."""
import numpy as np
import pandas as pd
def independent(d,params,key,allow_open=False,allowed_sides=(1,-1),indicator_bank=None,require_slope_trend=True,entry_trend_field=None,entry_gate=None):
    """Rebuild full order flow without calling production signals/stop helpers."""
    config={(r.month,r.side):r for r in params.itertuples()}
    cash=1.;position=None;entry_order=None;exit_order=None;streak={1:0,-1:0};month=None;curve=[];records=[]
    bars=list(d.itertuples(index=False))
    bank=None if indicator_bank is None else {r:{c:f[c].to_numpy() for c in f.columns if c in ['slope','slope_scaled','model_accel_scaled'] or c.startswith('trend')} for r,f in indicator_bank.items()}
    def value(c,i,r=None):return getattr(bars[i],c) if bank is None else bank[r][c][i]
    for i,b in enumerate(bars):
        m=b.date.strftime('%Y-%m')
        if month!=m:month=m;entry_order=None;streak={1:0,-1:0}
        did_exit=False
        def liquidate(price,reason):
            nonlocal cash,position,did_exit,exit_order
            z=position;s=z['s'];fill=price*(1-s*0.)
            cash+=s*z['units']*(fill-z['ep'])-z['units']*fill*.0005
            records.append(dict(entry_i=z['i'],exit_i=i,side='long' if s==1 else 'short',entry_price=z['ep'],exit_price=fill,exit_reason=reason,pnl=cash-z['before']))
            position=None;did_exit=True;exit_order=None
        if position is not None:
            s=position['s'];level=position['ep']*(1-s*position['cfg'].stop_pct/100)
            gap=s*(b.open-level)<=0
            if gap:liquidate(b.open,'stop_gap')
            elif exit_order is not None:liquidate(b.open,exit_order)
        if entry_order is not None:
            assert position is None and not did_exit
            s=entry_order;cfg=config[(m,'long' if s==1 else 'short')];ep=b.open*(1+s*0.)
            ir=None if bank is None else float(cfg.R)
            units=2/(ep*1.0005);position=dict(i=i,s=s,ep=ep,units=units,before=cash,cfg=cfg,armed=s*value('slope',i-1,ir)>0,normal_count=0,early_count=0,indicator_R=ir)
            cash-=units*ep*.0005
        entry_order=None;exit_order=None
        if position is not None:
            z=position;s=z['s'];level=z['ep']*(1-s*z['cfg'].stop_pct/100)
            if (b.low<=level if s==1 else b.high>=level):liquidate(b.open if s*(b.open-level)<=0 else level,'stop_gap' if s*(b.open-level)<=0 else 'stop_touch')
        if position is not None:
            z=position;s=z['s'];cfg=z['cfg'];ir=z['indicator_R'];slope=value('slope',i,ir);z['armed']|=s*slope>0
            opposite=s*slope<0 and (cfg.exit_kind=='sign' or s*b.momentum4 < -cfg.exit_value)
            z['normal_count']=z['normal_count']+1 if z['armed'] and opposite else 0
            loss=100*s*(b.close/z['ep']-1)<=-cfg.early_loss_pct
            z['early_count']=z['early_count']+1 if loss and s*b.momentum4<0 and s*value('model_accel_scaled',i,ir)<0 else 0
            if z['normal_count']>=int(cfg.C4):exit_order='core_'+cfg.exit_kind
            elif z['early_count']>=2:exit_order='dual_loss_any'
        for s,name in [(1,'long'),(-1,'short')]:
            if s not in allowed_sides:
                streak[s]=0
                continue
            cfg=config[(m,name)]
            allowed=True
            if key!='baseline' and s==-1:allowed=b.fast<=b.slow
            if key=='symmetric' and s==1:allowed=b.fast>=b.slow
            ir=None if bank is None else float(cfg.R)
            allowed=allowed and (entry_gate is None or bool(entry_gate[s][i]))
            signal=allowed and s*b.momentum4>cfg.C1 and s*value('slope_scaled',i,ir)>cfg.C2 and s*value('model_accel_scaled',i,ir)>cfg.C3
            if require_slope_trend:signal=signal and s*value(entry_trend_field or 'trend'+str(int(cfg.L)),i,ir)>0
            streak[s]=streak[s]+1 if position is None and not did_exit and signal else 0
            if streak[s]>=2:entry_order=s
        eq=cash if position is None else cash+position['s']*position['units']*(b.close-position['ep'])
        curve.append(eq)
    assert allow_open or position is None,'audit expects the observed closed year end'
    return pd.DataFrame(records),np.array(curve)
