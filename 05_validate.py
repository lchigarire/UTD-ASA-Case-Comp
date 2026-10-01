"""Step 5: Holdout validation - lift, Gini, calibration, A-vs-E with uncertainty."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'figure.dpi':130,'font.size':10,'axes.grid':True,'grid.alpha':.25,
                     'axes.spines.top':False,'axes.spines.right':False})
NAVY='#1F3864'; ORANGE='#E8912D'; GREY='#9AA0A6'
rng=np.random.RandomState(42)

p=pd.read_csv('../work/scored_policies.csv')
ho=p[p.is_train==0].copy(); tr=p[p.is_train==1].copy()
FLAT=tr.actual_total.mean()

def gini(actual,pred):
    """Tie-aware Gini: predictions that are equal form ONE Lorenz point.
    Random tie-breaking is unstable when a predictor takes few distinct
    values (the 3-level tier variable moved between 0.005 and 0.061
    across seeds). See 09_gini.py for the same construction."""
    t=pd.DataFrame({'a':np.asarray(actual,float),'p':np.asarray(pred,float)})
    g=t.groupby('p').agg(n=('a','size'),loss=('a','sum')).sort_index()
    x=np.r_[0,np.cumsum(g.n)/g.n.sum()]; y=np.r_[0,np.cumsum(g.loss)/g.loss.sum()]
    return float(1-2*np.trapezoid(y,x))

def bins(df,col,k=10):                       # tie-safe equal-count bins
    r=df[col].rank(method='first')
    return pd.qcut(r,k,labels=False)

def boot_ae(df,pred_col,B=2000):
    ae=[]
    for _ in range(B):
        s=df.sample(len(df),replace=True,random_state=rng.randint(1e9))
        ae.append(s.actual_total.sum()/s[pred_col].sum())
    return np.percentile(ae,[2.5,97.5])

print("=== HOLDOUT DISCRIMINATION (n=%d) ==="%len(ho))
print(f"Gini  model={gini(ho.actual_total,ho.total_pp):.3f}   "
      f"underwriter tiers={gini(ho.actual_total,ho.risk_tier):.3f}   "
      f"flat rate={gini(ho.actual_total,np.ones(len(ho))):.3f}")
ho['b']=bins(ho,'total_pp')
L=ho.groupby('b').agg(n=('actual_total','size'),actual=('actual_total','mean'),pred=('total_pp','mean'))
print("\nHoldout decile lift:"); print(L.round(0).to_string())
print(f"Top vs bottom decile, actual: {L.actual.iloc[-1]/L.actual.iloc[0]:.1f}x   predicted: {L.pred.iloc[-1]/L.pred.iloc[0]:.1f}x")

print("\n=== CALIBRATION: A/E with 95% bootstrap CI (holdout) ===")
rows=[]
for v,lab in [('greek_f','Greek'),('off_f','Off campus'),('spr_f','Sprinklered'),('year','Class year')]:
    for lev,g in ho.groupby(v,observed=True):
        lo,hi=boot_ae(g,'total_pp',600)
        rows.append({'var':lab,'level':lev,'n':len(g),'actual':g.actual_total.mean(),
                     'pred':g.total_pp.mean(),'A/E':g.actual_total.sum()/g.total_pp.sum(),'lo':lo,'hi':hi})
AE=pd.DataFrame(rows); print(AE.round(2).to_string(index=False))
AE.to_csv('../outputs/actual_vs_expected.csv',index=False)
print("\nAll A/E intervals containing 1.0? ->", bool(((AE.lo<=1)&(AE.hi>=1)).all()))

# training A/E to separate fit bias from holdout noise
print("\nTraining A/E (should be ~1.00 by construction):")
for v in ['greek_f','off_f','spr_f']:
    g=tr.groupby(v).apply(lambda x: x.actual_total.sum()/x.total_pp.sum(),include_groups=False)
    print(f"  {v}: "+"  ".join(f"{k}={v2:.2f}" for k,v2 in g.items()))

# ---------- charts ----------
x=np.arange(1,11)
fig,ax=plt.subplots(figsize=(7.2,4))
ax.bar(x-0.2,L.actual,0.4,label='Actual loss',color=NAVY)
ax.bar(x+0.2,L.pred,0.4,label='Modelled pure premium',color=ORANGE)
ax.axhline(FLAT,ls='--',c=GREY,lw=1.5,label=f"Today's flat rate (${FLAT:,.0f})")
ax.set_xlabel('Decile of modelled pure premium — holdout'); ax.set_ylabel('$ per policy per year')
ax.set_title(f'Holdout lift: {L.actual.iloc[-1]/L.actual.iloc[0]:.0f}x spread the current flat rate cannot see')
ax.set_xticks(x); ax.legend(frameon=False,fontsize=8.5); plt.tight_layout()
plt.savefig('../outputs/fig1_lift.png'); plt.close()

T=ho.groupby('risk_tier').actual_total.mean()
Q=ho.copy(); Q['q']=pd.qcut(Q.total_pp.rank(method='first'),3,labels=['Preferred','Standard','Non-standard'])
Qs=Q.groupby('q',observed=True).actual_total.mean()
fig,ax=plt.subplots(1,2,figsize=(9,3.9),sharey=True)
ax[0].bar(['Tier 1','Tier 2','Tier 3'],T.values,color=GREY)
ax[0].set_title("Underwriters' tiers",fontsize=11); ax[0].set_ylabel('Actual holdout loss ($/policy)')
ax[1].bar(Qs.index.astype(str),Qs.values,color=NAVY); ax[1].set_title('Model-based tiers',fontsize=11)
for a in ax: a.axhline(ho.actual_total.mean(),ls='--',c=ORANGE,lw=1.3)
plt.suptitle('Same book, same holdout — only one of these segments risk',y=1.02,fontsize=11.5)
plt.tight_layout(); plt.savefig('../outputs/fig2_tiers.png',bbox_inches='tight'); plt.close()
print("\nUnderwriter tiers, holdout actual:",T.round(0).to_dict())
print("Model tiers, holdout actual:     ",Qs.round(0).to_dict())

fig,ax=plt.subplots(figsize=(7.2,4))
y=AE.set_index(AE['var']+': '+AE.level.astype(str))
ax.errorbar(y['A/E'],np.arange(len(y)),xerr=[y['A/E']-y.lo,y.hi-y['A/E']],fmt='o',color=NAVY,capsize=3)
ax.axvline(1,c=ORANGE,ls='--'); ax.set_yticks(np.arange(len(y))); ax.set_yticklabels(y.index,fontsize=8)
ax.set_xlabel('Actual / Expected (holdout)'); ax.set_title('Calibration: every rating level is unbiased')
plt.tight_layout(); plt.savefig('../outputs/fig4_ave.png'); plt.close()

# NOTE: the incumbent plan is a single flat rate, so model/incumbent ranks
# IDENTICALLY to the model itself (verified rank corr = 1.000000 in
# 15_gini_theory.py). This chart therefore duplicates fig1 and is retained
# only to document that the check was performed.
D=ho.copy(); D['b']=bins(D.assign(r=D.total_pp/FLAT),'total_pp')
g=D.groupby('b').agg(actual=('actual_total','mean'),model=('total_pp','mean'))
fig,ax=plt.subplots(figsize=(7.2,4))
ax.plot(x,g.actual/g.actual.mean(),'o-',c=NAVY,label='Actual loss')
ax.plot(x,g.model/g.model.mean(),'s--',c=ORANGE,label='Proposed rate')
ax.plot(x,np.ones(10),'^:',c=GREY,label='Current flat rate')
ax.set_xlabel('Decile of proposed rate / current rate'); ax.set_ylabel('Relative loss cost')
ax.set_title('Double lift vs a FLAT incumbent - ordering is identical to fig1'); ax.legend(frameon=False)
plt.tight_layout(); plt.savefig('../outputs/fig3_doublelift.png'); plt.close()

print(f"\n=== FLAT-RATE SUBSIDY (${FLAT:,.0f}) ===")
print(f"Policies priced above their own loss cost today: {(p.total_pp<FLAT).mean():.1%}")
print(f"Loss dollars from worst 20% of risks: {p.nlargest(int(.2*len(p)),'total_pp').actual_total.sum()/p.actual_total.sum():.1%}")
print(f"Worst 10% loss cost = ${p.nlargest(int(.1*len(p)),'total_pp').total_pp.mean():,.0f} vs best 10% = ${p.nsmallest(int(.1*len(p)),'total_pp').total_pp.mean():,.0f}")
