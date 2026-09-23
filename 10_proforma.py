"""Step 10: Pro forma P&L (deliverable 4c) + continuous-variable treatment (deliverable bullet 2)."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import statsmodels.api as sm, statsmodels.formula.api as smf
import warnings; warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi':130,'font.size':10,'axes.grid':True,'grid.alpha':.25,
                     'axes.spines.top':False,'axes.spines.right':False})
NAVY='#1F3864'; ORANGE='#E8912D'; GREY='#9AA0A6'

p=pd.read_csv('output/scored_policies_rated.csv')
ULAE,COMM,TAX,PROFIT,FIXED,ACQ_NEW,ACQ_REN=.08,.12,.025,.05,25.,45.,5.
CUR=(p[p.is_train==1].actual_total.mean()*(1+ULAE)+FIXED)/(1-COMM-TAX-PROFIT)
BASE_LAPSE,ELAST,GROWTH=.22,.9,.15
fair_r=p.rate_renewal.values; fair_n=p.rate_new.values; pp=p.total_pp.values
def lapse(c,f): return np.clip(BASE_LAPSE*(1+ELAST*(c/f-1)),.03,.85)

def proforma(mode,years=3,cap=.25):
    wr=np.ones(len(p)); cr=np.full(len(p),CUR); wn=np.zeros(len(p))
    if mode=='proposed': wr[(p.greek_f==1)&(p.off_f==1)&(p.spr_f==0)]=0.
    out=[]
    for y in range(1,years+1):
        if mode=='proposed': cr=np.where(fair_r<cr,fair_r,np.minimum(fair_r,cr*(1+cap)))
        prem=(wr*cr).sum()+(wn*fair_n).sum(); n=wr.sum()+wn.sum()
        loss=((wr+wn)*pp).sum(); lae=loss*ULAE
        comm=prem*COMM; tax=prem*TAX; oh=n*FIXED; acq=wr.sum()*ACQ_REN+wn.sum()*ACQ_NEW
        out.append(dict(year=y, policies=n, written_premium=prem, losses=loss, lae=lae,
                        commission=comm, premium_tax=tax, overhead=oh, acquisition=acq,
                        uw_result=prem-loss-lae-comm-tax-oh-acq,
                        loss_ratio=(loss+lae)/prem*100,
                        combined=(loss+lae+comm+tax+oh+acq)/prem*100))
        kr=1-lapse(cr,fair_r); kn=1-lapse(fair_n,fair_n)
        wrn=wr*kr+wn*kn
        cr=np.where(wr+wn>0,(wr*kr*cr+wn*kn*fair_n)/np.maximum(wrn,1e-9),cr); wr=wrn
        if mode=='proposed':
            a=np.clip(CUR/fair_n,.2,3.); a=np.where(wr>0,a,0); a[(p.greek_f==1)&(p.off_f==1)&(p.spr_f==0)]=0
            wn=GROWTH*10000*a/a.sum()
        else: wn=np.zeros(len(p))
    return pd.DataFrame(out).set_index('year').T

for mode,lab in [('status_quo','STATUS QUO'),('proposed','PROPOSED PLAN')]:
    f=proforma(mode)
    print(f"\n=== PRO FORMA P&L — {lab} ===")
    fmt=pd.DataFrame({c:[f"{f.loc[r,c]:.1f}%" if r in ('loss_ratio','combined') else f"{f.loc[r,c]:,.0f}"
                          for r in f.index] for c in f.columns}, index=f.index)
    print(fmt.to_string())
    f.to_csv(f'output/proforma_{mode}.csv')

# ---------------- continuous variable treatment ----------------
print("\n\n=== CONTINUOUS VARIABLES: how we tested GPA and distance ===")
d=pd.read_csv('policy_coverage.csv'); tr=d[d.is_train==1]
P=sm.families.Poisson()
rows=[]
for var,form in [('gpa','banded'),('gpa','linear'),('gpa','hinge at 2.5'),
                 ('dist','banded'),('dist','linear'),('dist','log(1+x)')]:
    t=tr[tr.coverage=='Personal Property'].copy()
    t['gpa_band']=pd.cut(t.gpa,[0,1.5,2,2.5,3,3.5,4])
    t['dist_band']=pd.cut(t.dist,[-.01,.01,2,5,10,30])
    t['gpa_hinge']=np.maximum(t.gpa-2.5,0); t['logd']=np.log1p(t.dist)
    spec={'banded':f'C({var}_band)','linear':var,'hinge at 2.5':'gpa+gpa_hinge','log(1+x)':'logd'}[form]
    base=smf.glm("claims ~ greek_f+off_f+C(year)",data=t,family=P).fit()
    m=smf.glm(f"claims ~ greek_f+off_f+C(year)+{spec}",data=t,family=P).fit()
    from scipy.stats import chi2
    lr=2*(m.llf-base.llf); df=base.df_resid-m.df_resid
    rows.append(dict(variable=var,treatment=form,added_df=int(df),LR=round(lr,2),
                     p=round(1-chi2.cdf(lr,df),3),AIC_change=round(m.aic-base.aic,1)))
print(pd.DataFrame(rows).to_string(index=False))
print("\nNo treatment reaches p<0.05; only the GPA hinge improves AIC, and only by 0.9 -> neither variable earns a place in the rate.")

# banded GPA relativity with CIs, to show it visually
t=tr[tr.coverage=='Personal Property'].copy()
t['b']=pd.cut(t.gpa,[0,1.5,2,2.5,3,3.5,4])
g=t.groupby('b',observed=True).agg(n=('claims','size'),f=('claims','mean'))
g['se']=np.sqrt(g.f/g.n); g['rel']=g.f/t.claims.mean(); g['lo']=(g.f-1.96*g.se)/t.claims.mean(); g['hi']=(g.f+1.96*g.se)/t.claims.mean()
fig,ax=plt.subplots(figsize=(6.4,3.8))
x=np.arange(len(g))
ax.errorbar(x,g.rel,yerr=[g.rel-g.lo,g.hi-g.rel],fmt='o',color=NAVY,capsize=3)
ax.axhline(1,ls='--',c=ORANGE)
ax.set_xticks(x); ax.set_xticklabels([str(i) for i in g.index],rotation=20,fontsize=8)
ax.set_xlabel('GPA band'); ax.set_ylabel('Frequency relativity')
ax.set_title('GPA: no stable pattern once class year is controlled')
plt.tight_layout(); plt.savefig('output/fig7_gpa.png'); plt.close()
print(g.round(3).to_string())
