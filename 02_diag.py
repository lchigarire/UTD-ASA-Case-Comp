import pandas as pd, numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf
import warnings; warnings.filterwarnings('ignore')
d = pd.read_csv('../work/policy_coverage.csv')
tr = d[d.is_train==1]
COVS=['Personal Property','Additional Living Expense','Guest Medical','Liability']

print("=== OVERDISPERSION (Poisson deviance/df, train) ===")
for c in COVS:
    t=tr[tr.coverage==c]
    m=smf.glm("claims ~ greek_f+off_f+spr_f+C(year)+gpa_c", data=t, family=sm.families.Poisson()).fit()
    pear = (m.resid_pearson**2).sum()/m.df_resid
    print(f"{c:28s} claims={t.claims.sum():5d}  pearson/df={pear:.3f}  maxclaims={t.claims.max()}")

print("\n=== DISTANCE shape, OFF-CAMPUS only (freq & sev) ===")
oc = tr[tr.off_f==1].copy()
oc['db']=pd.cut(oc.dist,[0,1,2,3,5,7,10,15,30])
for c in COVS:
    t=oc[oc.coverage==c]
    print(f"\n{c}")
    print(t.groupby('db',observed=True).agg(n=('claims','size'),freq=('claims','mean'),
          sev=('severity','mean'),pp=('loss','mean')).round(2).to_string())

print("\n=== GPA shape (all, pure premium) ===")
t=tr.copy(); t['gb']=pd.cut(t.gpa,[0,1,1.5,2,2.5,3,3.5,4])
print(t.pivot_table(index='gb',columns='coverage',values='loss',aggfunc='mean',observed=True).round(1).to_string())
print("\nGPA vs freq")
print(t.pivot_table(index='gb',columns='coverage',values='claims',aggfunc='mean',observed=True).mul(100).round(2).to_string())

print("\n=== Severity distribution shape: PP claims ===")
for c in COVS:
    s=tr.loc[(tr.coverage==c)&(tr.claims>0),'severity']
    print(f"{c:28s} n={len(s):4d} mean={s.mean():9.1f} cv={s.std()/s.mean():.2f} skew={s.skew():.2f} min={s.min():.0f} max={s.max():.0f}")

print("\n=== Interaction check: greek x off_campus (PP pure prem) ===")
print(tr[tr.coverage=='Personal Property'].pivot_table(index='greek',columns='off_campus',values='loss',aggfunc=['mean','size']).round(1).to_string())
