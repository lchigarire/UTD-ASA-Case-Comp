import pandas as pd, numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.stats import chi2
import warnings; warnings.filterwarnings('ignore')
d=pd.read_csv('../work/policy_coverage.csv'); tr=d[d.is_train==1]
COVS=['Personal Property','Additional Living Expense','Guest Medical','Liability']
P=sm.families.Poisson(); G=sm.families.Gamma(sm.families.links.Log())
print("=== GPA on SEVERITY ===")
for c in COVS:
    t=tr[(tr.coverage==c)&(tr.claims>0)]
    m=smf.glm("loss ~ greek_f+off_f+spr_f+C(year)+gpa_c",data=t,family=G).fit()
    print(f"{c:28s} n={len(t):4d} gpa={m.params['gpa_c']:+.3f} p={m.pvalues['gpa_c']:.4f}")
print("\n=== YEAR on FREQUENCY (relativities, Freshman=1) ===")
for c in COVS:
    t=tr[tr.coverage==c]
    m=smf.glm("claims ~ greek_f+off_f+spr_f+C(year)",data=t,family=P).fit()
    ys={k.split('T.')[1][:-1]:np.exp(v) for k,v in m.params.items() if 'year' in k}
    base=smf.glm("claims ~ greek_f+off_f+spr_f",data=t,family=P).fit()
    lr=2*(m.llf-base.llf)
    print(f"{c:28s} LRp={1-chi2.cdf(lr,4):.3f}  "+"  ".join(f"{k}={v:.2f}" for k,v in ys.items()))
print("\n=== SPRINKLER on FREQUENCY (should be null) ===")
for c in COVS:
    t=tr[tr.coverage==c]
    m=smf.glm("claims ~ greek_f+off_f+spr_f+C(year)",data=t,family=P).fit()
    print(f"{c:28s} spr={m.params['spr_f']:+.3f} p={m.pvalues['spr_f']:.3f}")
print("\n=== Combined year effect on PURE PREMIUM (empirical, train) ===")
print(tr.pivot_table(index='year',columns='coverage',values='loss',aggfunc='mean').round(1).to_string())
