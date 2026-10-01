import pandas as pd, numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.stats import chi2
import warnings; warnings.filterwarnings('ignore')
d=pd.read_csv('../work/policy_coverage.csv'); tr=d[d.is_train==1]
COVS=['Personal Property','Additional Living Expense','Guest Medical','Liability']
F="claims ~ greek_f + off_f + C(year)"

print("Claim-count distribution (training, Personal Property):")
print(tr[tr.coverage=='Personal Property'].claims.value_counts().sort_index().to_string())

print("\n=== Poisson vs Negative Binomial (does NB find heterogeneity Poisson missed?) ===")
for c in COVS:
    t=tr[tr.coverage==c]
    p=smf.glm(F,data=t,family=sm.families.Poisson()).fit()
    # profile alpha for NB
    best=(None,np.inf)
    for a in [1e-6,0.05,0.1,0.25,0.5,1.0,2.0]:
        nb=smf.glm(F,data=t,family=sm.families.NegativeBinomial(alpha=a)).fit()
        if nb.aic<best[1]: best=(a,nb.aic)
    lr=2*(smf.glm(F,data=t,family=sm.families.NegativeBinomial(alpha=best[0])).fit().llf - p.llf)
    print(f"{c:28s} Pearson/df={(p.resid_pearson**2).sum()/p.df_resid:.3f}  AIC_Poisson={p.aic:8.1f}  "
          f"best alpha={best[0]:<8g} AIC_NB={best[1]:8.1f}  LR={lr:+.2f}")

print("\n=== Wald vs Likelihood-Ratio p-values, Greek term ===")
for c in COVS:
    t=tr[tr.coverage==c]
    full=smf.glm(F,data=t,family=sm.families.Poisson()).fit()
    red=smf.glm("claims ~ off_f + C(year)",data=t,family=sm.families.Poisson()).fit()
    lr=2*(full.llf-red.llf); plr=1-chi2.cdf(lr,1)
    print(f"{c:28s} Wald p={full.pvalues['greek_f']:.2e}   LR p={plr:.2e}   LR stat={lr:.1f}")
