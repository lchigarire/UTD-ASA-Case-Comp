import pandas as pd, numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf
import warnings; warnings.filterwarnings('ignore')
d=pd.read_csv('../work/policy_coverage.csv'); tr=d[d.is_train==1]
COVS=['Personal Property','Additional Living Expense','Guest Medical','Liability']

print("Decomposing the off-balance factor\n")
print(f"{'coverage':28s} {'lambda':>7s} {'E[N|N>0]':>9s} {'Gamma rel-bal':>14s} {'balance used':>13s}")
for c in COVS:
    t=tr[tr.coverage==c]; tc=t[t.claims>0]
    lam=t.claims.mean(); enn=t[t.claims>0].claims.mean()
    # Gamma log-link balance check: mean(y/mu) == 1 but mean(y) vs mean(mu)?
    m=smf.glm("loss ~ greek_f+off_f+spr_f+C(year)",data=tc,
              family=sm.families.Gamma(sm.families.links.Log())).fit()
    mu=m.fittedvalues
    relbal=(tc.loss/mu).mean()      # == 1 by construction
    absbal=tc.loss.mean()/mu.mean() # NOT 1 in general
    print(f"{c:28s} {lam:7.4f} {enn:9.3f} {absbal:14.4f} ", end="")
    # what the pipeline actually used
    fm=smf.glm("claims ~ greek_f+off_f+C(year)",data=t,family=sm.families.Poisson()).fit()
    pred=fm.fittedvalues*mu.reindex(t.index).fillna(0)  # placeholder
    print()
    print(f"{'':28s}   mean(y/mu)={relbal:.4f} (1.000 by construction)")

print("\nEffect of modelling TOTAL loss given a claim, vs PER-CLAIM severity:")
for c in COVS:
    t=tr[tr.coverage==c]; tc=t[t.claims>0]
    print(f"  {c:28s} E[N|N>0]={tc.claims.mean():.4f}  -> lambda*E[L|N>0] overstates E[L] by {(tc.claims.mean()-1)*100:.2f}%")
