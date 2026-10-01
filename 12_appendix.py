import pandas as pd, numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf, json
import warnings; warnings.filterwarnings('ignore')
d=pd.read_csv('../work/policy_coverage.csv')
d['year']=pd.Categorical(d.year,['Freshman','Sophomore','Junior','Senior','Grad student'])
tr=d[d.is_train==1]
COVS=['Personal Property','Additional Living Expense','Guest Medical','Liability']
P=sm.families.Poisson(); G=sm.families.Gamma(sm.families.links.Log())
FREQ="claims ~ greek_f + off_f + C(year)"; SEV="loss ~ greek_f + off_f + spr_f + C(year)"

def tab(m):
    return pd.DataFrame({'coef':m.params.round(4),'se':m.bse.round(4),'p':m.pvalues.round(4),
                         'exp':np.exp(m.params).round(3)})
print("########## FREQUENCY GLM (Poisson, log link) ##########")
for c in COVS:
    t=tr[tr.coverage==c]; m=smf.glm(FREQ,data=t,family=P).fit()
    print(f"\n--- {c} | n={len(t)} claims={t.claims.sum()} | deviance={m.deviance:.1f} df={m.df_resid} AIC={m.aic:.1f} ---")
    print(tab(m).to_string())
print("\n\n########## SEVERITY GLM (Gamma, log link) ##########")
for c in COVS:
    t=tr[(tr.coverage==c)&(tr.claims>0)]; m=smf.glm(SEV,data=t,family=G).fit()
    print(f"\n--- {c} | n={len(t)} | deviance={m.deviance:.1f} df={m.df_resid} AIC={m.aic:.1f} | dispersion={m.scale:.3f} ---")
    print(tab(m).to_string())
print("\n\n########## DIAGNOSTICS ##########")
for c in COVS:
    t=tr[tr.coverage==c]; m=smf.glm(FREQ,data=t,family=P).fit()
    ts=t[t.claims>0]
    g=smf.glm(SEV,data=ts,family=G).fit()
    ig=smf.glm(SEV,data=ts,family=sm.families.InverseGaussian(sm.families.links.Log())).fit()
    print(f"{c:28s} pearson/df={(m.resid_pearson**2).sum()/m.df_resid:.3f}  "
          f"AIC_gamma={g.aic:.1f}  AIC_invgauss={ig.aic:.1f}  CV={ts.severity.std()/ts.severity.mean():.2f}  Z={min(1,np.sqrt(t.claims.sum()/1082)):.2f}")
print("\n\n########## BASE CELL DECOMPOSITION ##########")
prm=json.load(open('../outputs/model_params.json'))
for c in COVS:
    f=prm[c]
    print(f"{c:28s} base_freq={f['base_freq']:.4f} base_sev={f['base_sev']:.0f} balance={f['balance_factor']:.4f} base_pp={f['base_pp']:.2f}")
