import pandas as pd, numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf
import warnings; warnings.filterwarnings('ignore')
d = pd.read_csv('../work/policy_coverage.csv'); tr=d[d.is_train==1]
COVS=['Personal Property','Additional Living Expense','Guest Medical','Liability']
P=sm.families.Poisson(); G=sm.families.Gamma(sm.families.links.Log())
IG=sm.families.InverseGaussian(sm.families.links.Log())

print("=== Is the Guest Medical distance effect real, or the >10mi tail? ===")
t=tr[tr.coverage=='Guest Medical']
for lab,sub in [('all off-campus',t[t.off_f==1]),('off-campus, <10mi',t[(t.off_f==1)&(t.dist<10)])]:
    m=smf.glm("claims ~ greek_f+spr_f+dist",data=sub,family=P).fit()
    print(f"{lab:22s} n={len(sub):5d} dist coef={m.params['dist']:+.4f} p={m.pvalues['dist']:.4f}")

print("\n=== GPA: linear freq effect per coverage (train) ===")
for c in COVS:
    t=tr[tr.coverage==c]
    m=smf.glm("claims ~ greek_f+off_f+C(year)+gpa_c",data=t,family=P).fit()
    print(f"{c:28s} gpa coef={m.params['gpa_c']:+.3f} p={m.pvalues['gpa_c']:.4f}  -> per +1 GPA x{np.exp(m.params['gpa_c']):.3f}")

print("\n=== Severity family comparison (AIC, lower better) ===")
for c in COVS:
    t=tr[(tr.coverage==c)&(tr.claims>0)]
    f="loss ~ greek_f+off_f+spr_f+C(year)"
    a=smf.glm(f,data=t,family=G).fit().aic
    b=smf.glm(f,data=t,family=IG).fit().aic
    ln=smf.ols("np.log(loss) ~ greek_f+off_f+spr_f+C(year)",data=t).fit()
    print(f"{c:28s} Gamma={a:10.1f}  InvGauss={b:10.1f}  (lognormal R2={ln.rsquared:.3f})")

print("\n=== Year: collapse to Fresh/Soph | Junior | Senior | Grad? severity coefs ===")
for c in COVS:
    t=tr[(tr.coverage==c)&(tr.claims>0)]
    m=smf.glm("loss ~ greek_f+off_f+spr_f+C(year)",data=t,family=G).fit()
    ys={k.split('T.')[1][:-1]:np.exp(v) for k,v in m.params.items() if 'year' in k}
    print(f"{c:28s} " + "  ".join(f"{k}={v:.2f}" for k,v in ys.items()))

print("\n=== study / gender joint significance (freq, LR test) ===")
for c in COVS:
    t=tr[tr.coverage==c]
    base=smf.glm("claims ~ greek_f+off_f+C(year)",data=t,family=P).fit()
    for extra in ['C(study)','C(gender)']:
        m=smf.glm(f"claims ~ greek_f+off_f+C(year)+{extra}",data=t,family=P).fit()
        from scipy.stats import chi2
        lr=2*(m.llf-base.llf); dfree=base.df_resid-m.df_resid
        print(f"{c:28s} {extra:10s} LR={lr:5.2f} df={dfree:.0f} p={1-chi2.cdf(lr,dfree):.3f}")
