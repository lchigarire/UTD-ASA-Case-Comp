"""
Step 8: Challenger model. Does a GBM find signal the GLM missed --
including in the variables we dropped (GPA, distance, major, gender) and in interactions?
"""
import pandas as pd, numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
import warnings; warnings.filterwarnings('ignore')
rng=np.random.RandomState(7)

d=pd.read_csv('../work/policy_coverage.csv')
COVS=['Personal Property','Additional Living Expense','Guest Medical','Liability']
FEATS=['greek_f','off_f','spr_f','gpa','dist','year_n','study_n','gender_n']
ymap={'Freshman':1,'Sophomore':2,'Junior':3,'Senior':4,'Grad student':5}
d['year_n']=d.year.map(ymap)
d['study_n']=pd.factorize(d.study)[0]; d['gender_n']=pd.factorize(d.gender)[0]

def gini(a,pred):
    o=np.lexsort((rng.rand(len(pred)),np.asarray(pred)))
    a=np.asarray(a)[o]; return 1-2*np.trapezoid(np.cumsum(a)/a.sum(),dx=1/len(a))

pred_ho={}
for c in COVS:
    tr=d[(d.coverage==c)&(d.is_train==1)]; ho=d[(d.coverage==c)&(d.is_train==0)]
    fm=HistGradientBoostingRegressor(loss='poisson',max_iter=300,learning_rate=.05,
        max_leaf_nodes=8,min_samples_leaf=150,l2_regularization=1.0,random_state=0)
    fm.fit(tr[FEATS],tr.claims)
    ts=tr[tr.claims>0]
    sm_=HistGradientBoostingRegressor(loss='gamma',max_iter=200,learning_rate=.05,
        max_leaf_nodes=6,min_samples_leaf=40,l2_regularization=1.0,random_state=0)
    sm_.fit(ts[FEATS],ts.severity)
    pred_ho[c]=pd.Series(fm.predict(ho[FEATS])*sm_.predict(ho[FEATS]),index=ho.student_id.values)

gbm=pd.DataFrame(pred_ho).sum(axis=1).rename('gbm_pp')
glm=pd.read_csv('../outputs/scored_policies_rated.csv').set_index('student_id')
h=glm[glm.is_train==0].join(gbm)

print("=== HOLDOUT DISCRIMINATION: GLM vs GBM challenger ===")
print(f"Proposed GLM               Gini = {gini(h.actual_total,h.total_pp):.3f}")
print(f"GBM challenger (all vars)  Gini = {gini(h.actual_total,h.gbm_pp):.3f}")
print(f"Underwriters' tiers        Gini = {gini(h.actual_total,h.risk_tier):.3f}")
print(f"\nCorrelation of the two model scores (holdout): {np.corrcoef(h.total_pp,h.gbm_pp)[0,1]:.3f}")
print(f"Spearman rank correlation:                     {h[['total_pp','gbm_pp']].corr(method='spearman').iloc[0,1]:.3f}")

# does blending the GBM in add anything?
for w in [0,.25,.5,.75,1]:
    bl=(1-w)*h.total_pp/h.total_pp.mean()+w*h.gbm_pp/h.gbm_pp.mean()
    print(f"  blend {1-w:.0%} GLM / {w:.0%} GBM -> Gini {gini(h.actual_total,bl):.3f}")

# what does the GBM lean on?
tr=d[(d.coverage=='Personal Property')&(d.is_train==1)]
m=HistGradientBoostingRegressor(loss='poisson',max_iter=300,learning_rate=.05,max_leaf_nodes=8,
    min_samples_leaf=150,l2_regularization=1.0,random_state=0).fit(tr[FEATS],tr.claims)
pi=permutation_importance(m,tr[FEATS],tr.claims,n_repeats=8,random_state=0,scoring='neg_mean_poisson_deviance')
imp=pd.Series(pi.importances_mean,index=FEATS).sort_values(ascending=False)
print("\n=== GBM permutation importance, Personal Property frequency ===")
print(imp.round(5).to_string())
print("\n-> variables we excluded (gpa, dist, study_n, gender_n) importance share: "
      f"{imp[['gpa','dist','study_n','gender_n']].clip(lower=0).sum()/imp.clip(lower=0).sum():.1%}")
h.to_csv('../outputs/challenger_comparison.csv')
