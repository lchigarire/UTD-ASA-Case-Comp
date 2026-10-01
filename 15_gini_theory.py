import pandas as pd, numpy as np
h=pd.read_csv('../outputs/challenger_comparison.csv')
def lorenz(a,p):
    t=pd.DataFrame({'a':np.asarray(a,float),'p':np.asarray(p,float)})
    g=t.groupby('p').agg(n=('a','size'),loss=('a','sum')).sort_index()
    x=np.r_[0,np.cumsum(g.n)/g.n.sum()]; y=np.r_[0,np.cumsum(g.loss)/g.loss.sum()]
    return x,y
def gini(a,p):
    x,y=lorenz(a,p); return float(1-2*np.trapezoid(y,x))

A=h.actual_total.values
print(f"holdout policies={len(h)}  zero-loss share={(A==0).mean():.1%}")
G_model=gini(A,h.total_pp); G_gbm=gini(A,h.gbm_pp); G_oracle=gini(A,A)
print(f"\nGini  GLM={G_model:.4f}   GBM={G_gbm:.4f}   ORACLE (sort by actual)={G_oracle:.4f}")
print(f"Normalised Gini (model / oracle):  GLM={G_model/G_oracle:.3f}   GBM={G_gbm/G_oracle:.3f}")

# theoretical ceiling: order by TRUE expected loss, not realised loss
print(f"\nGini if we could sort by true expected loss is unknowable, but note:")
print(f"  even a perfect pricing model cannot exceed the oracle, and the oracle is inflated")
print(f"  by the {(A==0).mean():.0%} of policies with exactly zero realised loss.")

# double-lift check: is ordering by model/flat the same as ordering by model?
FLAT=pd.read_csv('../outputs/scored_policies_rated.csv').query('is_train==1').actual_total.mean()
r=h.total_pp/FLAT
print(f"\nDouble-lift ordering check: corr(rank(model/flat), rank(model)) = {np.corrcoef(r.rank(),h.total_pp.rank())[0,1]:.6f}")
