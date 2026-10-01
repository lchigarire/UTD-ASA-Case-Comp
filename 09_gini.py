"""Tie-aware Gini + Lorenz curves. Ties are grouped, not randomly broken."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'figure.dpi':130,'font.size':10,'axes.grid':True,'grid.alpha':.25,
                     'axes.spines.top':False,'axes.spines.right':False})
NAVY='#1F3864'; ORANGE='#E8912D'; GREY='#9AA0A6'; TEAL='#2E8B8B'

h=pd.read_csv('../outputs/challenger_comparison.csv')

def lorenz(actual,pred):
    """Ties grouped into one point. Returns cumulative exposure %, cumulative loss %."""
    t=pd.DataFrame({'a':np.asarray(actual),'p':np.asarray(pred)})
    g=t.groupby('p').agg(n=('a','size'),loss=('a','sum')).sort_index()
    x=np.r_[0,np.cumsum(g.n)/g.n.sum()]; y=np.r_[0,np.cumsum(g.loss)/g.loss.sum()]
    return x,y
def gini(actual,pred):
    x,y=lorenz(actual,pred); return float(1-2*np.trapezoid(y,x))

models={'Proposed GLM':(h.total_pp,NAVY,'-'),
        'GBM challenger':(h.gbm_pp,TEAL,'--'),
        "Underwriters' tiers":(h.risk_tier,ORANGE,'-.'),
        'Current flat rate':(np.ones(len(h)),GREY,':')}
print("=== TIE-AWARE GINI ON HOLDOUT (ties grouped, not randomly broken) ===")
fig,ax=plt.subplots(figsize=(5.6,5.2))
for k,(pr,c,ls) in models.items():
    x,y=lorenz(h.actual_total,pr); gg=gini(h.actual_total,pr)
    print(f"  {k:24s} {gg:+.3f}")
    ax.plot(x,y,ls,color=c,label=f"{k} (Gini {gg:.3f})",lw=1.8)
ax.plot([0,1],[0,1],c='k',lw=.8)
ax.set_xlabel('Cumulative share of policies (ranked least to most risky)')
ax.set_ylabel('Cumulative share of losses'); ax.set_title('Lorenz curves — holdout')
ax.legend(frameon=False,fontsize=8,loc='upper left'); plt.tight_layout()
plt.savefig('../outputs/fig6_lorenz.png'); plt.close()

# bootstrap the GLM-vs-GBM gap so the comparison has an error bar
rs=np.random.RandomState(3); diffs=[]
for _ in range(800):
    s=h.sample(len(h),replace=True,random_state=rs.randint(1e9))
    diffs.append(gini(s.actual_total,s.total_pp)-gini(s.actual_total,s.gbm_pp))
lo,hi=np.percentile(diffs,[2.5,97.5])
print(f"\nGLM minus GBM Gini gap: {np.mean(diffs):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
print("GBM beats GLM in", f"{np.mean(np.array(diffs)<0):.1%}", "of bootstrap samples")
