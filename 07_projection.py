"""
Step 7: 3-year projection. Two cohorts tracked separately:
  RENEWAL book - decreases granted in full at once; increases capped at +25%/yr until indicated.
  NEW business - written at the full indicated new-business rate; mix skews to segments where we
                 are now cheaper than the market (proxied by current flat rate).
  Underwriting action - Greek + off-campus + no sprinkler is non-renewed unless sprinklers installed.
"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'figure.dpi':130,'font.size':10,'axes.grid':True,'grid.alpha':.25,
                     'axes.spines.top':False,'axes.spines.right':False})
NAVY='#1F3864'; ORANGE='#E8912D'; GREY='#9AA0A6'

p=pd.read_csv('../outputs/scored_policies_rated.csv')
ULAE,VAR,FIXED,PROFIT=0.08,0.145,25.0,0.05
CUR=(p[p.is_train==1].actual_total.mean()*(1+ULAE)+FIXED)/(1-VAR-PROFIT)
BASE_LAPSE,ELAST,GROWTH=0.22,0.9,0.15
pp=p.total_pp.values; fair_r=p.rate_renewal.values; fair_n=p.rate_new.values

def lapse(charged,fair): return np.clip(BASE_LAPSE*(1+ELAST*(charged/fair-1)),0.03,0.85)
def result(w,charged):
    n=w.sum(); prem=(w*charged).sum()/n; loss=(w*pp).sum()/n
    lr=loss*(1+ULAE)/prem
    return dict(policies=n,avg_prem=prem,avg_loss=loss,loss_ratio=lr*100,
                comb_ratio=(lr+VAR+FIXED/prem)*100,uw_result=(prem*(1-VAR)-loss*(1+ULAE)-FIXED)*n)

def run(mode,years=5,cap=0.25,phase_down=False):
    wr=np.ones(len(p)); cr=np.full(len(p),CUR)          # renewal cohort
    wn=np.zeros(len(p))                                  # new-business cohort
    if mode=='proposed':
        wr[(p.greek_f==1)&(p.off_f==1)&(p.spr_f==0)]=0.0
    rows=[]
    for y in range(1,years+1):
        if mode=='proposed':
            cr=(np.clip(fair_r,cr*(1-cap),cr*(1+cap)) if phase_down else np.where(fair_r<cr,fair_r,np.minimum(fair_r,cr*(1+cap))))
        w=wr+wn; charged=np.where(w>0,(wr*cr+wn*fair_n)/np.maximum(w,1e-9),0)
        rows.append(dict(year=y,**result(w,np.where(w>0,charged,CUR))))
        # renewals lapse; new business rolls into the renewal cohort at its indicated rate
        keep_r=1-lapse(cr,fair_r); keep_n=1-lapse(fair_n,fair_n)
        wr_new=wr*keep_r+wn*keep_n
        cr=np.where(wr+wn>0,(wr*keep_r*cr+wn*keep_n*fair_n)/np.maximum(wr_new,1e-9),cr)
        wr=wr_new
        if mode=='proposed':
            att=np.clip(CUR/fair_n,0.2,3.0); att=np.where(wr+wn>0,att,0); att[(p.greek_f==1)&(p.off_f==1)&(p.spr_f==0)]=0
            wn=GROWTH*10000*att/att.sum()
        else:
            wn=np.zeros(len(p))
    return pd.DataFrame(rows)

print(f"Current revenue-neutral flat rate: ${CUR:,.0f}\n")
res={'Status quo':run('status_quo'),'Plan A: decreases now':run('proposed'),'Plan B: phase both ways':run('proposed',phase_down=True)}
for k,r in res.items():
    print(f"=== {k.upper()} ===")
    print(r.round({'policies':0,'avg_prem':0,'avg_loss':0,'loss_ratio':1,'comb_ratio':1,'uw_result':0}).to_string(index=False)); print()

fig,ax=plt.subplots(1,3,figsize=(12.5,3.7))
yr=[1,2,3,4,5]; sty={'Status quo':(GREY,'o-'),'Plan A: decreases now':(NAVY,'s-'),'Plan B: phase both ways':(ORANGE,'^-')}
for k,r in res.items():
    c,m=sty[k]
    ax[0].plot(yr,r.comb_ratio,m,color=c,label=k); ax[1].plot(yr,r.policies,m,color=c,label=k)
    ax[2].plot(yr,r.uw_result/1e6,m,color=c,label=k)
ax[0].axhline(100,ls='--',c='k',lw=.8); ax[0].set_ylabel('Combined ratio (%)'); ax[0].set_title('Profitability')
ax[1].set_ylabel('Policies in force'); ax[1].set_title('Book size')
ax[2].axhline(0,ls='--',c='k',lw=.8); ax[2].set_ylabel('Underwriting result ($m)'); ax[2].set_title('Underwriting result')
for a in ax: a.set_xticks(yr); a.set_xlabel('Year')
ax[0].legend(frameon=False,fontsize=8)
plt.suptitle('Five-year outlook under three strategies',y=1.04,fontsize=11.5)
plt.tight_layout(); plt.savefig('../outputs/fig5_projection.png',bbox_inches='tight'); plt.close()

a=p[(p.greek_f==1)&(p.off_f==1)&(p.spr_f==0)]; b=p[(p.greek_f==1)&(p.off_f==1)&(p.spr_f==1)]
print("=== UNDERWRITING ACTION ===")
print(f"Greek + off-campus + NO sprinkler: n={len(a)} ({len(a)/len(p):.1%} of book), loss cost ${a.total_pp.mean():,.0f}, indicated ${a.rate_renewal.mean():,.0f}")
print(f"Greek + off-campus + sprinklered : n={len(b)} ({len(b)/len(p):.1%}), loss cost ${b.total_pp.mean():,.0f}, indicated ${b.rate_renewal.mean():,.0f}")
print(f"Sprinkler requirement cuts expected loss {1-b.total_pp.mean()/a.total_pp.mean():.0%}; that {len(a)/len(p):.1%} of the book holds {a.total_pp.sum()/p.total_pp.sum():.1%} of expected loss")
