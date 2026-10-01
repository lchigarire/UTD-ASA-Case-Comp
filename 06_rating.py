"""Step 6: Rating algorithm, tiers, expense loading, final rate pages."""
import pandas as pd, numpy as np, json
p=pd.read_csv('../work/scored_policies.csv')
COVS=['Personal Property','Additional Living Expense','Guest Medical','Liability']
YR=['Freshman','Sophomore','Junior','Senior','Grad student']
FLAT=p[p.is_train==1].actual_total.mean()

# ---- expense & profit assumptions (stated, editable) -------------------------
A=dict(ulae=0.08, fixed_expense=25.0, commission=0.12, prem_tax=0.025, profit=0.05,
       acq_new=45.0, acq_renewal=5.0)
VAR=A['commission']+A['prem_tax']+A['profit']
def rate(pp, new=False):
    fixed=A['fixed_expense']+(A['acq_new'] if new else A['acq_renewal'])
    return (pp*(1+A['ulae'])+fixed)/(1-VAR)

p['rate_renewal']=rate(p.total_pp); p['rate_new']=rate(p.total_pp,new=True)

# ---- all 40 rate cells -------------------------------------------------------
cells=(p.groupby(['year','greek_f','off_f','spr_f'],observed=True)
        .agg(n=('total_pp','size'),pp=('total_pp','mean'),rate=('rate_renewal','mean')).reset_index())
cells['year']=pd.Categorical(cells.year,YR); cells=cells.sort_values('pp')
cells['ratio_to_flat']=(cells.pp/FLAT).round(2)
cells.to_csv('../outputs/rate_cells_all.csv',index=False)
print("=== FULL RATE TABLE: all 40 cells, sorted by loss cost ===")
pr=cells.copy(); pr['Greek']=np.where(pr.greek_f==1,'Greek','Non-Greek')
pr['Loc']=np.where(pr.off_f==1,'Off','On'); pr['Spr']=np.where(pr.spr_f==1,'Y','N')
print(pr[['year','Greek','Loc','Spr','n','pp','rate','ratio_to_flat']].to_string(index=False,
      formatters={'pp':'${:,.0f}'.format,'rate':'${:,.0f}'.format}))

# ---- tier definition by loss cost thresholds --------------------------------
T1,T2=600,1400
def tier(pp): return np.where(pp<T1,'Preferred',np.where(pp<T2,'Standard','Non-standard'))
p['tier']=tier(p.total_pp)
ho=p[p.is_train==0]
print(f"\n=== PROPOSED TIERS (thresholds ${T1} / ${T2} modelled loss cost) ===")
g=p.groupby('tier').agg(policies=('total_pp','size'),share=('total_pp',lambda x:len(x)/len(p)),
                        model_pp=('total_pp','mean'),avg_rate=('rate_renewal','mean'))
g2=ho.groupby('tier').actual_total.mean().rename('holdout_actual')
print(pd.concat([g,g2],axis=1).round(2).to_string())
print("\nOld tiers (holdout actual):", p[p.is_train==0].groupby('risk_tier').actual_total.mean().round(0).to_dict())

# ---- dislocation -------------------------------------------------------------
cur=rate(FLAT)
p['change']=p.rate_renewal/cur-1
print(f"\n=== DISLOCATION vs today's flat rate (${cur:,.0f}) ===")
b=pd.cut(p.change,[-1,-.5,-.25,0,.25,.5,1,10],
         labels=['<-50%','-50..-25%','-25..0%','0..+25%','+25..+50%','+50..+100%','>+100%'])
print(b.value_counts().sort_index().to_frame('policies').assign(pct=lambda x:(x.policies/len(p)*100).round(1)).to_string())
print(f"\nCapped transition (+/-25% yr1, +/-25% yr2): pct at full indicated rate after yr1 = {(p.change.abs()<=.25).mean():.1%}")

print(f"\n=== PORTFOLIO ECONOMICS ===")
print(f"Revenue-neutral flat rate today:      ${cur:,.0f}")
print(f"Proposed renewal rate: min ${p.rate_renewal.min():,.0f}  median ${p.rate_renewal.median():,.0f}  max ${p.rate_renewal.max():,.0f}")
print(f"Proposed new-business rate (same risk, +$45 acquisition): median ${p.rate_new.median():,.0f}")
print(f"Expense assumptions: ULAE {A['ulae']:.0%} of loss, fixed ${A['fixed_expense']:.0f}, "
      f"commission {A['commission']:.0%}, tax {A['prem_tax']:.1%}, profit {A['profit']:.0%} -> PLR {1-VAR:.1%}")
json.dump(A,open('../outputs/expense_assumptions.json','w'),indent=2)
p.to_csv('../outputs/scored_policies_rated.csv',index=False)

# ---- sample rate pages -------------------------------------------------------
print("\n=== SAMPLE QUOTES ===")
ex=[('Freshman',0,0,1,'Freshman, non-Greek, on-campus, sprinklered'),
    ('Sophomore',0,1,1,'Sophomore, non-Greek, off-campus, sprinklered'),
    ('Junior',1,0,1,'Junior, Greek, on-campus, sprinklered'),
    ('Senior',1,1,0,'Senior, Greek, off-campus, NOT sprinklered'),
    ('Grad student',1,1,0,'Grad, Greek, off-campus, NOT sprinklered')]
for y,gk,of,sp,lab in ex:
    m=p[(p.year==y)&(p.greek_f==gk)&(p.off_f==of)&(p.spr_f==sp)]
    print(f"{lab:48s} loss cost ${m.total_pp.mean():6,.0f}  ->  renewal rate ${m.rate_renewal.mean():6,.0f}  ({m.rate_renewal.mean()/cur-1:+.0%} vs today)")
