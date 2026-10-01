"""Step 4: Final frequency/severity GLMs, credibility, rating relativities."""
import pandas as pd, numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf
import json, warnings; warnings.filterwarnings('ignore')

d=pd.read_csv('../work/policy_coverage.csv')
d['year']=pd.Categorical(d.year,['Freshman','Sophomore','Junior','Senior','Grad student'])
tr=d[d.is_train==1].copy(); ho=d[d.is_train==0].copy()
COVS=['Personal Property','Additional Living Expense','Guest Medical','Liability']
P=sm.families.Poisson(); G=sm.families.Gamma(sm.families.links.Log())

FREQ_F="claims ~ greek_f + off_f + C(year)"
SEV_F ="loss   ~ greek_f + off_f + spr_f + C(year)"

models={}; rel_rows=[]
for c in COVS:
    t=tr[tr.coverage==c]
    fm=smf.glm(FREQ_F,data=t,family=P).fit()
    sm_=smf.glm(SEV_F,data=t[t.claims>0],family=G).fit()
    models[c]={'freq':fm,'sev':sm_,'n_claims':int(t.claims.sum())}

def rels(m,kind):
    """exp(coef) keyed by rating level; base levels implicitly 1.00"""
    out={'Greek':np.exp(m.params.get('greek_f',0.0)),
         'Off campus':np.exp(m.params.get('off_f',0.0))}
    if kind=='sev': out['Sprinklered']=np.exp(m.params.get('spr_f',0.0))
    for lev in ['Sophomore','Junior','Senior','Grad student']:
        out[lev]=np.exp(m.params.get(f'C(year)[T.{lev}]',0.0))
    return out

# ---- credibility: square-root rule, full credibility at 1,082 claims (+/-5% @ 90%) --
FULL=1082
cred={}
for c in COVS:
    n=models[c]['n_claims']; cred[c]=min(1.0,np.sqrt(n/FULL))

raw={c:{'freq':rels(models[c]['freq'],'freq'),'sev':rels(models[c]['sev'],'sev')} for c in COVS}
# complement of credibility = claim-weighted avg relativity across all four coverages
def complement(kind,key):
    num=sum(models[c]['n_claims']*np.log(raw[c][kind].get(key,1.0)) for c in COVS)
    den=sum(models[c]['n_claims'] for c in COVS)
    return np.exp(num/den)

final={}
for c in COVS:
    Z=cred[c]; final[c]={}
    for kind in ['freq','sev']:
        final[c][kind]={}
        for k,v in raw[c][kind].items():
            comp=complement(kind,k)
            final[c][kind][k]=np.exp(Z*np.log(v)+(1-Z)*np.log(comp))

# ---- base values at base level (Freshman, Non-greek, On campus, Not sprinklered) ----
for c in COVS:
    fm,sm_=models[c]['freq'],models[c]['sev']
    bf=np.exp(fm.params['Intercept']); bs=np.exp(sm_.params['Intercept'])
    final[c]['base_freq']=bf; final[c]['base_sev']=bs; final[c]['base_pp']=bf*bs
    final[c]['Z']=cred[c]; final[c]['n_claims']=models[c]['n_claims']

print("=== CREDIBILITY & BASE VALUES (base cell: Freshman, Non-Greek, On-campus, No sprinkler) ===")
for c in COVS:
    f=final[c]
    print(f"{c:28s} claims={f['n_claims']:4d}  Z={f['Z']:.2f}  base freq={f['base_freq']:.4f}  base sev=${f['base_sev']:8,.0f}  base PP=${f['base_pp']:7,.2f}")

print("\n=== COMBINED RATING RELATIVITIES (frequency x severity) ===")
keys=['Greek','Off campus','Sprinklered','Sophomore','Junior','Senior','Grad student']
tab=pd.DataFrame({c:{k:final[c]['freq'].get(k,1.0)*final[c]['sev'].get(k,1.0) for k in keys} for c in COVS})
print(tab.round(3).to_string())
tab.to_csv('../outputs/relativities_combined.csv')
pd.DataFrame({c:final[c]['freq'] for c in COVS}).to_csv('../outputs/relativities_frequency.csv')
pd.DataFrame({c:final[c]['sev'] for c in COVS}).to_csv('../outputs/relativities_severity.csv')

# ---- scoring function --------------------------------------------------------
def score(df):
    out=pd.DataFrame(index=df.index)
    for c in COVS:
        f=final[c]
        fr=f['base_freq']*np.where(df.greek_f==1,f['freq']['Greek'],1)*np.where(df.off_f==1,f['freq']['Off campus'],1)
        sv=f['base_sev']*np.where(df.greek_f==1,f['sev']['Greek'],1)*np.where(df.off_f==1,f['sev']['Off campus'],1)*np.where(df.spr_f==1,f['sev']['Sprinklered'],1)
        for lev in ['Sophomore','Junior','Senior','Grad student']:
            fr=fr*np.where(df.year==lev,f['freq'][lev],1); sv=sv*np.where(df.year==lev,f['sev'][lev],1)
        out[c]=fr*sv
    out['total_pp']=out[COVS].sum(axis=1)
    return out

pol=d.drop_duplicates('student_id')[['student_id','year','greek_f','off_f','spr_f','risk_tier','is_train','gpa']].copy()
pol['year']=pd.Categorical(pol.year,['Freshman','Sophomore','Junior','Senior','Grad student'])
s=score(pol); pol=pd.concat([pol.reset_index(drop=True),s.reset_index(drop=True)],axis=1)
actual=d.pivot_table(index='student_id',columns='coverage',values='loss',aggfunc='sum')
actual['actual_total']=actual.sum(axis=1)
pol=pol.merge(actual[['actual_total']],on='student_id')
# ---- off-balance correction: product of two GLMs is not automatically balanced ----
bal={}
trm=pol[pol.is_train==1]
act_tr=d[d.is_train==1].pivot_table(index='student_id',columns='coverage',values='loss',aggfunc='sum')
for c in COVS:
    bal[c]=act_tr[c].mean()/trm[c].mean()
    pol[c]=pol[c]*bal[c]; final[c]['balance_factor']=bal[c]; final[c]['base_pp']*=bal[c]
pol['total_pp']=pol[COVS].sum(axis=1)
print("\nOff-balance correction factors:", {k:round(v,4) for k,v in bal.items()})
pol.to_csv('../work/scored_policies.csv',index=False)

print(f"\nModelled pure premium: mean=${pol.total_pp.mean():,.0f}  min=${pol.total_pp.min():,.0f}  max=${pol.total_pp.max():,.0f}  spread={pol.total_pp.max()/pol.total_pp.min():.1f}x")
print(f"Actual book pure premium: ${pol.actual_total.mean():,.0f}")
json.dump({c:{k:(v if not isinstance(v,dict) else v) for k,v in final[c].items()} for c in COVS}, open('../outputs/model_params.json','w'), indent=2, default=float)
