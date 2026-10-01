"""
ABG Dormitory Insurance - Step 1: Data preparation
Grain of raw file: one row per policyholder x coverage x claim.
Target grain:      one row per policyholder x coverage (claim count + total loss).
"""
import pandas as pd, numpy as np

RAW = '/mnt/user-data/uploads/06_-_CAS_Predictive_Modeling_Case_Competition-_Dataset.csv'
COVS = ['Personal Property','Additional Living Expense','Guest Medical','Liability']

def load():
    df = pd.read_csv(RAW)
    df.columns = [c.strip() for c in df.columns]
    return df

def audit(df):
    a = {}
    a['raw_rows'] = len(df)
    a['policyholders'] = df.student_id.nunique()
    g = df.groupby(['student_id','coverage']).size()
    a['multi_claim_rows'] = int((g>1).sum())
    a['extra_rows_from_multiclaim'] = int((g-1).clip(lower=0).sum())
    a['claims'] = int((df.claim_id>0).sum())
    a['claim_id_unique'] = df.loc[df.claim_id>0,'claim_id'].nunique()
    a['nulls'] = int(df.isna().sum().sum())
    # limit breaches
    lim = {'Personal Property':10000,'Liability':500000,'Guest Medical':150000}
    a['limit_breaches'] = {k:int(((df.coverage==k)&(df.amount>v)).sum()) for k,v in lim.items()}
    return a

def build(df):
    agg = (df.groupby(['student_id','coverage'], as_index=False)
             .agg(claims=('claim_id', lambda s:(s>0).sum()),
                  loss=('amount','sum')))
    pol = (df.drop_duplicates('student_id')
             [['student_id','class','study','gpa','greek','off_campus',
               'distance_to_campus','gender','sprinklered','risk_tier','training_holdout']]
             .rename(columns={'class':'year'}))
    d = agg.merge(pol, on='student_id', how='left')

    # --- feature engineering -------------------------------------------------
    d['greek_f']   = (d.greek == 'Greek').astype(int)
    d['off_f']     = (d.off_campus == 'Off campus').astype(int)
    d['spr_f']     = d.sprinklered.astype(int)
    # distance is structurally 0 for on-campus; model it only within off-campus
    d['dist']      = d.distance_to_campus
    d['log_dist']  = np.log1p(d.dist)
    d['year']      = pd.Categorical(d.year, ['Freshman','Sophomore','Junior','Senior','Grad student'])
    d['upper']     = d.year.isin(['Senior','Grad student']).astype(int)
    d['gpa_c']     = d.gpa - 2.33
    d['is_train']  = (d.training_holdout == 'training').astype(int)
    d['has_claim'] = (d.claims > 0).astype(int)
    d['severity']  = np.where(d.claims>0, d.loss/d.claims, np.nan)
    return d

if __name__ == '__main__':
    df = load()
    a = audit(df)
    for k,v in a.items(): print(f"{k:28s} {v}")
    d = build(df)
    d.to_csv('../work/policy_coverage.csv', index=False)
    print("\nbuilt:", d.shape, "| expected", 10000*4)
    print(d.groupby('coverage').agg(freq=('claims','mean'), sev=('severity','mean'),
                                    pp=('loss','mean')).round(2))
    print("\ntier == year check:", pd.crosstab(d.year, d.risk_tier).to_dict())
