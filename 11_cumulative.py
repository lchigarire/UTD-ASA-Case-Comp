import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'figure.dpi':130,'font.size':10,'axes.grid':True,'grid.alpha':.25,
                     'axes.spines.top':False,'axes.spines.right':False})
NAVY='#1F3864'; ORANGE='#E8912D'; GREY='#9AA0A6'
exec(open('10_proforma.py').read().split('for mode,lab in')[0])
sq=proforma('status_quo',5); pr=proforma('proposed',5)
c=pd.DataFrame({'Status quo':sq.loc['uw_result'].cumsum(),'Proposed plan':pr.loc['uw_result'].cumsum()})
print("=== CUMULATIVE UNDERWRITING RESULT ($) ===")
print(c.round(0).to_string())
print("\n=== POLICIES IN FORCE ===")
print(pd.DataFrame({'Status quo':sq.loc['policies'],'Proposed plan':pr.loc['policies']}).round(0).to_string())
print(f"\n5-year swing in favour of the proposed plan: ${c['Proposed plan'].iloc[-1]-c['Status quo'].iloc[-1]:,.0f}")
print(f"Crossover year (cumulative): {int((c['Proposed plan']>c['Status quo']).idxmax())}")
fig,ax=plt.subplots(figsize=(6.6,4))
ax.plot(c.index,c['Status quo']/1e6,'o-',c=GREY,label='Status quo')
ax.plot(c.index,c['Proposed plan']/1e6,'s-',c=NAVY,label='Proposed plan')
ax.axhline(0,c='k',lw=.8); ax.set_xticks(c.index)
ax.set_xlabel('Year'); ax.set_ylabel('Cumulative underwriting result ($m)')
ax.set_title('The transition pays for itself'); ax.legend(frameon=False)
plt.tight_layout(); plt.savefig('../outputs/fig8_cumulative.png'); plt.close()
c.to_csv('../outputs/cumulative_uw.csv')
