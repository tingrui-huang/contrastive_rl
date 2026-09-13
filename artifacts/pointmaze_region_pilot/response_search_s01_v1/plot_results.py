"""Plots saved results; no new samples."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
out=Path(__file__).resolve().parent
d=json.loads((out/'results.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
rows=[];labels=[]
for s in ['0','1']:
    for k in range(3):
        rows.append(d['terminal'][f's{s}_k{k}']['final_minus_start'])
        labels.append(f'Baseline {s}, '+('saved start' if k==0 else f'random {k}'))
values=np.array([r['mean'] for r in rows]);bounds=np.array([r['root_ci95'] for r in rows])
axes[0].errorbar(values,np.arange(6),xerr=np.stack([values-bounds[:,0],bounds[:,1]-values]),fmt='o',color='#296ea4',capsize=4)
axes[0].axvline(0,color='#8b939a',linewidth=1)
axes[0].set(yticks=np.arange(6),yticklabels=labels,xlabel='Normalized full-return change, root 95% CI',ylabel='Baseline / response initialization',title='24 updates do not improve every start')
axes[0].invert_yaxis()
for s,marker,color in [('0','o','#626970'),('1','s','#296ea4')]:
    rows=[r for n,r in d['local_checks'].items() if n.startswith('s'+s+'_')]
    x=np.array([r['critic']['mean'] for r in rows]);y=np.array([r['mc']['mean'] for r in rows])
    lo=np.array([r['mc']['conditional_mc_ci95'][0] for r in rows]);hi=np.array([r['mc']['conditional_mc_ci95'][1] for r in rows])
    axes[1].errorbar(x,y,yerr=np.stack([y-lo,hi-y]),fmt=marker,color=color,alpha=.8,capsize=2,label='Baseline '+s)
axes[1].axhline(0,color='#8b939a',linewidth=1);axes[1].axvline(0,color='#8b939a',linewidth=1)
axes[1].set(xlabel='Critic local surrogate change',ylabel='Matched MC local surrogate change',title='9/18 opposite mean signs; none resolved')
axes[1].legend(frameon=False)
for ax in axes:ax.spines[['top','right']].set_visible(False)
fig.suptitle('Frozen-diagonal ETT response search',fontsize=14)
fig.text(.5,-.04,'Source: response_search_s01_v1. Left: 64 fresh paths/root; lower is more pessimistic. Right: conditional MC 95% intervals.',ha='center',fontsize=9)
fig.savefig(out/'response_search.png',dpi=170,bbox_inches='tight')
fig.savefig(out/'response_search.pdf',bbox_inches='tight')
