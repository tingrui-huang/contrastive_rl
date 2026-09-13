"""Recompute summaries, verify arithmetic, and render saved evidence; no model imports."""
import csv
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def write(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
with (OUT/'native_pairs.pkl').open('rb') as f:raw=pickle.load(f)
with (OUT/'contexts.pkl').open('rb') as f:contexts=pickle.load(f)
for row in raw:
    a=np.asarray(row['after_a']['obs'],float);b=np.asarray(row['after_b']['obs'],float)
    np.testing.assert_array_equal(a[2:],b[2:])
    den=np.linalg.norm(np.asarray(row['effective_a'])-row['effective_b'])
    assert den==row['denominator']
    if den:
        np.testing.assert_allclose(row['ratio'],np.linalg.norm(a-b)/den,rtol=1e-13)
        physical=np.linalg.norm(np.asarray(row['after_a']['physical'])-row['after_b']['physical'])/den
        np.testing.assert_allclose(physical,row['physical_ratio'],rtol=1e-13)
    assert np.array_equal(row['after_a']['bits'],row['after_b']['bits'])

def csv_rows(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

native=read('native_summary.json');models=read('model_summary.json');matrices=read('matrices.json')
native_flat=[]
for r in native:
    q={k:v for k,v in r.items() if k not in ['physical','visible']}
    for field in ['physical','visible']:q.update({field+'_'+k:v for k,v in r[field].items()})
    native_flat.append(q)
csv_rows('native_summary.csv',native_flat)
model_flat=[]
for r in models:
    q={k:v for k,v in r.items() if k!='ratios'};q.update(r['ratios']);model_flat.append(q)
csv_rows('model_summary.csv',model_flat)

data=np.load(OUT/'model_ratios.npz')
same=[]
for name in ['frobenius_s0','spectral_s0','frobenius_s1','spectral_s1']:
    for condition in ['diagonal_center_fixed_xprime','off_diagonal_fixed_xprime']:
        a=data[name+'__frobenius__'+condition+'__ratio'];b=data[name+'__spectral__'+condition+'__ratio']
        np.testing.assert_array_equal(a,b)
        same.append(name+'__'+condition)
jac=np.load(OUT/'jacobians.npz')
np.testing.assert_allclose(np.linalg.svd(jac['jacobian'],compute_uv=False)[:,0],jac['norm'])
costs=read('costs.json')['counts']
assert costs['pairs']==2*len(raw)==9600
assert costs['continuation']==6*len(read('continuation.json'))==114
assert costs['model']==51200+320+512+240==52272
assert costs['jacobian']==len(jac['jacobian'])==240
assert sum(costs[k] for k in ['collection','setup','pairs','validation','continuation'])==15399
provenance=read('provenance.json')
for path,digest in provenance['source_sha256'].items():
    with (OUT.parent.parent/path).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest,path
assert hashlib.sha256((OUT/'PROTOCOL.md').read_bytes()).hexdigest()==provenance['protocol_sha256']

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
for label in ['free','wall_near','hazard_boundary']:
    rows=[r for r in native if r['source']=='natural' and r['stratum']==label and r['event']=='all']
    axes[0].plot([r['epsilon'] for r in rows],[r['physical']['max'] for r in rows],'-o',label=label.replace('_',' '))
for label in ['upper_wall','corner','action_clipping']:
    rows=[r for r in native if r['source']=='targeted' and r['stratum']==label and r['event']=='all']
    axes[1].plot([r['epsilon'] for r in rows],[r['physical']['max'] for r in rows],'-o',label=label.replace('_',' '))
for ax,title in zip(axes,['Naturally collected strata: observed maxima','Targeted boundary probes: observed maxima']):
    ax.set_xscale('log');ax.invert_xaxis();ax.axhline(1,color='black',ls='--',lw=1,label='L = 1')
    ax.set(xlabel='Requested pair scale epsilon (action units)',ylabel='XY gain (maze units / effective action unit)',title=title)
    ax.legend(fontsize=8);ax.grid(alpha=.2)
axes[1].set_yscale('log')
fig.suptitle('Current PointMaze F4 native one-step sensitivity · HEAD 3d997ef')
fig.savefig(OUT/'native_sensitivity.png',dpi=180);plt.close(fig)

fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
for label,ident in [('Upper-wall threshold',176),('Corner threshold',184)]:
    rows=[r for r in raw if r['context']==ident and r['direction']==1]
    ax.loglog([r['denominator'] for r in rows],[r['physical_ratio'] for r in rows],'-o',label=label)
ax.axhline(1,color='black',ls='--',label='L = 1');ax.invert_xaxis()
ax.set(xlabel='Actual effective initial action difference (action units)',
       ylabel='One-step XY/F4 gain (maze units / action unit)',
       title='Fixed coordinate-direction witnesses across perturbation scales')
ax.legend();ax.grid(alpha=.2);fig.savefig(OUT/'boundary_witnesses.png',dpi=180);plt.close(fig)

payload=json.dumps(dict(native=native,model=models,matrices=matrices))
html='''<!doctype html><html><head><meta charset="utf-8"><title>F4 action sensitivity audit</title>
<style>body{font:15px system-ui;margin:32px;color:#20252b;max-width:1500px}h1{font-size:25px}h2{font-size:20px}select{padding:6px;margin-right:12px}table{border-collapse:collapse;font-size:13px;width:100%;margin:15px 0}th,td{padding:7px;text-align:right;border-bottom:1px solid #ddd}th:first-child,td:first-child{text-align:left}img{width:100%;max-width:1100px}.scroll{overflow-x:auto}p{max-width:1000px}</style></head><body>
<h1>Current PointMaze F4 action sensitivity</h1><p>HEAD 3d997ef. Free-motion local gain is 1. Collision thresholds violate a global L=1 samplewise assumption; death switching need not change immediate F4 gain. Counts and tails below are empirical, not population guarantees. Natural strata overlap. Targeted probes are never pooled with natural contexts.</p>
<img src="native_sensitivity.png" alt="Native gain by stratum and epsilon">
<h2>Native one-step measurements</h2><p>Gain uses actual post-noise, post-clipping action difference. Physical = float64 XY; visible = float32 next F4, differenced in float64. Historical successor coordinates match in all 4,800 pairs. Exceedance uses threshold + .002. Zero denominators are excluded.</p>
<select id="source"></select><select id="stratum"></select><select id="event"></select><select id="measure"><option>physical</option><option>visible</option></select><div id="native" class="scroll"></div>
<h2>Learned sample comparisons</h2><p>Fixed-xprime rows vary only execution action. “diagonal_center” locates the pair center at xprime; endpoints generally are off diagonal. “both_action_slots_vary” changes the observational diagonal and is not constrained by the partial-action proof.</p>
<select id="parameters"></select><select id="emitter"></select><select id="condition"></select><div id="model" class="scroll"></div>
<p>Each ETT pair shares state, goal, mixture randomness and—except observational-diagonal comparisons—xprime. Frobenius and spectral emitters give identical ratios for each shared parameter array: all block norms are already below 1.</p>
<p>Files: <a href="REPORT.md">Concise report</a> · <a href="PROTOCOL.md">Protocol</a> · <a href="native_summary.csv">Native CSV</a> · <a href="model_summary.csv">Model CSV</a> · <a href="native_witnesses.json">Witnesses</a> · <a href="verification.json">Verification</a></p>
<script>const data=PAYLOAD;
const el=id=>document.getElementById(id);
function options(id,values){const old=el(id).value;el(id).replaceChildren(...[...new Set(values)].map(v=>{let o=document.createElement('option');o.textContent=v;o.value=v;return o}));if(values.includes(old))el(id).value=old}
function fmt(x){return x==null?'—':typeof x==='number'?(Number.isInteger(x)?x:x.toPrecision(6)):x}
const fields=['count','median','p90','p95','p99','p99_9','max','above_1','above_1.05','above_1.25','above_2','above_5'];
function table(id,rows){el(id).innerHTML='<table><thead><tr>'+['epsilon','zero denominators',...fields].map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+[r.epsilon,r.zero_denominators,...fields.map(k=>r.stats[k])].map(x=>'<td>'+fmt(x)+'</td>').join('')+'</tr>').join('')+'</tbody></table>'}
function native(){options('stratum',data.native.filter(r=>r.source===el('source').value).map(r=>r.stratum));options('event',data.native.filter(r=>r.source===el('source').value&&r.stratum===el('stratum').value).map(r=>r.event));table('native',data.native.filter(r=>r.source===el('source').value&&r.stratum===el('stratum').value&&r.event===el('event').value).map(r=>({...r,stats:r[el('measure').value]})))}
function model(){options('emitter',data.model.filter(r=>r.parameters===el('parameters').value).map(r=>r.emitter));options('condition',data.model.filter(r=>r.parameters===el('parameters').value&&r.emitter===el('emitter').value).map(r=>r.condition));table('model',data.model.filter(r=>r.parameters===el('parameters').value&&r.emitter===el('emitter').value&&r.condition===el('condition').value).map(r=>({...r,stats:r.ratios})))}
options('source',data.native.map(r=>r.source));options('parameters',data.model.map(r=>r.parameters));['source','stratum','event','measure'].forEach(id=>el(id).onchange=native);['parameters','emitter','condition'].forEach(id=>el(id).onchange=model);native();model();</script></body></html>'''.replace('PAYLOAD',payload)
(OUT/'RESULTS.html').write_text(html,encoding='utf-8')
write('verification.json',dict(status='pass',arithmetic_rechecked_pairs=len(raw),f4_xy_one_step_equality=True,
      same_mask_all_pairs=True,frobenius_spectral_equal=same,jacobian_svd_recomputed=True,
      checkpoint_source_hashes_unchanged=True,protocol_unchanged=True,
      native_calls=15399,model_evaluations_including_jacobian_primals=52272,jacobian_evaluations=240,
      all_physical_excesses_over_1_002_change_collision_branch=all(r['collision_change'] for r in raw if r['physical_ratio'] is not None and r['physical_ratio']>1.002),
      maximum_float32_float64_difference=max(r['visible_physical_difference'] for r in raw),
      html='Self-contained data and DOM controls; no external dependencies. Static content and data verified; browser interaction not automated.',
      additional_simulator_or_model_calls=0))
print('Saved arithmetic verified, CSV/HTML/plots generated; no new simulator/model calls.')
