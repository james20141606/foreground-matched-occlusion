#!/usr/bin/env python3
"""Post-hoc image-cluster sensitivity on normalized-TV receipts; no inference rerun."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

SEED=20260922
B=20000

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tv(p,q):
 p=np.asarray(p,float);q=np.asarray(q,float)
 assert p.sum()>0 and q.sum()>0
 return .5*np.abs(p/p.sum()-q/q.sum()).sum()
def bh(p):
 p=np.asarray(p);o=np.argsort(p);v=np.minimum.accumulate((p[o]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];q=np.empty_like(v);q[o]=np.minimum(1,v);return q

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);ap.add_argument('--mapping',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 mapping=json.loads(a.mapping.read_text());lookup={r['sample_id']:r for r in mapping['items']}
 models=[];ids=None;sources={};W=[];D=[];T=[]
 checks={r['file']:r['sha256'] for r in mapping['receipt_checks']}
 for fp in sorted((a.root/'results/cipe_scale').glob('bbox_fg_*_n400.json')):
  assert checks[str(fp.relative_to(a.root))]==sha(fp), 'Mapping/receipt version mismatch'
  rec=json.loads(fp.read_text());rows=sorted(rec['rows'],key=lambda x:int(x['sample_id'].split('_')[1]));current=[r['sample_id'] for r in rows]
  if ids is None:ids=current
  assert current==ids
  for r in rows:
   m=lookup[r['sample_id']];assert m['organ_boxes']==r['organ_boxes'] and m['image_size']==r['image_size'];assert len(m['fingerprint_candidates'])==1
  models.append(fp.stem.removeprefix('bbox_fg_').removesuffix('_n400'));sources[str(fp.relative_to(a.root))]=sha(fp)
  w=[];d=[];ties=[]
  for r in rows:
   target=tv(r['p_original'],r['p_organ']);ctl=np.array([tv(r['p_original'],p) for p in r['p_control_draws']]);assert len(ctl)==5
   ties.append(float(np.mean(target==ctl)));w.append(float(np.mean(target>ctl)));d.append(float(target-ctl.mean()))
  W.append(w);D.append(d);T.append(ties)
 W=np.asarray(W).T;D=np.asarray(D).T;T=np.asarray(T).T
 images=sorted({lookup[i]['image_sha256'] for i in ids});idx=np.array([images.index(lookup[i]['image_sha256']) for i in ids]);sizes=np.bincount(idx,minlength=len(images))
 sumw=np.stack([W[idx==g].sum(0) for g in range(len(images))]);sumd=np.stack([D[idx==g].sum(0) for g in range(len(images))]);imagew=sumw/sizes[:,None]
 rng=np.random.default_rng(SEED);draws=rng.integers(0,len(images),(B,len(images)));den=sizes[draws].sum(1)
 wb=sumw[draws].sum(1)/den[:,None];db=sumd[draws].sum(1)/den[:,None];iwb=imagew[draws].mean(1)
 # Cluster-level sign flip around the specified null W=.5, preserving QA weights.
 signs=np.random.default_rng(SEED+1).choice([-1.,1.],size=(B,len(images)))
 perm=np.einsum('bg,gm->bm',signs,sumw-.5*sizes[:,None],optimize=False)/len(ids)
 assert np.all(np.isfinite(perm)) and np.max(np.abs(perm))<=.5+1e-12
 obs=W.mean(0)-.5;p=(np.sum(np.abs(perm)>=np.abs(obs)[None,:]-1e-12,axis=0)+1)/(B+1)
 q=bh(p);results=[]
 # Holm requires no dependence assumption across the six model tests.
 order=np.argsort(p);holm=np.empty_like(p);holm[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
 half=W+.5*T;sumhalf=np.stack([half[idx==g].sum(0) for g in range(len(images))]);half_boot=sumhalf[draws].sum(1)/den[:,None]
 half_perm=np.einsum('bg,gm->bm',signs,sumhalf-.5*sizes[:,None],optimize=False)/len(ids);half_p=(np.sum(np.abs(half_perm)>=np.abs(half.mean(0)-.5)[None,:]-1e-12,axis=0)+1)/(B+1)
 assert np.all(np.isfinite(half_perm)) and np.max(np.abs(half_perm))<=.5+1e-12
 half_q=bh(half_p)
 order_half=np.argsort(half_p);holm_half=np.empty_like(half_p);holm_half[order_half]=np.minimum(1,np.maximum.accumulate(half_p[order_half]*(len(half_p)-np.arange(len(half_p)))))
 old={r['model']:r for r in json.loads((a.root/'outputs/region_tv_corrected.json').read_text())}
 for k,m in enumerate(models):
  assert abs(W[:,k].mean()-old[m]['W_mean'])<1e-12
  assert abs(D[:,k].mean()-old[m]['D_mean'])<1e-12
  results.append({'model':m,'n_qa':len(ids),'n_images':len(images),'W_qa_weighted':float(W[:,k].mean()),'W_image_cluster_ci95':np.percentile(wb[:,k],[2.5,97.5]).tolist(),'D_qa_weighted':float(D[:,k].mean()),'D_image_cluster_ci95':np.percentile(db[:,k],[2.5,97.5]).tolist(),'W_image_equal':float(imagew[:,k].mean()),'W_image_equal_ci95':np.percentile(iwb[:,k],[2.5,97.5]).tolist(),'W_signflip_p':float(p[k]),'W_signflip_BH_q':float(q[k]),'W_signflip_Holm_p':float(holm[k]),'exact_tie_fraction':float(T[:,k].mean()),'W_half_ties':float(half[:,k].mean()),'W_half_ties_cluster_ci95':np.percentile(half_boot[:,k],[2.5,97.5]).tolist(),'W_half_ties_signflip_p':float(half_p[k]),'W_half_ties_signflip_BH_q':float(half_q[k]),'W_half_ties_signflip_Holm_p':float(holm_half[k]),'legacy_item_bootstrap_ci95':old[m]['W_ci'],'legacy_item_bootstrap_q':old[m]['W_q']})
 result={'analysis_status':'post-hoc sensitivity; not a new confirmatory registration','seed':SEED,'resamples':B,'primary_estimand':'QA-weighted mean win fraction; images resampled with all QA rows and five control draws','test_assumption':'image-content-hash clusters independent; sign-flip null requires symmetry of image-level score sums about W=.5. Strict-win ties make chance .5 conservative under swap exchangeability; half-tie sensitivity explicitly restores per-comparison symmetry. These conditional post-hoc tests do not establish a general clinical null.','mapping_sha256':sha(a.mapping),'generator_sha256':sha(Path(__file__)),'receipt_sha256':sources,'n_qa':len(ids),'n_images':len(images),'qa_per_image':{'min':int(sizes.min()),'max':int(sizes.max()),'mean':float(sizes.mean()),'distribution':{str(x):int(sum(sizes==x)) for x in sorted(set(sizes))}},'results':results}
 a.out.write_text(json.dumps(result,indent=2)+'\n')
 for r in results:print(r['model'], 'nimages=',r['n_images'],'W=',round(r['W_qa_weighted'],4),'CI=',[round(x,4) for x in r['W_image_cluster_ci95']],'q=',round(r['W_signflip_BH_q'],5),'image_equal=',round(r['W_image_equal'],4))
if __name__=='__main__':main()
