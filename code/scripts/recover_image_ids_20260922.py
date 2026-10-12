#!/usr/bin/env python3
"""Read historical SLAKE metadata; recover image identities only after exact checks."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
from PIL import Image

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fingerprint(boxes,size):return json.dumps([boxes,list(size)],separators=(',',':'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--snapshot',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 imgs=a.root/'data/slake/imgs'; source_hashes={}; records=[]
 train=a.snapshot/'train.json'
 if train.exists(): records=json.loads(train.read_text());source_hashes[str(train)]=sha(train);branch='snapshot_train'
 else:
  branch='sorted_image_questions'
  for qf in sorted(imgs.glob('*/question.json')):
   source_hashes[str(qf.relative_to(a.root))]=sha(qf)
   records.extend(json.loads(qf.read_text()))
 closed=[r for r in records if r.get('q_lang')=='en' and str(r.get('answer')).strip().lower() in {'yes','no'}]
 random.Random(42).shuffle(closed); candidates=[]; signatures=collections.defaultdict(set)
 # Inventory only image files referenced by QA, excluding segmentation/overlay PNGs.
 for ref in sorted({r['img_name'] for r in records}):
  ip=imgs/ref;dp=ip.parent/'detection.json'
  if not ip.exists() or not dp.exists():continue
  source_hashes[str(dp.relative_to(a.root))]=sha(dp)
  raw=json.loads(dp.read_text());boxes=[list(x.values())[0] for x in raw]
  boxes=[b for b in boxes if isinstance(b,(list,tuple)) and len(b)==4]
  with Image.open(ip) as im:size=list(im.size)
  signatures[fingerprint(boxes,size)].add(str(ip.relative_to(imgs)))
 for r in closed:
  ip=imgs/r['img_name'];dp=ip.parent/'detection.json'
  if not ip.exists() or not dp.exists():continue
  raw=json.loads(dp.read_text());pairs=[(next(iter(x)),list(x.values())[0]) for x in raw];pairs=[(k,v) for k,v in pairs if isinstance(v,(tuple,list)) and len(v)==4]
  if not pairs:continue
  with Image.open(ip) as im:size=list(im.size)
  boxes=[v for _,v in pairs]
  candidates.append({'sample_id':f'slake_{len(candidates)}','image_ref':r['img_name'],'image_sha256':sha(ip),'image_size':size,'organ_boxes':boxes,'bbox_labels':[k for k,_ in pairs],'question':r['question'],'answer':str(r['answer']).strip().lower(),'question_type':r.get('q_type',r.get('question_type')),'modality':r.get('modality'),'source_qa_id':r.get('qid',r.get('id')),'fingerprint_candidates':sorted(signatures[fingerprint(boxes,size)])})
  if len(candidates)==400:break
 assert len(candidates)==400,len(candidates)
 mapping={r['sample_id']:r for r in candidates};checks=[]
 for fp in sorted((a.root/'results/cipe_scale').glob('bbox_fg_*_n400.json')):
  d=json.loads(fp.read_text());assert d['seed']==42
  old=d['rows']+d['skipped'];bad=[]
  for row in old:
   got=mapping[row['sample_id']]
   if row['organ_boxes']!=got['organ_boxes'] or row['image_size']!=got['image_size']:bad.append(row['sample_id'])
   if len(got['fingerprint_candidates'])!=1 and row in d['rows']:bad.append('ambiguous_included:'+row['sample_id'])
  checks.append({'file':str(fp.relative_to(a.root)),'sha256':sha(fp),'n':len(old),'mismatches':bad})
 assert len(checks)==6 and all(c['n']==400 and not c['mismatches'] for c in checks), [(c['file'],len(c['mismatches']),c['mismatches'][:4]) for c in checks]
 result={'source_branch':branch,'seed':42,'n_candidates':400,'unique_candidate_images':len({r['image_ref'] for r in candidates}),'validation':'2400/2400 ordered box/dimension matches; every INCLUDED analysis item has a unique image fingerprint; excluded ambiguous identities remain unresolved' , 'ambiguous_candidate_ids':[r['sample_id'] for r in candidates if len(r['fingerprint_candidates'])!=1],'qa_validation_limit':'Historical receipts lack question text/answer; image identity is validated by unique annotation fingerprint, not exact historical QA text.','source_hashes':source_hashes,'receipt_checks':checks,'items':candidates}
 a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['items','source_hashes','receipt_checks']},indent=2))
if __name__=='__main__':main()
