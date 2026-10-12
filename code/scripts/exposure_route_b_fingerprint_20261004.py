#!/usr/bin/env python3
"""Route B of PREREG_exposure_recompute_20261004: exposure membership from a JPEG/box fingerprint.

Shares no code with region_exposure_audit.py or Route A: no shuffle, no candidate pool, no
image_mapping.json. For every SLAKE image directory, fingerprint = (decoded JPEG size, the
detection.json boxes as floats in file order). Each retained receipt row is matched on its own
(image_size, organ_boxes); it is assigned only if the match is unique up to byte-identical JPEG
sha256. Split membership comes from the official SLAKE split files.

  python code/scripts/exposure_route_b_fingerprint_20261004.py --imgs <Slake1.0/imgs> --splits <dir> --out <json>
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image


def boxes_of(det_path):
    try:
        d = json.loads(det_path.read_text())
    except Exception:
        return None
    out = []
    for e in d if isinstance(d, list) else []:
        if isinstance(e, dict) and e:
            b = list(e.values())[0]
            if isinstance(b, (list, tuple)) and len(b) == 4:
                out.append(tuple(float(x) for x in b))
    return tuple(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs", required=True)
    ap.add_argument("--splits", required=True, help="dir with official train.json validate.json test.json")
    ap.add_argument("--receipt", default="results/cipe_scale/bbox_fg_medgemma4b_n400.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    imgs = Path(a.imgs)
    index = defaultdict(list)
    n_dirs = n_empty = 0
    for d in sorted(p for p in imgs.iterdir() if p.is_dir()):
        n_dirs += 1
        det, jpg = d / "detection.json", d / "source.jpg"
        if not det.exists() or not jpg.exists():
            continue
        bx = boxes_of(det)
        if not bx:
            n_empty += 1
            continue
        with Image.open(jpg) as im:
            size = tuple(im.size)
        index[(size, bx)].append((f"{d.name}/source.jpg", hashlib.sha256(jpg.read_bytes()).hexdigest()))
    sp_files = {"train": "train.json", "validation": "validate.json", "test": "test.json"}
    splits = {k: {r["img_name"] for r in json.load(open(Path(a.splits) / v))} for k, v in sp_files.items()}
    rec = json.load(open(a.receipt))
    per_item, problems = {}, []
    for row in rec["rows"]:
        key = (tuple(row["image_size"]), tuple(tuple(float(x) for x in b) for b in row["organ_boxes"]))
        cands = index.get(key, [])
        shas = {s for _, s in cands}
        if not cands:
            problems.append(f"{row['sample_id']}: no image matches the fingerprint"); continue
        if len(shas) > 1:
            problems.append(f"{row['sample_id']}: {len(cands)} non-identical images match: {[c for c, _ in cands]}"); continue
        mems = set()
        for ref, _ in cands:
            mems.add(",".join(sp for sp in ("test", "train", "validation") if ref in splits[sp]) or "none")
        if len(mems) > 1:
            problems.append(f"{row['sample_id']}: byte-identical duplicates differ in split membership {mems}"); continue
        per_item[row["sample_id"]] = ([c for c, _ in cands], mems.pop(), cands[0][1])
    counts = Counter(m for _, m, _ in per_item.values())
    n_train = sum(c for k, c in counts.items() if "train" in k.split(","))
    out = {"route": "B_fingerprint", "n_image_dirs": n_dirs, "n_dirs_with_empty_detection": n_empty,
           "n_items_assigned": len(per_item), "n_rows": len(rec["rows"]), "problems": problems,
           "membership_counts": dict(sorted(counts.items())), "n_on_train_images": n_train,
           "pct_on_train_images_1dp": f"{100 * n_train / max(1, len(per_item)):.1f}",
           "n_test_and_train": counts.get("test,train", 0), "n_test_only": counts.get("test", 0),
           "n_duplicate_fingerprint_items": sum(1 for v in per_item.values() if len(v[0]) > 1),
           "per_item": {k: v for k, v in sorted(per_item.items(), key=lambda kv: int(kv[0].split('_')[1]))}}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_item"}, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
