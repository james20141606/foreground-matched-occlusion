#!/usr/bin/env python3
"""Route A of PREREG_exposure_recompute_20261004: exposure membership from the 09-22 image mapping.

Independent of region_exposure_audit.py: no shuffle, no candidate pool, no image tree. Each retained
sample_id of the fg receipt is joined to image_ref through image_mapping.json (fingerprint recovery,
2026-09-22) and image_ref is looked up by img_name in the official SLAKE split files.

Importable (compute()) so the paper checker can use the same route without reading any generator
output. Standard library only.
"""
from __future__ import annotations
import hashlib, json, os, sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RECEIPT = "results/cipe_scale/bbox_fg_medgemma4b_n400.json"
MAPPING = "outputs/image_cluster_20260922/image_mapping.json"
SPLITS = {"train": "outputs/legacy_mapping_20260923/source_train_a9083ce6.json",
          "validation": "results/exposure_audit_corrected_20261004/slake_official/validate.json",
          "test": "results/exposure_audit_corrected_20261004/slake_official/test.json"}
SPLIT_SHA = {"train": "7a9fb81f9d7bb145c3d7e9bc8d3b9016ba18b6539ea5031a53bae8fd3cd991c3",
             "validation": "32b016440b0c3be11056a78a18eeab46333268407fbb6e6b32f9f4c2debc50f6",
             "test": "6be8f7b4c5a46cdbc713a5210a25b6ed5aa1fd1574c83cefb4f998131f17c2c3"}
ORDER = ("test", "train", "validation")


class RouteError(Exception):
    pass


def _p(rel, root):
    return os.path.join(root, rel)


def compute(root=ROOT, receipt=RECEIPT, mapping=MAPPING):
    """Return {'per_item': {sample_id: (image_ref, membership)}, 'membership_counts', 'n_on_train_images', ...}.

    Raises RouteError on any identity failure instead of returning a partial answer."""
    splits = {}
    for sp, rel in SPLITS.items():
        raw = open(_p(rel, root), "rb").read()
        if hashlib.sha256(raw).hexdigest() != SPLIT_SHA[sp]:
            raise RouteError(f"split file {rel} sha256 differs from the registered official file")
        splits[sp] = {r["img_name"] for r in json.loads(raw)}
    rec = json.load(open(_p(receipt, root)))
    mp = json.load(open(_p(mapping, root)))
    by_id = {it["sample_id"]: it for it in mp["items"]}
    ambiguous = set(mp.get("ambiguous_candidate_ids", []))
    per_item = {}
    for row in rec["rows"]:
        sid = row["sample_id"]
        if sid in ambiguous:
            raise RouteError(f"{sid}: retained item has an ambiguous fingerprint in the mapping")
        it = by_id.get(sid)
        if it is None or not it.get("image_ref"):
            raise RouteError(f"{sid}: no image_ref in the mapping")
        # identity: the mapping entry must describe the same annotation as the receipt row
        if [list(map(float, b)) for b in it["organ_boxes"]] != [list(map(float, b)) for b in row["organ_boxes"]] \
                or list(it["image_size"]) != list(row["image_size"]):
            raise RouteError(f"{sid}: mapping boxes/size differ from the receipt row")
        hit = tuple(sp for sp in ORDER if it["image_ref"] in splits[sp])
        per_item[sid] = (it["image_ref"], ",".join(hit) if hit else "none")
    counts = Counter(m for _, m in per_item.values())
    n = len(per_item)
    n_train = sum(c for k, c in counts.items() if "train" in k.split(","))
    all_imgs = set().union(*splits.values())
    return {"route": "A_mapping", "n_items": n, "per_item": per_item,
            "membership_counts": dict(sorted(counts.items())),
            "n_on_train_images": n_train,
            "pct_on_train_images_1dp": f"{100 * n_train / n:.1f}",
            "n_test_and_train": counts.get("test,train", 0),
            "n_test_only": counts.get("test", 0),
            "split_sizes_by_imgname": {k: len(v) for k, v in splits.items()},
            "n_distinct_images_across_splits": len(all_imgs),
            "splits_are_image_disjoint": sum(len(v) for v in splits.values()) == len(all_imgs),
            "n_distinct_images_in_cohort": len({r for r, _ in per_item.values()})}


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        r = compute()
    except RouteError as e:
        print(f"[FATAL] route A: {e}"); sys.exit(1)
    s = {k: v for k, v in r.items() if k != "per_item"}
    print(json.dumps(s, indent=2))
    if out:
        json.dump({**s, "per_item": {k: list(v) for k, v in sorted(r["per_item"].items(),
                   key=lambda kv: int(kv[0].split('_')[1]))}}, open(out, "w"), indent=1)
        print(f"[SAVED] {out}")
