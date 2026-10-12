"""Outcome-blind identity gate for the MedGemma-4B legacy-vs-matched pairing.

Reads only item identifiers, box counts/lists and the annotated-mask CIPE.
It never reads the legacy control value (cipe_random) or any matched control
draw, so it cannot reveal the outcome of the paired contrast.

Alignment is by sample_id (never by list position), and identity is then
checked against three independent fingerprints per item:
  (1) legacy n_boxes == matched n_boxes == len(image_mapping organ_boxes)
  (2) matched organ_boxes == image_mapping organ_boxes (ordered, exact)
  (3) legacy cipe_relevant == matched cipe_organ (exact float equality)
Items whose annotated-mask CIPE is exactly 0 or 1 (clip boundary) carry no
fingerprint information in (3); they are counted and reported separately.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

LEGACY = "results/cipe_scale/bbox_val_medgemma4b_n400.json"
MATCHED = "results/cipe_scale/bbox_fg_medgemma4b_n400.json"
MAPPING = "outputs/image_cluster_20260922/image_mapping.json"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    lp, mp, gp = (a.repo / LEGACY, a.repo / MATCHED, a.repo / MAPPING)
    L, M, G = (json.loads(p.read_text()) for p in (lp, mp, gp))
    lrows, mrows, grows = L["rows"], M["rows"], G["items"]
    for name, rows in (("legacy", lrows), ("matched", mrows), ("mapping", grows)):
        c = Counter(r["sample_id"] for r in rows)
        if any(v > 1 for v in c.values()):
            raise RuntimeError(f"duplicate sample_id in {name}")
    lby = {r["sample_id"]: r for r in lrows}
    gby = {r["sample_id"]: r for r in grows}
    ids = [r["sample_id"] for r in mrows]
    missing_legacy = [i for i in ids if i not in lby]
    missing_map = [i for i in ids if i not in gby]
    nbox_mismatch, box_mismatch, cipe_mismatch, boundary = [], [], [], []
    position_differs = 0
    lpos = {r["sample_id"]: k for k, r in enumerate(lrows)}
    for k, r in enumerate(mrows):
        sid = r["sample_id"]
        if sid in missing_legacy or sid in missing_map:
            continue
        lr, gr = lby[sid], gby[sid]
        if lpos[sid] != k:
            position_differs += 1
        if not (lr["n_boxes"] == r["n_boxes"] == len(gr["organ_boxes"])):
            nbox_mismatch.append(sid)
        if [list(map(float, b)) for b in r["organ_boxes"]] != \
                [list(map(float, b)) for b in gr["organ_boxes"]]:
            box_mismatch.append(sid)
        if lr["cipe_relevant"] != r["cipe_organ"]:
            cipe_mismatch.append(sid)
        if r["cipe_organ"] in (0.0, 1.0):
            boundary.append(sid)
    # legacy vs mapping box counts over the full 400 (sequence identity)
    full_nbox_mismatch = [r["sample_id"] for r in lrows
                          if r["sample_id"] not in gby
                          or r["n_boxes"] != len(gby[r["sample_id"]]["organ_boxes"])]
    images = Counter(gby[i]["image_sha256"] for i in ids if i in gby)
    out = {
        "scope": "outcome-blind identity gate; reads no control value",
        "legacy_receipt": LEGACY, "legacy_sha256": sha(lp),
        "matched_receipt": MATCHED, "matched_sha256": sha(mp),
        "mapping": MAPPING, "mapping_sha256": sha(gp),
        "script_sha256": sha(Path(__file__)),
        "legacy_model_path": L["model_path"], "matched_model_path": M["model_path"],
        "legacy_image_sensitivity": L["image_sensitivity"],
        "matched_image_sensitivity": M["image_sensitivity"],
        "n_legacy": len(lrows), "n_matched_admitted": len(ids),
        "missing_in_legacy": missing_legacy, "missing_in_mapping": missing_map,
        "matched_rows_at_different_list_position_in_legacy": position_differs,
        "n_boxes_mismatch": nbox_mismatch,
        "ordered_boxes_mismatch_vs_mapping": box_mismatch,
        "annotated_cipe_mismatch": cipe_mismatch,
        "annotated_cipe_at_clip_boundary": len(boundary),
        "n_distinct_annotated_cipe_values": len({lby[i]["cipe_relevant"] for i in ids}),
        "legacy_vs_mapping_n_boxes_mismatch_all400": len(full_nbox_mismatch),
        "n_images_229": len(images),
        "qa_per_image_histogram": dict(sorted(Counter(images.values()).items())),
    }
    out["gate_pass"] = (not missing_legacy and not missing_map and not nbox_mismatch
                        and not box_mismatch and not cipe_mismatch
                        and out["legacy_vs_mapping_n_boxes_mismatch_all400"] == 0
                        and L["model_path"] == M["model_path"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    raise SystemExit(0 if out["gate_pass"] else 1)


if __name__ == "__main__":
    main()
