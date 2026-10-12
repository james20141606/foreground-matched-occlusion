"""Reconstruct the old SLAKE validation item order from its pinned train.json.

This script checks sample identity using annotation counts, never model outcomes.
It deliberately leaves historical receipts untouched.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--imgs", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    recs = json.loads(args.train.read_text())
    closed = [r for r in recs if r.get("q_lang") == "en"
              and str(r.get("answer")).strip().lower() in {"yes", "no"}]
    random.Random(42).shuffle(closed)
    items = []
    for r in closed:
        image_ref = r["img_name"]
        det = args.imgs / Path(image_ref).parent / "detection.json"
        image = args.imgs / image_ref
        if not det.exists() or not image.exists():
            continue
        try:
            entries = json.loads(det.read_text())
            # Exact old-loader rule in commit 783e737: take each first value,
            # then keep four-component lists/tuples in original order.
            boxes = [list(d.values())[0] for d in entries]
            boxes = [b for b in boxes if isinstance(b, (list, tuple)) and len(b) == 4]
        except Exception:
            continue
        if not boxes:
            continue
        items.append({
            "sample_id": f"slake_{len(items)}", "image_ref": image_ref,
            "question": r["question"],
            "answer": str(r["answer"]).strip().lower(),
            "qid": r.get("qid"), "img_id": r.get("img_id"),
            "n_boxes": len(boxes), "organ_boxes": boxes,
            "detection_sha256": sha256(det),
        })
        if len(items) == 400:
            break
    if len(items) != 400:
        raise RuntimeError(f"Recovered {len(items)} items, expected 400")

    receipt_paths = [
        "results/cipe_scale/bbox_val_gemma3_4b_n400.json",
        "results/cipe_scale/bbox_val_internvl3_8b_n400.json",
        "results/cipe_scale/bbox_val_qwen25vl7b_n400.json",
        "outputs/cipe_v0/bbox_val_qwen2vl7b_n400.json",
        "results/cipe_scale/bbox_val_medgemma4b_n400.json",
    ]
    checks = {}
    for name in receipt_paths:
        p = args.repo / name
        if not p.exists():
            # The oldest Qwen2 receipt may be archived only in the local repo.
            # Permit an unchanged copy inside this run's isolated output dir.
            p = args.out.parent / p.name
        rows = json.loads(p.read_text())["rows"]
        matched = sum(row["sample_id"] == item["sample_id"]
                      and row["n_boxes"] == item["n_boxes"]
                      for row, item in zip(rows, items))
        checks[name] = {"sha256": sha256(p), "rows": len(rows),
                        "ordered_sample_id_and_n_boxes_matches": matched}
    expected_general = receipt_paths[:4]
    if any(checks[name]["rows"] != 400 or
           checks[name]["ordered_sample_id_and_n_boxes_matches"] != 400
           for name in expected_general):
        raise RuntimeError("General legacy receipts failed 400/400 identity gate")

    payload = {
        "source": "BoKelvin/Slake train.json exact revision a9083ce6c34ac3ffb17671a605962924d8a8f9e9",
        "source_url": "https://huggingface.co/datasets/BoKelvin/Slake/blob/a9083ce6c34ac3ffb17671a605962924d8a8f9e9/train.json",
        "train_sha256": sha256(args.train),
        "loader_code_commit": "783e7379856708a6740efc06e6747cf7ebba0289",
        "loader_code_path": "code/scripts/cipe_bbox_external_validation.py",
        "selection": "English yes/no, random.Random(42).shuffle, image and detection existence, nonempty four-element boxes, first 400",
        "identity_gate": "All four old general receipts have 400/400 ordered sample_id+n_boxes matches; this does not independently prove historical QA text or pixel bytes.",
        "receipt_checks": checks,
        "items": items,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "train_sha256": payload["train_sha256"],
                      "receipt_checks": checks}, indent=2))


if __name__ == "__main__":
    main()
