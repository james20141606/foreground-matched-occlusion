"""Compare recovered old SLAKE items to the foreground control's item map.

Identity is decided by image/question/answer and ordered boxes, before any
CIPE values are inspected. This script does not estimate effects or p-values.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(item):
    return (item["image_ref"], item["question"], str(item["answer"]).strip().lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root = args.repo
    oldp = root / "outputs/legacy_mapping_20260923/general_identity_mapping.json"
    newp = root / "outputs/image_cluster_20260922/image_mapping.json"
    old = json.loads(oldp.read_text())["items"]
    new = json.loads(newp.read_text())["items"]
    old_counts, new_counts = Counter(map(key, old)), Counter(map(key, new))
    # No ambiguous duplicate key may be treated as a one-to-one shared case.
    one_to_one = set(old_counts) & set(new_counts)
    one_to_one = {k for k in one_to_one if old_counts[k] == new_counts[k] == 1}
    new_by_key = {key(x): x for x in new if key(x) in one_to_one}
    pairs = [(x, new_by_key[key(x)]) for x in old if key(x) in one_to_one]
    selected = [(x, y) for x, y in pairs
                if y["sample_id"] in {
                    r["sample_id"] for r in json.loads(
                        (root / "results/cipe_scale/bbox_fg_gemma3_4b_n400.json").read_text()
                    )["rows"]
                }]
    box_matches = sum(x["organ_boxes"] == y["organ_boxes"] for x, y in pairs)
    if box_matches != len(pairs):
        raise RuntimeError("Shared QA keys have discordant ordered boxes")

    model_checks = {}
    for model in ("gemma3_4b", "internvl3_8b", "qwen25vl7b", "qwen2vl7b"):
        old_receipt = root / ("outputs/cipe_v0" if model == "qwen2vl7b"
                              else "results/cipe_scale") / f"bbox_val_{model}_n400.json"
        new_receipt = root / "results/cipe_scale" / f"bbox_fg_{model}_n400.json"
        old_rows = {r["sample_id"]: r for r in json.loads(old_receipt.read_text())["rows"]}
        new_rows = {r["sample_id"]: r for r in json.loads(new_receipt.read_text())["rows"]}
        deltas = [abs(old_rows[x["sample_id"]]["cipe_relevant"] -
                      new_rows[y["sample_id"]]["cipe_organ"])
                  for x, y in selected if y["sample_id"] in new_rows]
        model_checks[model] = {
            "n_shared_included": len(deltas),
            "exact_cipe_relevant_vs_organ": sum(d == 0 for d in deltas),
            "max_absolute_difference": max(deltas) if deltas else None,
            "old_receipt_sha256": sha256(old_receipt),
            "foreground_receipt_sha256": sha256(new_receipt),
        }

    payload = {
        "old_mapping_sha256": sha256(oldp), "new_mapping_sha256": sha256(newp),
        "identity_rule": "one-to-one exact (image_ref, question, normalized answer); ordered organ boxes must also match",
        "old_candidates": len(old), "new_candidates": len(new),
        "old_unique_qa_keys": len(old_counts), "new_unique_qa_keys": len(new_counts),
        "shared_one_to_one_qa": len(pairs),
        "shared_ordered_boxes_equal": box_matches,
        "shared_foreground_included": len(selected),
        "same_sample_id_same_qa": sum(key(x) == key(new[int(x["sample_id"].split("_")[1])])
                                      for x in old),
        "shared_id_pairs": [{"old_sample_id": x["sample_id"],
                             "new_sample_id": y["sample_id"],
                             "image_ref": x["image_ref"],
                             "qid": x.get("qid"),
                             "foreground_included": (x, y) in selected}
                            for x, y in pairs],
        "model_checks": model_checks,
        "limit": "Historical receipts omit QA text and image bytes. Source-revision loader replay and cross-run exact CIPE equality corroborate shared cases, not independently prove every historical QA/pixel.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items()
                      if k not in {"shared_id_pairs"}}, indent=2))


if __name__ == "__main__":
    main()
