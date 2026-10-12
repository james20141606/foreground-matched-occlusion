"""Outcome-blind identity gate, then stored-mask paired description for 34 QA.

Protocol: outputs/legacy_mapping_20260923/paired34_protocol.md.
No p-values, model selection, new inference, or MedTrust-TV analysis.
"""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


MODELS = ("gemma3_4b", "internvl3_8b", "qwen25vl7b", "qwen2vl7b")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row):
    return (row["image_ref"], row["question"], str(row["answer"]).strip().lower())


def mean(xs):
    return sum(xs) / len(xs)


def percentile(sorted_xs, q):
    pos = q * (len(sorted_xs) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (pos - lo) * (sorted_xs[hi] - sorted_xs[lo])


def ci(values, alpha):
    xs = sorted(values)
    return [percentile(xs, alpha / 2), percentile(xs, 1 - alpha / 2)]


def half_tie(delta):
    return 1.0 if delta > 0 else 0.0 if delta < 0 else 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root = args.repo
    run = root / "outputs/legacy_mapping_20260923"
    old_map_path = run / "general_identity_mapping.json"
    new_map_path = root / "outputs/image_cluster_20260922/image_mapping.json"
    protocol_path = run / "paired34_protocol.md"
    old_map = json.loads(old_map_path.read_text())
    new_map = json.loads(new_map_path.read_text())
    old, new = old_map["items"], new_map["items"]
    oc, nc = Counter(map(key, old)), Counter(map(key, new))
    old_by_key = {key(x): x for x in old if oc[key(x)] == 1}
    new_by_key = {key(x): x for x in new if nc[key(x)] == 1}
    shared = sorted(set(old_by_key) & set(new_by_key))
    if len(shared) != 62 or any(old_by_key[k]["organ_boxes"] !=
                                new_by_key[k]["organ_boxes"] for k in shared):
        raise RuntimeError("Shared QA/box identity gate changed")
    fg_selection_path = root / "results/cipe_scale/bbox_fg_gemma3_4b_n400.json"
    fg_ids = {x["sample_id"] for x in json.loads(fg_selection_path.read_text())["rows"]}
    selected = [k for k in shared if new_by_key[k]["sample_id"] in fg_ids]
    if len(selected) != 34:
        raise RuntimeError(f"Expected 34 frozen selected QA, found {len(selected)}")
    image_counts = dict(sorted(Counter(k[0] for k in selected).items()))
    models = {}
    for model in MODELS:
        op = root / ("outputs/cipe_v0" if model == "qwen2vl7b"
                     else "results/cipe_scale") / f"bbox_val_{model}_n400.json"
        fp = root / "results/cipe_scale" / f"bbox_fg_{model}_n400.json"
        old_rows = {r["sample_id"]: r for r in json.loads(op.read_text())["rows"]}
        new_rows = {r["sample_id"]: r for r in json.loads(fp.read_text())["rows"]}
        if set(new_rows) != fg_ids:
            raise RuntimeError(f"Foreground inclusion set differs for {model}")
        rows = []
        for k in selected:
            om, nm = old_by_key[k], new_by_key[k]
            a, b = old_rows[om["sample_id"]], new_rows[nm["sample_id"]]
            if a["cipe_relevant"] != b["cipe_organ"]:
                raise RuntimeError(f"Original organ CIPE differs for {model} {k}")
            draws = b["cipe_control_draws"]
            if len(draws) != 5:
                raise RuntimeError(f"Expected five foreground draws for {model} {k}")
            dl = a["cipe_random"] - a["cipe_relevant"]
            df = mean(draws) - b["cipe_organ"]
            rows.append({
                "old_sample_id": om["sample_id"], "new_sample_id": nm["sample_id"],
                "image_ref": om["image_ref"], "image_sha256": nm["image_sha256"],
                "question": om["question"], "answer": om["answer"], "qid": om["qid"],
                "ordered_boxes": om["organ_boxes"],
                "cipe_relevant_equals_organ": a["cipe_relevant"],
                "legacy_random": a["cipe_random"],
                "foreground_control_draws": draws,
                "D_L": dl, "D_F": df, "D_L_minus_D_F": dl - df,
                "legacy_half_tie_win": half_tie(dl),
                "foreground_half_tie_win": mean([half_tie(x - b["cipe_organ"])
                                                 for x in draws]),
            })
        by_image = defaultdict(list)
        for row in rows:
            by_image[row["image_ref"]].append(row)
        if dict(sorted((k, len(v)) for k, v in by_image.items())) != image_counts:
            raise RuntimeError("Image roster differs across models")
        image_keys = sorted(by_image)
        rng = random.Random(20260923)
        replicates = {"D_L": [], "D_F": [], "D_L_minus_D_F": []}
        for _ in range(20000):
            sampled = [rng.choice(image_keys) for _ in image_keys]
            sampled_rows = [r for image in sampled for r in by_image[image]]
            for statistic in replicates:
                replicates[statistic].append(mean([r[statistic] for r in sampled_rows]))
        estimates = {}
        for statistic in replicates:
            estimates[statistic] = {
                "qa_weighted_mean": mean([r[statistic] for r in rows]),
                "cluster_bootstrap_95_percentile_ci": ci(replicates[statistic], 0.05),
                "cluster_bootstrap_bonferroni_98_75_percentile_ci":
                    ci(replicates[statistic], 0.05 / len(MODELS)),
            }
        models[model] = {
            "old_receipt_sha256": digest(op), "foreground_receipt_sha256": digest(fp),
            "n_qa": len(rows), "n_images": len(image_keys),
            "estimates": estimates,
            "legacy_half_tie_win_fraction": mean([r["legacy_half_tie_win"] for r in rows]),
            "foreground_half_tie_win_fraction":
                mean([r["foreground_half_tie_win"] for r in rows]),
            "image_contributions": [{"image_ref": image,
                                     "image_sha256": by_image[image][0]["image_sha256"],
                                     "n_qa": len(by_image[image]),
                                     "mean_D_L_minus_D_F":
                                         mean([r["D_L_minus_D_F"] for r in by_image[image]])}
                                    for image in image_keys],
            "qa_contributions": rows,
        }
    payload = {
        "scope": "34 true shared included SLAKE QA, four models, stored masks only; no extrapolation to 229/400 or MedTrust-TV",
        "protocol_sha256": digest(protocol_path),
        "source_train_sha256": old_map["train_sha256"],
        "old_mapping_sha256": digest(old_map_path),
        "new_mapping_sha256": digest(new_map_path),
        "foreground_selection_receipt_sha256": digest(fg_selection_path),
        "n_shared_candidate_qa": len(shared), "n_selected_shared_qa": len(selected),
        "n_images": len(image_counts), "qa_counts_by_image": image_counts,
        "bootstrap": {"unit": "image_ref (bound to current image_sha256)",
                      "within_cluster_weight": "all QA rows, variable-cluster-size QA mean",
                      "replicates": 20000, "seed": 20260923,
                      "intervals": "percentile 95%, Bonferroni 98.75% across four models"},
        "models": models,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({name: {"n_images": v["n_images"],
                             "estimates": v["estimates"],
                             "legacy_half_tie_win_fraction": v["legacy_half_tie_win_fraction"],
                             "foreground_half_tie_win_fraction":
                                 v["foreground_half_tie_win_fraction"]}
                      for name, v in models.items()}, indent=2))


if __name__ == "__main__":
    main()
