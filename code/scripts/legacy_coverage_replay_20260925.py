"""CPU replay of the legacy (minimum-overlap) control placement and its tissue coverage.

Registered in notes/PREREG_legacy_coverage_replay_20260925.md. No VLM is loaded.

The helper functions are taken byte-for-byte from the two run scripts by AST extraction
(so no torch import), and their source hashes are written to the identity receipt:
  legacy  : cipe_bbox_external_validation.py  random_boxes, mask_boxes
  matched : cipe_bbox_foreground_control.py   foreground_mask, _ibox, _coverage, _overlaps,
                                              place_control, control_draw

Usage:
  /usr/bin/python3 code/scripts/legacy_coverage_replay_20260925.py --imgs /path/to/slake/imgs
"""
from __future__ import annotations
import argparse, ast, hashlib, json, random, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "code" / "scripts"
MAPPING = ROOT / "outputs/image_cluster_20260922/image_mapping.json"
FG_RECEIPT = ROOT / "results/cipe_scale/bbox_fg_medgemma4b_n400.json"
LEGACY_RECEIPT = ROOT / "results/cipe_scale/bbox_val_medgemma4b_n400.json"
OUT = ROOT / "outputs/legacy_coverage_replay_20260925"
SEED_RUN = 42
SEED_BOOT = 20260925
N_BOOT = 20000


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def extract(path, names, ns):
    src = Path(path).read_text()
    tree = ast.parse(src)
    hashes = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            seg = ast.get_source_segment(src, node)
            exec(compile(seg, str(path), "exec"), ns)
            hashes[node.name] = hashlib.sha256(seg.encode()).hexdigest()
    missing = set(names) - set(hashes)
    assert not missing, missing
    return hashes


def boot_ci(values, clusters, seed=SEED_BOOT, n=N_BOOT):
    """Image-cluster percentile bootstrap of the item-level mean (items weighted equally)."""
    values = np.asarray(values, float)
    uniq = sorted(set(clusters))
    idx = {c: [i for i, k in enumerate(clusters) if k == c] for c in uniq}
    sums = np.array([values[idx[c]].sum() for c in uniq])
    cnts = np.array([len(idx[c]) for c in uniq])
    rs = np.random.default_rng(seed)
    draws = rs.integers(0, len(uniq), size=(n, len(uniq)))
    stats = sums[draws].sum(1) / cnts[draws].sum(1)
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))], len(uniq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs", required=True, help="SLAKE imgs root (xmlabN/source.jpg)")
    args = ap.parse_args()
    imgs = Path(args.imgs)

    # Constants of the matched script (FG_THRESH_FRAC, FG_TOL, MAX_PLACEMENT_TRIES) are
    # module-level; read them from its source, not re-typed here.
    ns = {"np": np, "Image": Image, "ImageDraw": ImageDraw, "random": random}
    fg_src = (SCRIPTS / "cipe_bbox_foreground_control.py").read_text()
    for node in ast.parse(fg_src).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and \
                node.targets[0].id in {"N_DRAWS", "FG_THRESH_FRAC", "FG_TOL", "MAX_PLACEMENT_TRIES"}:
            exec(compile(ast.get_source_segment(fg_src, node), "fg", "exec"), ns)
    h_leg = extract(SCRIPTS / "cipe_bbox_external_validation.py", ["random_boxes", "mask_boxes"], ns)
    h_fg = extract(SCRIPTS / "cipe_bbox_foreground_control.py",
                   ["foreground_mask", "_ibox", "_coverage", "_overlaps", "place_control",
                    "control_draw"], ns)
    random_boxes, mask_boxes = ns["random_boxes"], ns["mask_boxes"]
    foreground_mask, _ibox, _coverage = ns["foreground_mask"], ns["_ibox"], ns["_coverage"]
    control_draw, N_DRAWS = ns["control_draw"], ns["N_DRAWS"]

    mapping = json.loads(MAPPING.read_text())
    items = mapping["items"]
    assert [it["sample_id"] for it in items] == [f"slake_{i}" for i in range(400)]
    fgr = json.loads(FG_RECEIPT.read_text())
    stored = {r["sample_id"]: r for r in fgr["rows"]}
    stored_skip = {r["sample_id"]: r for r in fgr["skipped"]}
    leg = json.loads(LEGACY_RECEIPT.read_text())["rows"]
    assert [r["sample_id"] for r in leg] == [f"slake_{i}" for i in range(400)]

    rng_leg = random.Random(SEED_RUN)   # legacy main(): fresh Random(seed), sole consumer random_boxes
    rng_fg = random.Random(SEED_RUN)    # matched main(): fresh Random(seed), sole consumer control_draw
    sha_bad, size_bad, nbox_bad = [], [], []
    i1_organ_bad, i1_ctrl_bad, i2_box_bad, i2_partition_bad = [], [], [], []
    per_item = []
    for it, lr in zip(items, leg):
        sid = it["sample_id"]
        p = imgs / it["image_ref"]
        if sha(p) != it["image_sha256"]:
            sha_bad.append(sid)
        img = Image.open(p).convert("RGB")
        W, H = img.size
        if [W, H] != it["image_size"]:
            size_bad.append(sid)
        boxes = it["organ_boxes"]
        if len(boxes) != lr["n_boxes"]:
            nbox_bad.append(sid)

        # --- matched path (I1, I2) ---
        fg = foreground_mask(img)
        organ = [_ibox(b, W, H) for b in boxes]
        organ_cov = [_coverage(fg, b) for b in organ]
        draws = [control_draw(fg, organ, rng_fg) for _ in range(N_DRAWS)]
        draws = [d for d in draws if d[0] is not None]
        admitted = len(draws) == N_DRAWS
        ref = stored.get(sid) or stored_skip.get(sid)
        if ref is None or (sid in stored) != admitted:
            i2_partition_bad.append(sid)
        if ref is not None and organ_cov != ref["organ_fg_coverage"]:
            i1_organ_bad.append(sid)
        if admitted and sid in stored:
            if [list(map(list, d[0])) for d in draws] != stored[sid]["control_boxes"]:
                i2_box_bad.append(sid)
            if [d[1] for d in draws] != stored[sid]["control_fg_coverage"]:
                i1_ctrl_bad.append(sid)

        # --- legacy path (replay; I3) ---
        lboxes = random_boxes(img, boxes, rng_leg)
        leg_cov = [_coverage(fg, b) for b in lboxes]
        unchanged = bool(np.array_equal(np.asarray(mask_boxes(img, lboxes)), np.asarray(img)))
        per_item.append({
            "sample_id": sid, "image_ref": it["image_ref"], "modality": it.get("modality"),
            "admitted": admitted, "legacy_boxes": [list(b) for b in lboxes],
            "phi_legacy_boxes": leg_cov, "phi_legacy": float(np.mean(leg_cov)),
            "phi_annot": float(np.mean(organ_cov)),
            "phi_matched": (float(np.mean([np.mean(d[1]) for d in draws])) if admitted else None),
            "legacy_image_unchanged": unchanged,
            "legacy_cipe_random": lr["cipe_random"],
            "_leg_box_phis": leg_cov, "_annot_box_phis": organ_cov,
            "_matched_box_phis": [c for d in draws for c in d[1]] if admitted else [],
        })

    U = [r for r in per_item if r["legacy_image_unchanged"]]
    U_ok = [r for r in U if r["legacy_cipe_random"] == 1.0]
    notU = [r for r in per_item if not r["legacy_image_unchanged"]]
    identity = {
        "registration": "notes/PREREG_legacy_coverage_replay_20260925.md",
        "registration_sha256": sha(ROOT / "notes/PREREG_legacy_coverage_replay_20260925.md"),
        "script_sha256": sha(__file__),
        "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (MAPPING, FG_RECEIPT, LEGACY_RECEIPT)},
        "function_source_sha256": {"legacy": h_leg, "matched": h_fg},
        "python": sys.version.split()[0], "numpy": np.__version__,
        "pillow": Image.__version__,
        "image_sha256_mismatch": sha_bad, "image_size_mismatch": size_bad,
        "legacy_n_boxes_mismatch": nbox_bad,
        "I1_annotated_coverage_mismatch": i1_organ_bad,
        "I1_matched_coverage_mismatch": i1_ctrl_bad,
        "I2_matched_box_mismatch": i2_box_bad,
        "I2_partition_mismatch": i2_partition_bad,
        "n_admitted_replayed": sum(r["admitted"] for r in per_item),
        "I3_U_size": len(U), "I3_U_with_cipe_random_1": len(U_ok),
        "I3_U_ids": [r["sample_id"] for r in U],
        "I3_notU_with_cipe_random_1": sum(r["legacy_cipe_random"] == 1.0 for r in notU),
        "I3_notU_size": len(notU),
        "model_forward_gate": "skipped (needs a VLM); I3 is the CPU-only stand-in",
    }
    I1 = not (sha_bad or size_bad or nbox_bad or i1_organ_bad or i1_ctrl_bad)
    I2 = not (i2_box_bad or i2_partition_bad) and identity["n_admitted_replayed"] == 229
    I3 = len(U) >= 10 and len(U_ok) == len(U)
    identity.update({"I1_pass": I1, "I2_pass": I2, "I3_pass": I3, "gate_pass": I1 and I2 and I3})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "identity.json").write_text(json.dumps(identity, indent=1))
    print(json.dumps({k: v for k, v in identity.items() if k != "I3_U_ids"}, indent=1))
    if not identity["gate_pass"]:
        print("GATE FAIL -> verdict: cannot reproduce (ELN only)")
        return 1

    adm = [r for r in per_item if r["admitted"]]
    d = [r["phi_matched"] - r["phi_legacy"] for r in adm]
    ci, n_img = boot_ci(d, [r["image_ref"] for r in adm])
    dA = [r["phi_annot"] - r["phi_legacy"] for r in per_item]
    ciA, n_imgA = boot_ci(dA, [r["image_ref"] for r in per_item])
    dA_no246 = [x for x, r in zip(dA, per_item) if r["sample_id"] != "slake_246"]
    ciA_no246, _ = boot_ci(dA_no246, [r["image_ref"] for r in per_item if r["sample_id"] != "slake_246"])
    delta = float(np.mean(d))
    if ci[0] > 0 and delta >= 0.10:
        verdict = "A"
    elif ci[0] > 0:
        verdict = "B"
    else:
        verdict = "C"

    def share_lt(vals, t=0.5):
        return float(np.mean(np.asarray(vals) < t)) if vals else None

    by_mod = {}
    for r in adm:
        by_mod.setdefault(r["modality"], []).append(r)
    cov = {
        "primary_229": {
            "n_items": len(adm), "n_images": n_img,
            "phi_annot": float(np.mean([r["phi_annot"] for r in adm])),
            "phi_matched": float(np.mean([r["phi_matched"] for r in adm])),
            "phi_legacy": float(np.mean([r["phi_legacy"] for r in adm])),
            "delta_matched_minus_legacy": delta, "delta_ci95_image_cluster": ci,
            "verdict": verdict,
        },
        "secondary_400": {
            "n_items": len(per_item), "n_images": n_imgA,
            "phi_annot": float(np.mean([r["phi_annot"] for r in per_item])),
            "phi_legacy": float(np.mean([r["phi_legacy"] for r in per_item])),
            "delta_annot_minus_legacy": float(np.mean(dA)), "ci95_image_cluster": ciA,
            "delta_annot_minus_legacy_without_slake_246": float(np.mean(dA_no246)),
            "ci95_without_slake_246": ciA_no246,
        },
        "descriptive_box_share_phi_lt_0.5": {
            "legacy_229": share_lt([x for r in adm for x in r["_leg_box_phis"]]),
            "matched_229": share_lt([x for r in adm for x in r["_matched_box_phis"]]),
            "annot_229": share_lt([x for r in adm for x in r["_annot_box_phis"]]),
            "legacy_400": share_lt([x for r in per_item for x in r["_leg_box_phis"]]),
            "annot_400": share_lt([x for r in per_item for x in r["_annot_box_phis"]]),
        },
        "descriptive_by_modality_229": {
            m: {"n": len(v), "phi_legacy": float(np.mean([r["phi_legacy"] for r in v])),
                "phi_matched": float(np.mean([r["phi_matched"] for r in v]))}
            for m, v in sorted(by_mod.items(), key=lambda kv: str(kv[0]))},
        "bootstrap": {"seed": SEED_BOOT, "draws": N_BOOT, "type": "image-cluster percentile"},
        "phi_definition": "fg script foreground_mask (> min + 5% dynamic range), box slice [y:y+h, x:x+w]; per-item mean over boxes; summaries are means of per-item means",
        "items": [{k: v for k, v in r.items() if not k.startswith("_")} for r in per_item],
    }
    (OUT / "coverage.json").write_text(json.dumps(cov, indent=1))
    print(json.dumps({k: v for k, v in cov.items() if k != "items"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
