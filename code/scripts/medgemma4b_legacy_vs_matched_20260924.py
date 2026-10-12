"""MedGemma-4B: legacy vs foreground-matched control on the same 229 QA (CIPE readout).

Registration: notes/PREREG_medgemma4b_legacy_vs_matched_20260924.md (commit 51a656b).
Receipt-only; no inference is re-run.

Per item i (aligned by sample_id), annotation wins when CIPE_A < CIPE_C:
  w^L_i = 1[CIPE_A,i < CIPE_L,i]                (one legacy draw)
  w^F_i = mean_k 1[CIPE_A,i < CIPE_F,ik]        (five matched draws, averaged indicator)
Primary: dW = mean_i (w^L_i - w^F_i), ties score 0.
Inference: image-cluster percentile bootstrap (95%) and image-cluster sign-flip test,
B = 20000, seed 20260924.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

LEGACY = "results/cipe_scale/bbox_val_medgemma4b_n400.json"
MATCHED = "results/cipe_scale/bbox_fg_medgemma4b_n400.json"
MAPPING = "outputs/image_cluster_20260922/image_mapping.json"
TVREF = "outputs/region_tv_corrected.json"
PREREG = "notes/PREREG_medgemma4b_legacy_vs_matched_20260924.md"
B = 20000
SEED = 20260924
K = 5


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def norm(v):
    a = np.asarray(v, float)
    return a / a.sum()


def tv(p, q):
    return 0.5 * float(np.abs(norm(p) - norm(q)).sum())


def strict(a, c):
    return 1.0 if a < c else 0.0


def half(a, c):
    return 1.0 if a < c else (0.5 if a == c else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root = args.repo
    sys.path.insert(0, str(root / "code"))
    from medtrust.cipe import cipe_score  # noqa: E402

    L = json.loads((root / LEGACY).read_text())
    M = json.loads((root / MATCHED).read_text())
    G = json.loads((root / MAPPING).read_text())
    lby = {r["sample_id"]: r for r in L["rows"]}
    gby = {r["sample_id"]: r for r in G["items"]}
    mrows = M["rows"]
    if len(mrows) != 229 or len(lby) != 400:
        raise RuntimeError("unexpected receipt size")

    # ---- pipeline checks (registered; not outcomes) -------------------------
    tv_w, cipe_recompute_maxdiff = [], 0.0
    for r in mrows:
        t_a = tv(r["p_original"], r["p_organ"])
        tv_w.append(np.mean([1.0 if t_a > tv(r["p_original"], pc) else 0.0
                             for pc in r["p_control_draws"]]))
        for pc, stored in zip(r["p_control_draws"], r["cipe_control_draws"]):
            c = cipe_score(r["p_original"], pc).cipe
            cipe_recompute_maxdiff = max(cipe_recompute_maxdiff, abs(c - stored))
        c = cipe_score(r["p_original"], r["p_organ"]).cipe
        cipe_recompute_maxdiff = max(cipe_recompute_maxdiff, abs(c - r["cipe_organ"]))
    tv_w_mean = float(np.mean(tv_w))
    ref = [x for x in json.loads((root / TVREF).read_text()) if x["model"] == "medgemma4b"][0]
    if round(tv_w_mean, 4) != round(ref["W_mean"], 4):
        raise RuntimeError(f"TV W check failed {tv_w_mean} vs {ref['W_mean']}")
    if cipe_recompute_maxdiff > 1e-9:
        raise RuntimeError(f"stored CIPE != recomputed ({cipe_recompute_maxdiff})")

    # ---- per-item rows, aligned by sample_id ---------------------------------
    rows = []
    for r in mrows:
        sid = r["sample_id"]
        lr, gr = lby[sid], gby[sid]
        a = r["cipe_organ"]
        if lr["cipe_relevant"] != a or lr["n_boxes"] != r["n_boxes"]:
            raise RuntimeError(f"identity gate broken at {sid}")
        draws = r["cipe_control_draws"]
        if len(draws) != K:
            raise RuntimeError(f"{sid}: {len(draws)} draws")
        cl = lr["cipe_random"]
        wl, wf = strict(a, cl), float(np.mean([strict(a, c) for c in draws]))
        wls, wfs = half(a, cl), float(np.mean([half(a, c) for c in draws]))
        rows.append({
            "sample_id": sid, "image_sha256": gr["image_sha256"],
            "cipe_annotation": a, "cipe_legacy_control": cl, "cipe_matched_draws": draws,
            "wL": wl, "wF": wf, "wL_half": wls, "wF_half": wfs,
            "wF_draw": [strict(a, c) for c in draws],
            "legacy_tie": a == cl, "matched_ties": sum(a == c for c in draws),
            "d": wl - wf, "d_half": wls - wfs,
            "dD_cipe": (cl - a) - (float(np.mean(draws)) - a),
        })

    by_img = defaultdict(list)
    for i, r in enumerate(rows):
        by_img[r["image_sha256"]].append(i)
    imgs = sorted(by_img)
    idx = [np.array(by_img[g]) for g in imgs]
    n = len(rows)

    def arr(key):
        return np.array([r[key] for r in rows], float)

    stats = {
        "dW_strict": arr("d"), "dW_half": arr("d_half"), "dD_cipe": arr("dD_cipe"),
        "W_L": arr("wL"), "W_F": arr("wF"), "W_L_half": arr("wL_half"),
        "W_F_half": arr("wF_half"),
    }
    for k in range(K):
        stats[f"dW_draw{k + 1}"] = arr("wL") - np.array([r["wF_draw"][k] for r in rows])

    # image-cluster bootstrap: per-image sums and counts, QA-weighted mean
    rng = np.random.default_rng(SEED)
    draws_idx = rng.integers(0, len(imgs), size=(B, len(imgs)))
    cnt = np.array([len(ix) for ix in idx], float)
    boot_n = cnt[draws_idx].sum(1)
    est = {}
    for name, x in stats.items():
        s = np.array([x[ix].sum() for ix in idx])
        bs = s[draws_idx].sum(1) / boot_n
        est[name] = {"estimate": float(x.mean()),
                     "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}

    # image-cluster sign-flip test on primary
    s = np.array([stats["dW_strict"][ix].sum() for ix in idx])
    t_obs = s.sum() / n
    rng2 = np.random.default_rng(SEED + 1)
    signs = rng2.choice([-1.0, 1.0], size=(B, len(imgs)))
    t_null = (signs * s).sum(1) / n
    p_signflip = float((1 + np.sum(np.abs(t_null) >= abs(t_obs) - 1e-15)) / (1 + B))

    lo, hi = est["dW_strict"]["ci95"]
    if lo > 0 and p_signflip < 0.05:
        verdict = "A"
    elif hi < 0 and p_signflip < 0.05:
        verdict = "B"
    elif lo <= 0 <= hi and p_signflip >= 0.05:
        verdict = "C"
    else:
        verdict = "C'"
    lo_h, hi_h = est["dW_half"]["ci95"]
    half_verdict = "A" if lo_h > 0 else ("B" if hi_h < 0 else "C")

    wf_bin = Counter()
    for r in rows:
        f = "F=1" if r["wF"] == 1 else ("F=0" if r["wF"] == 0 else "F=mixed")
        wf_bin[f"L={int(r['wL'])},{f}"] += 1

    out = {
        "scope": ("MedGemma-4B only; SLAKE 229 admitted QA shared by legacy and matched runs; "
                  "CIPE readout (legacy stores no probability vectors, so TV is not computable); "
                  "one legacy draw vs five matched draws; post-hoc"),
        "registration": PREREG, "registration_sha256": sha(root / PREREG),
        "registration_commit": "51a656b",
        "inputs": {p: sha(root / p) for p in (LEGACY, MATCHED, MAPPING, TVREF)},
        "script_sha256": sha(Path(__file__)),
        "seed": SEED, "bootstrap_replicates": B, "signflip_replicates": B,
        "signflip_seed": SEED + 1,
        "resampling_unit": "source image (image_sha256), all QA rows, QA-weighted mean",
        "n_qa": n, "n_images": len(imgs),
        "pipeline_checks": {
            "matched_TV_W_recomputed": tv_w_mean, "region_tv_corrected_W": ref["W_mean"],
            "stored_vs_recomputed_CIPE_maxabsdiff": cipe_recompute_maxdiff,
        },
        "estimates": est,
        "primary": {"dW_strict": t_obs, "ci95": est["dW_strict"]["ci95"],
                    "signflip_p_two_sided": p_signflip, "verdict": verdict},
        "half_tie_verdict_by_ci": half_verdict,
        "ties": {"legacy_tie_fraction": float(np.mean([r["legacy_tie"] for r in rows])),
                 "matched_tie_fraction": float(np.mean([r["matched_ties"] / K for r in rows]))},
        "discordance": dict(sorted(wf_bin.items())),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    brief = {k: v for k, v in out.items() if k != "rows"}
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
