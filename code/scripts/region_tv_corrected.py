#!/usr/bin/env python3
"""Corrected region readout: normalize the stored vectors, report both estimands.

Implements PREREG_region_tv_correction.md, committed before this file.

THE DEFECT. cipe_bbox_external_validation.probs() returns
[exp(lp_yes - m), exp(lp_no - m)] with m = max(lp), so the max element is exactly
1.0 and the vector sums to (1, 2]. Measured over the committed receipts: 2378 of
2748 stored vectors do not sum to 1. cipe.py normalizes internally, so CIPE is
unaffected; TV and |dp(yes)| were computed on the raw weights, so the published
"total variation" is not total variation.

THE ESTIMANDS, both reported, neither chosen after the fact:
  D_i = TV_organ,i - mean_k TV_control,ik            item bootstrap CI on mean D
  W_i = mean_k 1[TV_organ,i > TV_control,ik]         controls clustered within item
  legacy = 1[TV_organ,i > mean_k TV_control,ik]      the published quantity, deprecated:
           the five-draw mean is fine for D but carries no chance-0.5 interpretation.
Plus all five single-draw win rates, so no draw is selected.

Receipt-only: no inference is re-run.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

BOOT = 10000
SEED = 42


def norm(v):
    a = np.asarray(v, float)
    s = a.sum()
    return a / s if s > 0 else a


def tv(p, q):
    return 0.5 * float(np.abs(norm(p) - norm(q)).sum())


def boot_ci(x, stat, rng, b=BOOT):
    x = np.asarray(x, float)
    n = len(x)
    vals = np.array([stat(x[rng.integers(0, n, n)]) for _ in range(b)])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bh(pvals):
    p = np.asarray(pvals, float)
    o = np.argsort(p)
    m = len(p)
    q = np.empty(m)
    prev = 1.0
    for rank, i in enumerate(o[::-1]):
        prev = min(prev, p[i] * m / (m - rank))
        q[i] = prev
    return q


def main():
    rng = np.random.default_rng(SEED)
    rows_out = []
    unnorm_total = unnorm_bad = 0

    for fp in sorted(glob.glob("results/cipe_scale/bbox_fg_*_n400.json")):
        d = json.load(open(fp))
        tag = os.path.basename(fp).replace("bbox_fg_", "").replace("_n400.json", "")
        rows = [r for r in d.get("rows", [])
                if r.get("p_original") and r.get("p_organ") and r.get("p_control_draws")]
        if not rows:
            continue
        D, W, legacy, per_draw = [], [], [], None
        for r in rows:
            for v in (r["p_original"], r["p_organ"]):
                unnorm_total += 1
                if abs(np.sum(v) - 1.0) > 1e-6:
                    unnorm_bad += 1
            t_org = tv(r["p_original"], r["p_organ"])
            t_ctl = [tv(r["p_original"], pc) for pc in r["p_control_draws"]]
            if per_draw is None:
                per_draw = [[] for _ in t_ctl]
            for k, tc in enumerate(t_ctl):
                per_draw[k].append(1.0 if t_org > tc else 0.0)
            D.append(t_org - float(np.mean(t_ctl)))
            W.append(float(np.mean([1.0 if t_org > tc else 0.0 for tc in t_ctl])))
            legacy.append(1.0 if t_org > float(np.mean(t_ctl)) else 0.0)

        n = len(rows)
        d_lo, d_hi = boot_ci(D, np.mean, rng)
        w_lo, w_hi = boot_ci(W, np.mean, rng)          # item bootstrap == clustering
        # two-sided p for W against 0.5 by the item bootstrap
        wb = np.array([np.mean(np.asarray(W)[rng.integers(0, n, n)]) for _ in range(BOOT)])
        p_w = 2 * min((wb <= 0.5).mean(), (wb >= 0.5).mean())
        p_w = max(p_w, 1.0 / BOOT)
        rows_out.append({
            "model": tag, "n": n,
            "D_mean": float(np.mean(D)), "D_ci": [d_lo, d_hi],
            "W_mean": float(np.mean(W)), "W_ci": [w_lo, w_hi], "W_p": float(p_w),
            "legacy_frac": float(np.mean(legacy)),
            "per_draw_frac": [float(np.mean(x)) for x in per_draw],
        })

    qs = bh([r["W_p"] for r in rows_out])
    for r, q in zip(rows_out, qs):
        r["W_q"] = float(q)

    print(f"normalization audit: {unnorm_bad} of {unnorm_total} stored vectors did not "
          f"sum to 1 before correction\n")
    print(f"{'model':14s} {'n':>4s} {'D (organ-ctl)':>22s} {'W (clustered)':>22s} "
          f"{'q':>7s} {'legacy':>7s}  per-draw win rates")
    for r in sorted(rows_out, key=lambda x: x["model"]):
        print(f"{r['model']:14s} {r['n']:4d} "
              f"{r['D_mean']:+.4f}[{r['D_ci'][0]:+.4f},{r['D_ci'][1]:+.4f}] "
              f"{r['W_mean']:.3f}[{r['W_ci'][0]:.3f},{r['W_ci'][1]:.3f}] "
              f"{r['W_q']:7.4f} {r['legacy_frac']:7.3f}  "
              + " ".join(f"{x:.3f}" for x in r["per_draw_frac"]))

    spec = [r for r in rows_out if "medgemma" in r["model"]]
    gen = [r for r in rows_out if "medgemma" not in r["model"]]
    spec_sig = [r for r in spec if r["W_q"] < 0.05 and r["W_ci"][0] > 0.5]
    gen_sig = [r for r in gen if r["W_q"] < 0.05 and r["W_ci"][0] > 0.5]
    print("\n=== PRE-REGISTERED BRANCH (PREREG_region_tv_correction.md) ===")
    if len(spec_sig) == 0:
        print(">>> R3 EFFECT DOES NOT SURVIVE: no specialist checkpoint is significant "
              "under the corrected estimands. The region result is retracted and joins "
              "the artefact catalogue as a fifth reversal.")
    elif len(spec_sig) == 1:
        print(f">>> R4 SPLIT: only {spec_sig[0]['model']} survives; family-level "
              "replication is gone. Report a single-checkpoint effect and say so.")
    elif len(gen_sig) <= 1:
        print(">>> R1 SEPARATION SURVIVES: both specialist checkpoints significant and "
              f"{len(gen)-len(gen_sig)} of {len(gen)} general models not. The "
              "specialist-vs-general claim stands on the corrected readout.")
    else:
        print(f">>> R2 EFFECT SURVIVES, SEPARATION DOES NOT: both specialists significant "
              f"but so are {len(gen_sig)} of {len(gen)} general models. RETIRE the "
              "specialist-vs-general claim, including in the abstract; report a graded "
              "occlusion sensitivity across the panel with the specialists strongest.")
    json.dump(rows_out, open("outputs/region_tv_corrected.json", "w"), indent=2)
    print("\n[SAVED] outputs/region_tv_corrected.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
