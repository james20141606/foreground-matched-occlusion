#!/usr/bin/env python3
"""Convert the disclosure-clean foreground receipts into the reader's per-model receipts.

Reads results/exposure_clean/bbox_fg_<tag>_chestxdet.json (written by the frozen
cipe_bbox_foreground_control.py with --dataset chestxdet) for the five registered
models, recomputes the CORRECTED total-variation W exactly as region_tv_corrected.py
does (item-level W, item bootstrap, two-sided p vs 0.5, BH-FDR across the five
models), and emits results/exposure_clean/expclean_<tag>.json in the schema
exposure_clean_read.py refuses to go without:

    {"W", "q", "n_items", "arms", "positive_control_flip", "readout": "corrected_tv"}

AMENDMENT 2 (PREREG_exposure_clean_region.md): the YES and NO arms are different
estimands (yes-items occlude the queried finding's boxes; no-items occlude ALL other
findings' boxes, 2.73 boxes/item vs the yes arm's 1.43), so per-arm W/p/q/n are
emitted alongside the combined quantities. The arm is encoded in sample_id
(cxdet_yes_* / cxdet_no_*); BH-FDR is applied across the five models WITHIN each
stratum (combined, yes, no). SEED is 42, matching the frozen analysis
(region_tv_corrected.SEED) as the registration's "same seeds" requires; the pipeline
originally ran with an accidental 0 (W is deterministic and did not move; p/q shift
within Monte-Carlo error).

The W math is imported from region_tv_corrected, not copied.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from region_tv_corrected import SEED, bh, norm, tv  # noqa: E402  (the frozen math+seed)

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "results/exposure_clean"
TAGS = ["medgemma4b", "qwen2vl_7b", "qwen25vl_7b", "gemma3_4b", "internvl3_8b"]
ARMS = ("yes", "no")
BOOT = 10_000


def arm_of(row):
    sid = row["sample_id"]
    for a in ARMS:
        if sid.startswith(f"cxdet_{a}_"):
            return a
    raise ValueError(f"sample_id {sid!r} encodes no registered arm (cxdet_yes_/cxdet_no_)")


def w_items(rows):
    W = []
    for r in rows:
        p0 = norm(np.asarray(r["p_original"], float))
        t_org = tv(p0, norm(np.asarray(r["p_organ"], float)))
        t_ctl = [tv(p0, norm(np.asarray(p, float))) for p in r["p_control_draws"]]
        W.append(float(np.mean([1.0 if t_org > tc else 0.0 for tc in t_ctl])))
    return np.asarray(W)


def boot_p(W, rng):
    n = len(W)
    wb = np.array([np.mean(W[rng.integers(0, n, n)]) for _ in range(BOOT)])
    p = float(2 * min(np.mean(wb <= 0.5), np.mean(wb >= 0.5)))
    return max(p, 1.0 / BOOT)


def main():
    recs, missing = {}, []
    for t in TAGS:
        fp = DIR / f"bbox_fg_{t}_chestxdet.json"
        if not fp.exists():
            missing.append(fp.name)
            continue
        recs[t] = json.loads(fp.read_text())
    if missing:
        print(f"REFUSED: missing foreground receipts: {missing}")
        return 2

    W = {t: {"all": w_items(r["rows"]),
             **{a: w_items([x for x in r["rows"] if arm_of(x) == a]) for a in ARMS}}
         for t, r in recs.items()}

    # One rng stream, consumed stratum by stratum (combined for all five models,
    # then yes, then no), so each stratum's five p-values share a reproducible draw.
    rng = np.random.default_rng(SEED)
    p = {s: {t: boot_p(W[t][s], rng) for t in TAGS} for s in ("all",) + ARMS}
    q = {s: dict(zip(TAGS, bh([p[s][t] for t in TAGS]))) for s in p}

    for t in TAGS:
        out = {"W": float(np.mean(W[t]["all"])), "q": float(q["all"][t]),
               "p": p["all"][t], "n_items": len(W[t]["all"]),
               "arms": {a: {"W": float(np.mean(W[t][a])), "q": float(q[a][t]),
                            "p": p[a][t], "n": len(W[t][a])} for a in ARMS},
               "positive_control_flip": float(recs[t]["image_sensitivity"]),
               "readout": "corrected_tv",
               "source": f"bbox_fg_{t}_chestxdet.json", "boot": BOOT, "seed": SEED}
        op = DIR / f"expclean_{t}.json"
        op.write_text(json.dumps(out, indent=1))
        arms = " | ".join(f"{a.upper()} W={out['arms'][a]['W']:.3f} "
                          f"q={out['arms'][a]['q']:.4f} n={out['arms'][a]['n']}"
                          for a in ARMS)
        print(f"  {t:14s} W={out['W']:.3f} p={out['p']:.4f} q={out['q']:.4f} "
              f"n={out['n_items']} | {arms} -> {op.name}")
    print("[OK] run exposure_clean_read.py for the branch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
