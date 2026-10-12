#!/usr/bin/env python3
"""Monte-Carlo seed sensitivity of two decision-adjacent numbers (BootLoops F8), 2026-10-04.

(1) InternVL3-8B strict-W image-cluster sign-flip Holm p (Fig. 2 caption prints 0.053 at seed
    20260922): reruns image_cluster_audit_20260922.py unchanged except SEED, seeds 1..40.
(2) MedGemma-4B paired dW_strict image-cluster bootstrap 95% CI lower bound (printed +0.005 at
    seed 20260924): same bootstrap as medgemma4b_legacy_vs_matched_20260924.py on the stored
    rows of paired_contrast.json, seeds 1..40.
Point estimates do not depend on the seed. Output: results/seed_sensitivity_20261004/summary.json
"""
import json, os, subprocess, sys, tempfile
from collections import defaultdict
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[2]
SEEDS = list(range(1, 41))


def holm_sweep():
    src = (R / "code/scripts/image_cluster_audit_20260922.py").read_text()
    if src.count("SEED=20260922") != 1:
        sys.exit("SEED line not found exactly once")
    out = {}
    with tempfile.TemporaryDirectory() as td:
        p = Path(td, "ica.py")
        p.write_text(src.replace("SEED=20260922", "import os\nSEED=int(os.environ['SEEDX'])"))
        for s in [20260922] + SEEDS:
            o = Path(td, f"s{s}.json")
            subprocess.run([sys.executable, str(p), "--root", str(R), "--mapping",
                            str(R / "outputs/image_cluster_20260922/image_mapping.json"), "--out", str(o)],
                           env={**os.environ, "SEEDX": str(s)}, check=True, capture_output=True)
            out[s] = {r["model"]: r["W_signflip_Holm_p"] for r in json.loads(o.read_text())["results"]}
    v = np.array([out[s]["internvl3_8b"] for s in SEEDS])
    return {"registered_seed_20260922": out[20260922]["internvl3_8b"], "seeds": "1..40",
            "min": float(v.min()), "median": float(np.median(v)), "max": float(v.max()),
            "frac_below_0.05": float((v < 0.05).mean()), "per_seed": v.tolist()}


def paired_sweep():
    pc = json.loads((R / "outputs/medgemma4b_paired_20260924/paired_contrast.json").read_text())
    rows = pc["rows"]; by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["image_sha256"]].append(i)
    idx = [np.array(by[g]) for g in sorted(by)]
    d = np.array([r["d"] for r in rows], float)
    s = np.array([d[ix].sum() for ix in idx]); cnt = np.array([len(ix) for ix in idx], float)
    B = pc["bootstrap_replicates"]
    def lo(seed):
        di = np.random.default_rng(seed).integers(0, len(idx), size=(B, len(idx)))
        return float(np.percentile(s[di].sum(1) / cnt[di].sum(1), 2.5))
    reg = lo(20260924)
    if abs(reg - pc["primary"]["ci95"][0]) > 1e-12:
        sys.exit("could not reproduce the stored CI lower bound at the registered seed")
    v = np.array([lo(x) for x in SEEDS])
    return {"registered_seed_20260924": reg, "seeds": "1..40", "B": B, "min": float(v.min()),
            "median": float(np.median(v)), "max": float(v.max()),
            "frac_le_0": float((v <= 0).mean()), "per_seed": v.tolist()}


if __name__ == "__main__":
    out = {"internvl3_holm_p": holm_sweep(), "mg4b_paired_dW_ci_lo": paired_sweep()}
    d = R / "results/seed_sensitivity_20261004"; d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out.items():
        print(k, {x: (round(y, 4) if isinstance(y, float) else y) for x, y in v.items() if x != "per_seed"})
