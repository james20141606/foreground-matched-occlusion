"""Deviation repair (2026-09-24): the registered sign-flip seed is 20260924, but
medgemma4b_legacy_vs_matched_20260924.py drew its sign vectors with SEED+1 = 20260925.
This recomputes the same statistic, with the same code path (per-image sums of
d_i = w^L_i - w^F_i, QA-weighted mean, B = 20000, two-sided, p = (1+#)/(1+B)),
at the registered seed. It also records the descriptive pool/control split of the
historical legacy rate (legacy strict rate on all 400 vs on the 229 paired items).
Reads paired_contrast.json and the legacy receipt only."""
import hashlib, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
root = Path(sys.argv[1]); run = root / "outputs/medgemma4b_paired_20260924"
src = run / "paired_contrast.json"; leg = root / "results/cipe_scale/bbox_val_medgemma4b_n400.json"
pc = json.loads(src.read_text()); rows = pc["rows"]; n = len(rows); B = 20000
by = defaultdict(list)
for i, r in enumerate(rows):
    by[r["image_sha256"]].append(i)
d = np.array([r["d"] for r in rows]); s = np.array([d[by[g]].sum() for g in sorted(by)])
t_obs = s.sum() / n
def p_at(seed):
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(B, len(s)))
    t = (signs * s).sum(1) / n
    return float((1 + np.sum(np.abs(t) >= abs(t_obs) - 1e-15)) / (1 + B))
p_reg, p_script = p_at(20260924), p_at(20260925)
if abs(p_script - pc["primary"]["signflip_p_two_sided"]) > 1e-12:
    raise RuntimeError("could not reproduce the script's sign-flip p")
L = json.loads(leg.read_text()); lby = {r["sample_id"]: r for r in L["rows"]}
all400 = float(np.mean([r["cipe_relevant"] < r["cipe_random"] for r in L["rows"]]))
on229 = float(np.mean([r["wL"] for r in rows]))
lo, hi = pc["primary"]["ci95"]
out = {"deviation": "sign-flip seed 20260925 used by the analysis script; registered seed is 20260924",
       "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
       "legacy_receipt_sha256": hashlib.sha256(leg.read_bytes()).hexdigest(),
       "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       "B": B, "signflip_p_registered_seed_20260924": p_reg,
       "signflip_p_script_seed_20260925": p_script,
       "verdict_at_registered_seed": "A" if (lo > 0 and p_reg < 0.05) else "not A",
       "descriptive_split": {"legacy_strict_rate_all400": all400, "legacy_strict_rate_229": on229,
                             "matched_W_F_229": pc["estimates"]["W_F"]["estimate"],
                             "item_pool_step": on229 - all400,
                             "control_step": pc["estimates"]["W_F"]["estimate"] - on229}}
(run / "signflip_registered_seed.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
