"""POST-HOC descriptive (not registered): how much of the MedGemma-4B paired dW sits on
items where every CIPE value (annotation, legacy draw, five matched draws) agrees within eps.
Reads outputs/medgemma4b_paired_20260924/paired_contrast.json only. No test, no decision."""
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]); src = root / "outputs/medgemma4b_paired_20260924/paired_contrast.json"
rows = json.loads(src.read_text())["rows"]; n = len(rows); out = {"post_hoc": True, "n_qa": n,
    "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "by_eps": {}}
for eps in (1e-6, 1e-4):
    sat = [r for r in rows if abs(r["cipe_annotation"] - r["cipe_legacy_control"]) < eps
           and all(abs(r["cipe_annotation"] - c) < eps for c in r["cipe_matched_draws"])]
    rest = [r for r in rows if r not in sat]
    out["by_eps"][str(eps)] = {"n_saturated": len(sat),
        "contribution_to_dW_strict": sum(r["d"] for r in sat) / n,
        "n_rest": len(rest), "dW_strict_on_rest": sum(r["d"] for r in rest) / len(rest)}
(root / "outputs/medgemma4b_paired_20260924/saturation_descriptive.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
