"""Estimands and registered verdicts for notes/PREREG_region_instrument_20260926.md (CPU only).

Modes
  --receipt_only : placebo (draw k vs the other four) and tolerance tightening on the stored
                   bbox_fg_*_n400.json receipts (descriptive, no decision).
  --tags ...     : analyses 1-3 on rows_<tag>.jsonl / meta_<tag>.json written by
                   region_instrument_20260926.py.

TV on renormalised vectors (as region_tv_corrected.py); image-cluster percentile bootstrap
(image sha256 from the image mapping), QA-weighted, B=20000, seed 20260926; paired contrasts
also get an image-cluster sign-flip test.
"""
from __future__ import annotations
import argparse, glob, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
MAPPING = ROOT / "outputs/image_cluster_20260922/image_mapping.json"
TVREF = ROOT / "outputs/region_tv_corrected.json"
B = 20000
SEED = 20260926
PRIMARY = "medgemma4b"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def norm(v):
    a = np.asarray(v, float); return a / a.sum()


def tv(p, q):
    return 0.5 * float(np.abs(norm(p) - norm(q)).sum())


IMG = {r["sample_id"]: r["image_sha256"] for r in json.loads(MAPPING.read_text())["items"]}


def cluster_stats(ids, vals, seed=SEED):
    """QA-weighted mean with image-cluster percentile CI."""
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return {"n": 0}
    imgs = sorted({IMG[i] for i in ids}); ix = {g: j for j, g in enumerate(imgs)}
    g = np.array([ix[IMG[i]] for i in ids])
    sums = np.bincount(g, weights=vals, minlength=len(imgs)); cnt = np.bincount(g, minlength=len(imgs))
    dr = np.random.default_rng(seed).integers(0, len(imgs), (B, len(imgs)))
    bs = sums[dr].sum(1) / cnt[dr].sum(1)
    return {"n": int(len(vals)), "n_images": len(imgs), "mean": float(vals.mean()),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def signflip(ids, d, seed=SEED):
    d = np.asarray(d, float)
    imgs = sorted({IMG[i] for i in ids}); ix = {g: j for j, g in enumerate(imgs)}
    g = np.array([ix[IMG[i]] for i in ids])
    S = np.bincount(g, weights=d, minlength=len(imgs))
    sg = np.random.default_rng(seed + 1).choice([-1.0, 1.0], (B, len(imgs)))
    T = (sg * S[None, :]).sum(1) / len(d); obs = d.mean()   # no BLAS (spurious Accelerate warnings)
    return float((np.sum(np.abs(T) >= abs(obs) - 1e-12) + 1) / (B + 1))


def item_scores(p_o, p_A, p_Cs):
    dA = tv(p_o, p_A); dC = np.array([tv(p_o, c) for c in p_Cs])
    w = float(np.mean(dA > dC)); t = float(np.mean(dA == dC))
    return {"W": w, "Ws": w + 0.5 * t, "D": float(dA - dC.mean()), "tie": t}


def arm_summary(ids, sc):
    out = {}
    for k in ("W", "Ws", "D", "tie"):
        out[k] = cluster_stats(ids, [s[k] for s in sc])
    return out


def paired(ids, a, b, key="Ws"):
    d = [x[key] - y[key] for x, y in zip(a, b)]
    s = cluster_stats(ids, d); s["signflip_p"] = signflip(ids, d); return s


# ------------------------------------------------------------------ receipt-only
def receipt_only(out_dir):
    res = {"status": "receipt-only descriptive (PREREG analysis 2 additions); no decision",
           "seed": SEED, "B": B, "receipts": {}, "placebo": {}, "tolerance_tightening": {}}
    n_excl = 0
    for fp in sorted(glob.glob(str(ROOT / "results/cipe_scale/bbox_fg_*_n400.json"))):
        tag = Path(fp).name.replace("bbox_fg_", "").replace("_n400.json", "")
        rec = json.loads(Path(fp).read_text()); res["receipts"][tag] = sha(fp)
        rows = sorted(rec["rows"], key=lambda r: int(r["sample_id"].split("_")[1]))
        ids = [r["sample_id"] for r in rows]
        # placebo: draw k as pseudo-annotated vs the other four draws
        pl = {}
        for k in range(5):
            sc = [item_scores(r["p_original"], r["p_control_draws"][k],
                              [c for j, c in enumerate(r["p_control_draws"]) if j != k]) for r in rows]
            st = cluster_stats(ids, [s["Ws"] for s in sc])
            st["excludes_half"] = not (st["ci95"][0] <= 0.5 <= st["ci95"][1]); n_excl += st["excludes_half"]
            st["W_strict"] = float(np.mean([s["W"] for s in sc]))
            pl[f"draw{k+1}"] = st
        res["placebo"][tag] = pl
        # tolerance tightening on stored draws
        tt = {}
        for t in (0.05, 0.10, 0.15):
            kid, ksc, nd = [], [], 0
            for r in rows:
                keep = [k for k in range(5)
                        if max(abs(c - o) for c, o in zip(r["control_fg_coverage"][k], r["organ_fg_coverage"])) <= t + 1e-12]
                if not keep:
                    continue
                kid.append(r["sample_id"]); nd += len(keep)
                ksc.append(item_scores(r["p_original"], r["p_organ"], [r["p_control_draws"][k] for k in keep]))
            s = arm_summary(kid, ksc); s["n_draws_kept"] = nd; tt[f"max_gap<={t}"] = s
        res["tolerance_tightening"][tag] = tt
    res["placebo_n_ci_excluding_half"] = n_excl
    res["placebo_flag"] = n_excl >= 6
    res["generator_sha256"] = sha(__file__); res["mapping_sha256"] = sha(MAPPING)
    out = Path(out_dir) / "receipt_only.json"; out.write_text(json.dumps(res, indent=2))
    print(f"[saved] {out}")
    for tag in res["placebo"]:
        p = res["placebo"][tag]; t = res["tolerance_tightening"][tag]
        print(tag, "placebo W*:", " ".join(f"{p[k]['mean']:.3f}[{p[k]['ci95'][0]:.3f},{p[k]['ci95'][1]:.3f}]" for k in p))
        print("   tighten:", " ".join(f"{k}: n={v['Ws']['n']} W*={v['Ws']['mean']:.3f}[{v['Ws']['ci95'][0]:.3f},{v['Ws']['ci95'][1]:.3f}]" for k, v in t.items()))
    print("placebo CIs excluding 0.5:", n_excl, "of", 5 * len(res["placebo"]), "flag:", res["placebo_flag"])


# ------------------------------------------------------------------ GPU-run analyses
def analyse_tag(out_dir, tag):
    from medtrust.cipe import cipe_score
    rows = [json.loads(l) for l in (Path(out_dir) / f"rows_{tag}.jsonl").read_text().splitlines() if l.strip()]
    meta = json.loads((Path(out_dir) / f"meta_{tag}.json").read_text())
    rfp = ROOT / "results/cipe_scale" / Path(meta["receipt"]).name     # local copy of the torch receipt
    assert sha(rfp) == meta["receipt_sha256"], "local receipt differs from the one the run read"
    rec = json.loads(rfp.read_text())
    stored = {r["sample_id"]: r for r in rec["rows"]}
    rows = sorted({r["sample_id"]: r for r in rows}.values(), key=lambda r: int(r["sample_id"].split("_")[1]))
    M = [r for r in rows if r.get("stored_boxes")]
    ids = [r["sample_id"] for r in M]
    out = {"tag": tag, "model_path": meta["model_path"], "meta": meta,
           "generator_sha256": sha(__file__), "rows_sha256": sha(Path(out_dir) / f"rows_{tag}.jsonl")}
    if meta.get("smoke"):
        stored = {i: stored[i] for i in ids}
    else:
        assert len(M) == len(stored) == 229, (len(M), len(stored))

    # I3 forward reproduction
    diffs = []
    for r in M:
        s = stored[r["sample_id"]]
        pairs = [(r["p_o"], s["p_original"]), (r["p_A"]["black"], s["p_organ"])] + \
                list(zip(r["p_C"]["matched_black"], s["p_control_draws"]))
        diffs += [abs(norm(a)[0] - norm(b)[0]) for a, b in pairs]
    ref = [item_scores(r["p_o"], r["p_A"]["black"], r["p_C"]["matched_black"]) for r in M]
    rec_sc = [item_scores(stored[i]["p_original"], stored[i]["p_organ"], stored[i]["p_control_draws"]) for i in ids]
    w_in, w_rc = float(np.mean([s["W"] for s in ref])), float(np.mean([s["W"] for s in rec_sc]))
    out["I3"] = {"max_abs_dpyes": float(np.max(diffs)), "p95_abs_dpyes": float(np.percentile(diffs, 95)),
                 "W_inrun": w_in, "W_receipt": w_rc, "Ws_inrun": float(np.mean([s["Ws"] for s in ref])),
                 "Ws_receipt": float(np.mean([s["Ws"] for s in rec_sc])),
                 "non_reproducing": abs(w_in - w_rc) > 0.02}

    # ---- Analysis 1: marker positive control
    A1 = {}
    for al in ("0.0", "0.15", "0.4", "1.0"):
        sc, m = [], []
        for r in M:
            mk = r["marker"]; a = mk[f"a{al}"]
            sc.append(item_scores(a["p_o"], mk["p_A"], a["p_C"]))
            m.append(tv(a["p_o"], mk["a0.0"]["p_o"]))
        A1[al] = arm_summary(ids, sc); A1[al]["m"] = cluster_stats(ids, m)
        A1[al]["p_yes_marked_mean"] = float(np.mean([norm(r["marker"][f"a{al}"]["p_o"])[0] for r in M]))
    bins = [(0, .02), (.02, .05), (.05, .10), (.10, .20), (.20, 1.01)]
    pooled = {f"[{lo},{hi})": [] for lo, hi in bins}
    for r in M:
        mk = r["marker"]
        for al in ("0.15", "0.4", "1.0"):
            a = mk[f"a{al}"]; mm = tv(a["p_o"], mk["a0.0"]["p_o"])
            for lo, hi in bins:
                if lo <= mm < hi:
                    pooled[f"[{lo},{hi})"].append((r["sample_id"], item_scores(a["p_o"], mk["p_A"], a["p_C"])["Ws"]))
    A1["m_bins_pooled"] = {k: (cluster_stats([x[0] for x in v], [x[1] for x in v]) if v else {"n": 0})
                           for k, v in pooled.items()}
    detect = [al for al in ("0.15", "0.4", "1.0") if A1[al]["Ws"]["ci95"][0] > 0.5]
    A1["smallest_alpha_Ws_ci_above_half"] = detect[0] if detect else None
    m1, ws1 = A1["1.0"]["m"]["mean"], A1["1.0"]["Ws"]
    A1["verdict"] = ("P-UNINFORMATIVE" if m1 < 0.10 else "P-PASS" if ws1["ci95"][0] > 0.5 else "P-FAIL")
    out["analysis1_marker"] = A1

    # ---- Analysis 2: robustness grid
    A2 = {"reference_black": arm_summary(ids, ref)}
    A2["reference_black"]["phi"] = {"annot": float(np.mean([np.mean(stored[i]["organ_fg_coverage"]) for i in ids])),
                                    "control": float(np.mean([np.mean(stored[i]["control_fg_coverage"]) for i in ids]))}
    for f in ("mean", "blur"):
        sc = [item_scores(r["p_o"], r["p_A"][f], r["p_C"][f"matched_{f}"]) for r in M]
        A2[f"F-{f}"] = arm_summary(ids, sc)
        A2[f"F-{f}"]["paired_dWs_vs_black"] = paired(ids, sc, ref)
    for name in ("T0.05", "T0.10", "T0.25", "H0.02", "H0.10"):
        sub = [r for r in rows if name in r.get("p_C", {})]
        sid = [r["sample_id"] for r in sub]
        sc = [item_scores(r["p_o"], r["p_A"]["black"], r["p_C"][name]) for r in sub]
        A2[name] = arm_summary(sid, sc)
        A2[name]["phi"] = {"annot": float(np.mean([np.mean(r["placement"][name]["organ_cov"]) for r in sub])) if sub else None,
                           "control": float(np.mean([np.mean([np.mean(c) for c in r["placement"][name]["cov"]]) for r in sub])) if sub else None}
        A2[name]["n_overlap_with_229"] = len(set(sid) & set(ids))
        A2[name]["underpowered"] = len(sub) < 100
    cfgs = ["F-mean", "F-blur", "T0.05", "T0.10", "T0.25", "H0.02", "H0.10"]
    elig = [c for c in cfgs if A2[c]["Ws"]["n"] >= 100]
    below = [c for c in elig if A2[c]["Ws"]["mean"] <= 0.5]
    reach = [c for c in elig if A2[c]["Ws"]["ci95"][0] <= 0.5 and c not in below]
    A2["eligible"] = elig
    A2["verdict"] = "R-SENSITIVE" if below else "R-PARTIAL" if reach else "R-ROBUST"
    A2["named_sensitive"] = below; A2["named_partial"] = reach
    out["analysis2_robustness"] = A2

    # ---- Analysis 3: coverage isolation
    A3 = {}
    L = [r for r in M if "p_legacy" in r]
    if L:
        lid = [r["sample_id"] for r in L]
        refL = [ref[ids.index(i)] for i in lid]
        scL = [item_scores(r["p_o"], r["p_A"]["black"], [r["p_legacy"]]) for r in L]
        A3["legacy"] = arm_summary(lid, scL)
        A3["legacy"]["paired_dWs_legacy_minus_matched"] = paired(lid, scL, refL)
        A3["legacy"]["paired_dW_legacy_minus_matched"] = paired(lid, scL, refL, key="W")
        # I4: CIPE on replayed legacy boxes vs the stored legacy value
        dl = [abs(cipe_score(r["p_o"], r["p_legacy"], sample_id=r["sample_id"], variant="legacy").cipe
                  - r["legacy_cipe_random_stored"]) for r in L if r.get("legacy_cipe_random_stored") is not None]
        do = [abs(cipe_score(r["p_o"], r["p_A"]["black"], sample_id=r["sample_id"], variant="organ").cipe
                  - stored[r["sample_id"]]["cipe_organ"]) for r in M]
        if dl:
            p95l, p95o = float(np.percentile(dl, 95)), float(np.percentile(do, 95))
            A3["I4"] = {"n": len(dl), "legacy_p95": p95l, "legacy_max": float(np.max(dl)),
                        "organ_p95": p95o, "organ_max": float(np.max(do)),
                        "pass": p95l <= max(1e-3, 2 * p95o)}
    for name in ("G0.3", "G0.6"):
        sub = [r for r in M if name in r.get("p_C", {})]
        sid = [r["sample_id"] for r in sub]
        scG = [item_scores(r["p_o"], r["p_A"]["black"], r["p_C"][name]) for r in sub]
        refG = [ref[ids.index(i)] for i in sid]
        A3[name] = arm_summary(sid, scG) if sub else {"n": 0}
        if sub:
            A3[name]["matched_same_items"] = arm_summary(sid, refG)
            A3[name]["paired_dWs"] = paired(sid, scG, refG)
            A3[name]["paired_dW"] = paired(sid, scG, refG, key="W")
            A3[name]["paired_dD"] = paired(sid, scG, refG, key="D")
            A3[name]["phi_control"] = float(np.mean([np.mean([np.mean(c) for c in r["placement"][name]["cov"]]) for r in sub]))
            A3[name]["phi_annot"] = float(np.mean([np.mean(r["placement"][name]["organ_cov"]) for r in sub]))
        A3[name]["n_admitted_all400"] = sum(1 for r in rows if name in r.get("p_C", {}))
    g = A3.get("G0.3", {})
    if g.get("n", 0) < 60:
        A3["verdict"] = "C-NULL-underpowered"
    else:
        pd = g["paired_dWs"]
        if pd["ci95"][0] > 0 and pd["signflip_p"] < 0.05:
            A3["verdict"] = "C-INFLATES"
        elif pd["ci95"][1] < 0 and pd["signflip_p"] < 0.05:
            A3["verdict"] = "C-DEFLATES"
        else:
            A3["verdict"] = "C-NULL" + ("" if (pd["ci95"][0] <= 0 <= pd["ci95"][1]) == (pd["signflip_p"] >= 0.05)
                                        else " (CI and sign-flip disagree)")
    out["analysis3_coverage"] = A3
    fp = Path(out_dir) / f"analysis_{tag}.json"; fp.write_text(json.dumps(out, indent=2))
    print(f"[saved] {fp}")
    return out


def fmt(s):
    return f"{s['mean']:.3f} [{s['ci95'][0]:.3f},{s['ci95'][1]:.3f}] (n={s['n']})" if s.get("n") else "n=0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(ROOT / "results/region_instrument_20260926"))
    ap.add_argument("--receipt_only", action="store_true")
    ap.add_argument("--tags", nargs="*", default=[])
    a = ap.parse_args()
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    if a.receipt_only:
        receipt_only(a.out_dir)
    for tag in a.tags:
        o = analyse_tag(a.out_dir, tag)
        print(f"== {tag} I3 {o['I3']}")
        A1 = o["analysis1_marker"]
        for al in ("0.0", "0.15", "0.4", "1.0"):
            print(f"  marker a={al}: W*={fmt(A1[al]['Ws'])} m={A1[al]['m']['mean']:.3f} tie={A1[al]['tie']['mean']:.3f}")
        print("  A1 verdict:", A1["verdict"], "smallest alpha:", A1["smallest_alpha_Ws_ci_above_half"])
        A2 = o["analysis2_robustness"]
        for c in ["reference_black", "F-mean", "F-blur", "T0.05", "T0.10", "T0.25", "H0.02", "H0.10"]:
            print(f"  {c:16s} W*={fmt(A2[c]['Ws'])} W={A2[c]['W']['mean']:.3f} D={A2[c]['D']['mean']:+.4f} tie={A2[c]['tie']['mean']:.3f}")
        print("  A2 verdict:", A2["verdict"], A2["named_sensitive"], A2["named_partial"])
        A3 = o["analysis3_coverage"]
        if "legacy" in A3:
            print(f"  legacy W*={fmt(A3['legacy']['Ws'])} dW*={fmt(A3['legacy']['paired_dWs_legacy_minus_matched'])} I4={A3.get('I4')}")
        for n in ("G0.3", "G0.6"):
            if A3[n].get("n"):
                print(f"  {n}: W*={fmt(A3[n]['Ws'])} dW*={fmt(A3[n]['paired_dWs'])} p={A3[n]['paired_dWs']['signflip_p']:.4f} phiC={A3[n]['phi_control']:.3f}")
        print("  A3 verdict:", A3["verdict"])


if __name__ == "__main__":
    main()
