#!/usr/bin/env python3
"""Score the disclosure-clean region experiment against PREREG_exposure_clean_region.md.
Committed BEFORE any VinDr-CXR / ChestX-Det receipt exists; amended twice, both
amendments non-blind and disclosed in the registration.

REFUSES PARTIAL INPUT: needs the specialist receipt AND all four general receipts, each
with the positive control present, per-arm quantities present, n >= 120 items, and the
corrected-TV readout. Prints the branch verbatim.

  E1'' SPECIALIST SIGNAL SURVIVES  medgemma passes the ingestion diagnostic, its YES-arm
       q<0.05, and its YES-arm W exceeds every general's YES-arm W (no exclusion).
  E2'' RANKING DISSOLVES           passes but YES arm not significant / not on top.
  E3'' INSTRUMENT FAIL/UNDERPOWERED medgemma fails the diagnostic, or the primary arm or
       combined roster floors < 120.

AMENDMENT 1 (non-blind, disclosed): the colour control's premise fails on
finding-presence questions; ingestion is diagnosed by within-question SD of p(yes)
across real images, computed from the foreground receipts. The colour flag is still
printed. The original reader's 0.5 bar was a transcription error against the
instrument's committed 0.02 (image_injection_ok).

AMENDMENT 2 (non-blind, disclosed): (a) the NO arm's construction does not match the
registered estimand (it occludes ALL other findings' boxes, 2.73/item vs the yes arm's
1.43), so arms are reported separately -- YES primary, NO a labelled secondary measuring
cross-finding context sensitivity -- and the verdict is taken from the YES arm.
(b) The SD bar is a post-hoc DIAGNOSTIC LABEL, not an exclusion: the ingestion SD is
printed continuously for all five models, a model below the bar is described as
image-insensitive under this yes/no readout and its W stays in the table, and no
exclusion enters the ordering (the specialist must exceed every general regardless).
The bar still gates the SPECIALIST (E3''): an image-insensitive specialist has no
interpretable W. Note the mechanics: the readout scores strict wins, so a perfectly
image-invariant model gives W=0 (all ties), while a low-but-noisy image effect
approaches the coin-flip 0.5.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "results/exposure_clean"
SPECIALIST = "medgemma4b"
GENERALS = ["qwen2vl_7b", "qwen25vl_7b", "gemma3_4b", "internvl3_8b"]
Q_BAR = 0.05
N_FLOOR = 120
SD_BAR = 0.05          # AMENDMENT 1 diagnostic; AMENDMENT 2 demotes it to a label
ARMS = ("yes", "no")
FG_DIR_NAME = "bbox_fg_{tag}_chestxdet.json"
INSENSITIVE = "image-insensitive under this yes/no readout"


def ingestion_sd(d: Path, tag: str):
    """Within-question mean SD of p(yes) across real images, from the fg receipt."""
    import numpy as np
    from collections import defaultdict
    fp = d / FG_DIR_NAME.format(tag=tag)
    if not fp.exists():
        print(f"REFUSED: missing foreground receipt {fp} for the ingestion diagnostic")
        sys.exit(2)
    rows = json.loads(fp.read_text())["rows"]
    by_q = defaultdict(list)
    for r in rows:
        p = np.asarray(r["p_original"], float)
        p = p / p.sum() if p.sum() > 0 else p
        by_q[r["sample_id"].split("_")[-1]].append(float(p[0]))
    return float(np.mean([np.std(v) for v in by_q.values()]))


def load(d: Path, tag: str):
    p = d / f"expclean_{tag}.json"
    if not p.exists():
        print(f"REFUSED: missing receipt {p}")
        sys.exit(2)
    r = json.loads(p.read_text())
    for k in ("W", "q", "n_items", "positive_control_flip", "readout", "arms"):
        if k not in r:
            print(f"REFUSED: {p.name} lacks field {k!r}; keys: {sorted(r)[:8]}")
            sys.exit(2)
    for a in ARMS:
        if a not in r["arms"] or any(k not in r["arms"][a] for k in ("W", "q", "n")):
            print(f"REFUSED: {p.name} lacks per-arm quantities for {a!r} "
                  "(AMENDMENT 2 requires W/q/n per arm)")
            sys.exit(2)
    if r["readout"] != "corrected_tv":
        print(f"REFUSED: {p.name} readout is {r['readout']!r}, not the registered corrected_tv")
        sys.exit(2)
    return r


def main(d=DEFAULT_DIR):
    d = Path(d)
    rs = {t: load(d, t) for t in [SPECIALIST] + GENERALS}
    sds = {t: ingestion_sd(d, t) for t in rs}
    print(f"candidate set seen: {len(rs)} of {1 + len(GENERALS)} registered models "
          f"({SPECIALIST} + {len(GENERALS)} generals)")
    print("arms per AMENDMENT 2: YES primary (matches the registered estimand); "
          "NO is a labelled secondary (cross-finding context sensitivity)")
    for t, r in rs.items():
        y, n_ = r["arms"]["yes"], r["arms"]["no"]
        print(f"  {t:14s} YES W={y['W']:.3f} q={y['q']:.4f} n={y['n']} | "
              f"NO W={n_['W']:.3f} q={n_['q']:.4f} n={n_['n']} | "
              f"comb W={r['W']:.3f} q={r['q']:.4f} n={r['n_items']} | "
              f"colour-ctl={r['positive_control_flip']:.3f} "
              f"ingestion-SD={sds[t]:.3f}"
              f"{'' if sds[t] >= SD_BAR else '  << ' + INSENSITIVE}")

    floored = sorted({t for t, r in rs.items()
                      if r["n_items"] < N_FLOOR or r["arms"]["yes"]["n"] < N_FLOOR})
    if sds[SPECIALIST] < SD_BAR or floored:
        why = []
        if sds[SPECIALIST] < SD_BAR:
            why.append(f"specialist ingestion-SD {sds[SPECIALIST]:.3f} < {SD_BAR}")
        if floored:
            why.append(f"primary arm or combined n < {N_FLOOR} on {floored}")
        print(f"\n>>> E3'' INSTRUMENT FAIL / UNDERPOWERED: {'; '.join(why)}. "
              "Reported as such; no number is promoted.")
        return 0

    flagged = [g for g in GENERALS if sds[g] < SD_BAR]
    s = rs[SPECIALIST]["arms"]["yes"]
    beats_all = all(s["W"] > rs[g]["arms"]["yes"]["W"] for g in GENERALS)
    if s["q"] < Q_BAR and beats_all:
        print(f"\n>>> E1'' SPECIALIST SIGNAL SURVIVES (AMENDMENTS 1+2, non-blind, "
              f"disclosed). On the primary YES arm MedGemma-4B W={s['W']:.3f} "
              f"(q={s['q']:.4f}, n={s['n']}) exceeds every general "
              f"({', '.join('%s %.3f' % (g, rs[g]['arms']['yes']['W']) for g in GENERALS)}) "
              f"on disclosure-clean items; combined W={rs[SPECIALIST]['W']:.3f} "
              f"(q={rs[SPECIALIST]['q']:.4f}, n={rs[SPECIALIST]['n_items']}); the NO arm "
              "is reported only as the labelled cross-finding secondary. "
              f"{INSENSITIVE.capitalize()}: "
              f"{[(g, round(sds[g], 3)) for g in flagged] or 'none'} -- W reported, not "
              "excluded; the specialist exceeds every general regardless, so the ordering "
              "needs no exclusion. The paper carries the non-blindness sentence for both "
              "amendments, all five ingestion SDs, and the 'disclosed training data' "
              "scope sentence. The SLAKE withdrawal stands regardless.")
    else:
        why = ("YES arm not significant" if s["q"] >= Q_BAR else
               "YES arm does not exceed every general")
        print(f"\n>>> E2'' RANKING DISSOLVES ({why}). The exposure explanation is "
              "confirmed as sufficient for the original headline; the specialist "
              "advantage did not survive its first disclosure-clean test.")
    return 0


def selftest():
    import tempfile

    def arms(yes_w, yes_q, no_w=0.55, no_q=0.5, n_yes=160, n_no=140):
        return {"yes": {"W": yes_w, "q": yes_q, "n": n_yes},
                "no": {"W": no_w, "q": no_q, "n": n_no}}

    base = {"readout": "corrected_tv", "n_items": 300, "positive_control_flip": 0.8}

    def fg_rows(sd_high):
        """Synthetic fg receipt rows: two questions; p(yes) either varies across
        images (sd_high -> SD ~0.5) or is constant (SD 0)."""
        rows = []
        for q in ("Effusion", "Mass"):
            for i in range(10):
                py = (1.0 if i % 2 else 0.0) if sd_high else 0.7
                rows.append({"sample_id": f"cxdet_yes_{i}_{q}",
                             "p_original": [py, 1.0 - py]})
        return rows

    def write(d, spec_w, spec_q, gen_w, sd_low=(), **over):
        for t in [SPECIALIST] + GENERALS:
            w = spec_w if t == SPECIALIST else gen_w
            r = dict(base, W=w, q=(spec_q if t == SPECIALIST else 0.5),
                     arms=arms(w, spec_q if t == SPECIALIST else 0.5))
            r.update(over.get(t, {}))
            (Path(d) / f"expclean_{t}.json").write_text(json.dumps(r))
            (Path(d) / FG_DIR_NAME.format(tag=t)).write_text(
                json.dumps({"rows": fg_rows(sd_high=t not in sd_low)}))

    import io
    from contextlib import redirect_stdout

    def run(**kw):
        with tempfile.TemporaryDirectory() as td:
            write(td, **kw)
            buf = io.StringIO()
            code = 0
            try:
                with redirect_stdout(buf):
                    main(td)
            except SystemExit as e:
                code = e.code
            return code, buf.getvalue()

    c, o = run(spec_w=0.65, spec_q=0.001, gen_w=0.55)
    assert c == 0 and "E1''" in o, o
    c, o = run(spec_w=0.55, spec_q=0.001, gen_w=0.60)
    assert c == 0 and "E2''" in o and "does not exceed" in o, o
    c, o = run(spec_w=0.65, spec_q=0.20, gen_w=0.55)
    assert c == 0 and "E2''" in o and "not significant" in o, o
    # the verdict reads the YES arm: combined W high but YES arm behind -> E2''
    c, o = run(spec_w=0.65, spec_q=0.001, gen_w=0.55,
               medgemma4b={"arms": arms(0.54, 0.001)})
    assert c == 0 and "E2''" in o and "does not exceed" in o, o
    # E3'': the SPECIALIST fails the ingestion diagnostic
    c, o = run(spec_w=0.65, spec_q=0.001, gen_w=0.55, sd_low=(SPECIALIST,))
    assert c == 0 and "E3''" in o, o
    # E3'': combined floor, and the primary-arm floor (AMENDMENT 2)
    c, o = run(spec_w=0.65, spec_q=0.001, gen_w=0.55, gemma3_4b={"n_items": 80})
    assert c == 0 and "E3''" in o, o
    c, o = run(spec_w=0.65, spec_q=0.001, gen_w=0.55,
               gemma3_4b={"arms": arms(0.55, 0.5, n_yes=80)})
    assert c == 0 and "E3''" in o, o
    # a general below the SD bar is LABELLED, not excluded; E1'' fires and its W stays
    c, o = run(spec_w=0.65, spec_q=0.001, gen_w=0.55, sd_low=("qwen2vl_7b",))
    assert c == 0 and "E1''" in o and INSENSITIVE in o, o
    assert "EXCLUDED" not in o and "qwen2vl_7b 0.550" in o, o
    # refusal on missing receipt
    with tempfile.TemporaryDirectory() as td:
        write(td, spec_w=0.65, spec_q=0.001, gen_w=0.55)
        (Path(td) / f"expclean_{GENERALS[0]}.json").unlink()
        buf = io.StringIO()
        code = 0
        try:
            with redirect_stdout(buf):
                main(td)
        except SystemExit as e:
            code = e.code
        assert code == 2 and "REFUSED" in buf.getvalue()
    # refusal on a receipt lacking the per-arm quantities (AMENDMENT 2 schema):
    # both a missing "arms" field and a missing per-arm subfield must refuse
    for mutate, needle in ((lambda r: r.pop("arms"), "lacks field 'arms'"),
                           (lambda r: r["arms"]["yes"].pop("n"), "per-arm")):
        with tempfile.TemporaryDirectory() as td:
            write(td, spec_w=0.65, spec_q=0.001, gen_w=0.55)
            fp = Path(td) / f"expclean_{SPECIALIST}.json"
            r = json.loads(fp.read_text())
            mutate(r)
            fp.write_text(json.dumps(r))
            buf = io.StringIO()
            code = 0
            try:
                with redirect_stdout(buf):
                    main(td)
            except SystemExit as e:
                code = e.code
            assert code == 2 and needle in buf.getvalue(), buf.getvalue()
    print("selftest: E1'' (incl. flagged-not-excluded), E2'' (three routes), "
          "E3'' (three routes), and both refusals all fire")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR))
