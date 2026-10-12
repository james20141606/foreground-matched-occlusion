# PRE-REGISTRATION: region-instrument positive control, robustness grid, and coverage isolation

Written 2026-09-26 ~03:20 +08:00. This file is committed **before** any analysis below is run
and before any GPU inference exists. After results exist it is not edited; any deviation is
logged as a dated "Deviation" entry in `experiments_detail.md` and in the result JSON.

Origin: `6_paper_work/final_review_20260924/ai_review_sim/medtrust_cc.md`, weaknesses #4
(no positive control with known dependence), #5 (black fill not neutral), #39/A-W12
(tolerance 0.15 / threshold 5% sensitivity untested), and the paper's own statement that
the coverage replay "does not say how much of MedGemma-4B's paired ΔW=+0.054 came from
coverage, since the placement rule changed too". Brief:
`6_paper_work/final_review_20260924/STRENGTHEN_BRIEF.md`.

## What is already known (not fully blind)

- Published (Table 1, image-cluster): MedGemma-4B W=0.612, W*≈0.64; D=+0.051. General models
  weaker; ties 4–9%. The robustness configurations below have never been run; no marker image
  has ever been built; no coverage-gap placement exists. The direction predicted for the
  coverage-gap arm (Proposition 1(b)) is "inflates W"; the rules are written to report either
  direction.

## Common definitions

- Items: the 400 SLAKE candidates of `results/cipe_scale/bbox_fg_<tag>_n400.json`
  (loader `load_slake_with_bbox(400, seed=42)`), identity-checked against the receipt by
  `sample_id` + `source_key` before any forward.
- Readout: the unchanged `LOGPROB_DISPATCH` path of the receipts (`probs()`), vectors
  **renormalised** before TV (as `region_tv_corrected.py`). Δ = TV(p_o, p_cf) = |Δp(yes)|.
- W, W*, D exactly as Eqs. (2)–(3) of the paper, K = number of draws of the arm.
- All arms of one model come from **one run** (the in-run black/matched arm is the reference,
  not the stored receipt). Receipt reproduction is reported as a pipeline check (I3).
- Inference: image-cluster percentile bootstrap (source image sha256 from
  `outputs/image_cluster_20260922/image_mapping.json`), 95%, B = 20,000, seed 20260926,
  QA-weighted. Paired contrasts: same bootstrap on per-item differences, plus image-cluster
  sign-flip test (B = 20,000, seed 20260926, two-sided).
- Models: **MedGemma-4B (`google/medgemma-4b-it`) is primary; every decision below is taken on
  it.** Gemma3-4B-it (its base) and Qwen2.5-VL-7B-Instruct run the identical script and are
  reported with the same rules applied per model, labelled secondary. No model is added or
  dropped after results.

## Pipeline checks (run before reading any outcome; a failure stops the analysis)

- I1 identity: all 400 candidates match the receipt's `sample_id`, `source_key`, `image_ref`.
- I2 placement: re-placing the matched control with `random.Random(42)`, tol 0.15, thresh 0.05,
  over the 400 candidates in order reproduces all 229×5 stored box sets and the 229/171 split.
- I3 forward reproduction (reported, not gating): max and 95th-percentile |p_yes| difference
  between in-run and stored black-arm vectors; W from in-run vectors vs published W.
  If |W_inrun − W_receipt| > 0.02 for MedGemma-4B the run is labelled "non-reproducing" in every
  sentence that uses it.
- I4 (MedGemma-4B only): CIPE computed on the replayed legacy boxes reproduces the stored legacy
  `legacy_cipe_random` (95th-pct abs diff ≤ max(1e-3, 2 × the organ-arm CIPE diff)). This is the
  model-forward check the legacy replay appendix says was not run. If I4 fails, analysis 3a is
  reported as "replay not confirmed" and no legacy sentence changes.

## Analysis 1 — positive control with a constructed, known region dependence

Construction (per admitted item, black fill, the item's stored organ and 5 control box sets):
a filled pure-red disc (RGB 255,0,0), alpha-blended at opacity α, centred in the largest
annotated box, diameter = max(4 px, 0.5 × the box's shorter side). The disc lies wholly inside
an annotated box, so the annotated-union occlusion removes it entirely and no control box
(never overlapping an annotated box) touches it. Question: "Is there a red circle in this
image?" (+ the pipeline's standard "Answer with yes or no."). Doses α ∈ {0 (no disc), 0.15,
0.4, 1.0}.

Estimands per α: W_α, W*_α, D_α on the marker question (Δ from the marked image); marker effect
m_α = mean_i TV(p(marked), p(unmarked)) on the marker question; tie fraction.

Decision (MedGemma-4B, α = 1.0):
- **P-PASS**: m_1.0 ≥ 0.10 and the W*_1.0 image-cluster CI lies above 0.5.
- **P-FAIL**: m_1.0 ≥ 0.10 and the CI reaches 0.5 or below (the instrument misses a dependence the
  model demonstrably has).
- **P-UNINFORMATIVE**: m_1.0 < 0.10 (the model does not register the disc; the construct has no
  dependence to detect).
Secondary, descriptive, no decision: W*_α and m_α for all α (power curve); the smallest α whose
W* CI excludes 0.5; W* pooled over α ∈ {0.15, 0.4, 1.0} in per-item m bins
[0,0.02), [0.02,0.05), [0.05,0.10), [0.10,0.20), [0.20,1] with counts; α = 0 is the same-question
no-dependence reference.

Paper text (English; numbers in <> from the receipt only):
- P-PASS: "A positive control with a constructed region dependence (a red disc inside the
  largest annotated box, question 'Is there a red circle?') gives W* = <w> [<lo>,<hi>] for
  MedGemma-4B at full opacity (disc effect <m> in p(yes)), and falls to <w0> without the disc
  (Appendix X); <secondary models sentence>." Limitations sentence "No item with a known
  annotation effect was constructed, so the instrument's power for small τ is unmeasured"
  becomes "Its power was measured only on a synthetic marker (Appendix X): W* exceeded 0.5 once
  the marker moved p(yes) by about <m_min>; effects smaller than that are not resolved."
- P-FAIL: "A constructed positive control (…) was not detected: W* = <w> [<lo>,<hi>] although the
  disc moved p(yes) by <m>. The region instrument is therefore not shown to detect a known
  region dependence, and weak or null W should not be read as weak reliance." Limitations
  sentence rewritten to say the instrument failed its constructed positive control.
- P-UNINFORMATIVE: "A constructed marker control was uninformative for MedGemma-4B (the disc
  moved p(yes) by only <m>), so instrument power remains unmeasured." Limitations unchanged in
  substance.

## Analysis 2 — robustness grid (one factor at a time)

Configurations, all 5 draws, black unless stated:
- F-mean: fill = the image's mean RGB (integer), stored receipt boxes.
- F-blur: fill = the image Gaussian-blurred (PIL GaussianBlur, radius = max(2, round(0.04 ×
  min(W,H)))) pasted into the same pixel footprint as the black fill, stored receipt boxes.
- T-0.05, T-0.10, T-0.25: tolerance changed, threshold 5%, re-placed with Random(42) over the 400.
- H-0.02, H-0.10: threshold changed (2%, 10% of dynamic range), tolerance 0.15, re-placed likewise.
- Reference: in-run black, stored boxes (= T-0.15/H-0.05).
Items are those admitted under each configuration (n reported); fills use the same 229.

Estimands per config: n, mean φ annotated/control (under that config's threshold), W, W*, D,
tie fraction, image-cluster CIs; for fills also the paired ΔW* (fill − black) with CI.

Decision (MedGemma-4B, W*):
- **R-ROBUST**: every configuration with n ≥ 100 has its W* CI above 0.5.
- **R-PARTIAL**: at least one such configuration has CI reaching 0.5 while W* > 0.5 (each named).
- **R-SENSITIVE**: at least one such configuration has W* ≤ 0.5 (each named).
Configurations with n < 100 are reported, labelled underpowered, and do not enter the decision.
General models: same table, descriptive.

Paper text:
- R-ROBUST: §3 "sensitivity to them, and to the black fill itself, is untested" becomes "varying
  the tolerance (0.05–0.25), the threshold (2–10%) and the fill (mean, blur) leaves MedGemma-4B's
  W* above one half in every setting (<range>; Appendix Y)".
- R-PARTIAL: "…MedGemma-4B's W* stays above one half in <k> of <K> settings; under <named> its
  interval reaches one half (Appendix Y)".
- R-SENSITIVE: "…under <named> MedGemma-4B's W* is <w>, at or below one half, so the result
  depends on <named choice> (Appendix Y)"; the same qualifier is added to the Limitations item on
  the foreground argument.

Receipt-only descriptive additions (no GPU; may be run before the GPU job, after this commit):
(i) tolerance tightening on stored draws: W/W* over draws whose max per-box coverage gap is
≤ 0.05 / ≤ 0.10 (items with ≥ 1 surviving draw); (ii) placebo: draw k as pseudo-annotated
against the other four, k = 1..5, all six models, image-cluster CI — expected 0.5 by
exchangeability; flagged if ≥ 6 of the 30 CIs exclude 0.5. Neither carries a decision.

## Analysis 3 — isolating what control coverage does to W

3a (MedGemma-4B, all models descriptive): the replayed legacy boxes
(`outputs/legacy_coverage_replay_20260925/coverage.json`, 229 admitted items, one draw, black)
run through the same readout. Report W_L, W*_L (TV) and paired ΔW* = W*_L − W*_matched with CI
and sign-flip p. Descriptive; it still mixes placement rule and coverage.

3b (primary): coverage-gap arm G(δ): the matched placement rule (same size, no overlap with any
annotated or already placed control box, rejection sampling, 500 tries, 5 draws, drop items with
< 5 admissible draws) except the coverage target, |φ(c) − (φ(b) − δ)| ≤ 0.075.
δ ∈ {0.3, 0.6}; seed Random(42) over the 400 in order; threshold 5%.
Primary estimand: paired ΔW*(0.3) = W*(G0.3) − W*(matched) on the items admitted to both, image-
cluster CI and sign-flip test. δ = 0.3 is chosen because the legacy-vs-matched mean gap was 0.284.
Secondary: δ = 0.6; ΔW (strict); ΔD.

Decision (MedGemma-4B, ΔW*(0.3)):
- **C-INFLATES**: CI above 0 and sign-flip p < 0.05.
- **C-NULL**: CI contains 0 (or CI and test disagree; stated).
- **C-DEFLATES**: CI below 0 and p < 0.05.
If fewer than 60 items admit G0.3, the primary is reported as underpowered and the decision is
taken as C-NULL-underpowered.

Paper text:
- C-INFLATES: after the replay sentence of §3.1: "Lowering only the control's coverage by 0.3,
  with placement otherwise unchanged, raises MedGemma-4B's W* by <d> [<lo>,<hi>] on <n> items
  (Appendix Z), the inflation Proposition 1(b) predicts." The appendix sentence "It does not say
  how much of MedGemma-4B's paired ΔW came from coverage" becomes "A coverage-gap arm isolates
  coverage: … (<d>); the legacy arm's ΔW* on the same readout is <dL>."
- C-NULL: "Lowering only the control's coverage by 0.3 changes MedGemma-4B's W* by <d>
  [<lo>,<hi>] (Appendix Z): Proposition 1(b)'s inflation is not detected on this model at this
  dose, so the foreground argument remains a model-based caution rather than a measured bias."
- C-DEFLATES: same as C-NULL but "lowers", plus "contrary to the monotone-dose assumption of
  Proposition 1"; added to the Limitations item on the foreground argument.

## Scope written into every outcome

One dataset (SLAKE, 229 of 400 items), one specialist family, image-level boxes; the marker is
synthetic and tests the instrument, not clinical grounding; the coverage-gap arm changes
coverage together with where boxes can go (low-coverage boxes sit near the body edge), so it
isolates coverage from the legacy *rule*, not from position; all analyses are post hoc to the
review.

## Receipts

Scripts `code/scripts/region_instrument_20260926.py` (GPU run) and
`code/scripts/region_instrument_analysis_20260926.py` (estimands, CPU);
sbatch `code/scripts/slurm/run_region_instrument_20260926.sbatch`; outputs in
`results/region_instrument_20260926/` with input SHA-256s, script SHA-256, JIDs, and per-item
probability vectors for every arm.
