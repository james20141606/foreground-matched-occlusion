# PRE-REGISTRATION: MedGemma-4B, legacy vs foreground-matched control on the same 229 QA

Written 2026-09-24T11:34+08:00, committed **before** the paired contrast is computed.
The commit time of this file is the evidence. After the result exists this file is not
edited; any deviation is recorded as a dated "Deviation" entry in `experiments_detail.md`
and in the result JSON.

Origin: final review 2026-09-24, reviewer A P1-1 (`6_paper_work/final_review_20260924/medtrust/A.md`);
brief `6_paper_work/final_review_20260924/FOLLOWUP_EXPERIMENTS_BRIEF.md` item 1.

## What is already known (this registration is not fully blind)

- Published aggregates, on **different denominators and estimators**: the legacy receipt's
  `frac_relevant_lower = 0.700` over all **400** legacy items (annotated vs one legacy draw,
  CIPE); the matched receipt's `frac_organ_lower = 0.585` over **229** items (annotated vs the
  *mean* of five matched draws, CIPE); the paper's TV-based `W = 0.612` (matched, 229).
- Not yet computed by this session: any legacy control value restricted to the 229 admitted
  items, and therefore any paired quantity below. The identity gate
  (`code/scripts/medgemma4b_legacy_matched_identity_20260924.py`, receipt
  `outputs/medgemma4b_paired_20260924/identity_gate.json`) reads no control value. The paper
  states the pairing was never analysed; this session cannot rule out that someone glanced
  at it earlier.
- Direction predicted by Proposition 1 (a control that can land on low-foreground regions
  inflates the annotated win rate): legacy > matched. The decision rule below is symmetric
  and does not depend on the prediction.

## Identity (verified before this registration, outcome-blind)

Identity gate PASS: all 229 admitted `sample_id`s exist in the legacy receipt; aligned by
`sample_id`, not list position (all 229 sit at different list positions); `n_boxes` agree
legacy = matched = source mapping for all 229, and legacy `n_boxes` agree with the mapping for
all 400; ordered annotation boxes of the matched receipt equal the mapping for all 229;
annotated-mask CIPE is float-identical for all 229 (49 of them sit at the clip boundary 0/1
and carry no fingerprint; 180 do). Same model path, same whole-run image sensitivity.
175 source images (129/39/6/1 QA per image).

## Readout: CIPE, not TV (forced by the receipts)

The legacy receipt stores only `cipe_relevant`, `cipe_random`, `n_boxes` per item: no
probability vectors, no control boxes. TV cannot be computed for the legacy arm and legacy
foreground coverage cannot be measured. The contrast is therefore on CIPE
(`1 - clip(KL(p_o || p_cf), 0, 1)`; `cipe.py` renormalises, so the S3 normalisation defect does
not touch it). A win for the annotation is `CIPE_A < CIPE_C` (the annotated occlusion moves the
answer distribution more). This is **not** the TV-based W of Table 1.

## Estimands (n = 229, fixed; no exclusions)

Per item i, with legacy draw L_i (K=1) and matched draws F_ik (K=5):

- `w^L_i = 1[CIPE_A,i < CIPE_L,i]`
- `w^F_i = (1/5) sum_k 1[CIPE_A,i < CIPE_F,ik]`   (indicator averaged over draws, as in Eq. (2))
- Ties score 0 (strict, the registered convention of W). Half-tie versions `w*` score a tie 0.5
  (as W*).

**Primary:** `dW = W_L - W_F = (1/229) sum_i (w^L_i - w^F_i)`, strict.

Secondary (reported, no decision attached):
1. `dW*` (half-tie).
2. `dD_CIPE = (1/229) sum_i [ (CIPE_L,i - CIPE_A,i) - ((1/5) sum_k CIPE_F,ik - CIPE_A,i) ]`
   = the D_L - D_F contrast of Table paired34, on 229 QA instead of 34.
3. Single-draw sensitivity: `dW_k` using matched draw k alone (k = 1..5), all five reported,
   none selected. (Addresses legacy K=1 vs matched K=5; the difference of means is unbiased
   for the difference in per-item win probabilities regardless of K, only the variance differs.)
4. Descriptives: W_L, W_F, W_L*, W_F*, tie fractions per arm, counts of items with
   (w^L=1, w^F=0), (w^L=0, w^F=1) etc. (discordance table with w^F binarised at >0.5 and =0/=1).

Pipeline check (not an outcome): recompute the matched-arm TV W from the stored,
renormalised probability vectors with the same code path; it must reproduce
`outputs/region_tv_corrected.json` MedGemma-4B W (0.6122 to 4 d.p.). If not, stop and log a
code defect before reading any paired number.

## Inference

- Unit of resampling: source image (`image_sha256` from `image_mapping.json`), 175 clusters,
  all QA rows of a drawn image kept, QA-weighted mean (same design as the paper's post-hoc
  image-cluster analysis and Table paired34).
- Primary interval: image-cluster percentile bootstrap, 95%, B = 20,000, seed 20260924.
- Primary test: image-cluster sign-flip randomisation test on per-image sums of
  `d_i = w^L_i - w^F_i`, statistic = QA-weighted mean, B = 20,000 random sign vectors, seed
  20260924, two-sided, p = (1 + #{|T*| >= |T_obs|}) / (1 + B).
- McNemar is not used for the primary estimand because `w^F_i` is a five-draw average, not a
  binary outcome; the sign-flip test is the paired analogue. alpha = 0.05; one model, one
  primary test, no multiplicity correction. Secondary quantities get the same bootstrap
  (95% CI) but no test and no decision.

## Decision rule (on the primary strict dW only)

- **A (legacy higher):** 95% CI entirely above 0 **and** sign-flip p < 0.05.
- **B (matched higher):** 95% CI entirely below 0 **and** sign-flip p < 0.05.
- **C (no detectable difference):** 95% CI contains 0 and p >= 0.05.
- **C' (indeterminate):** CI and test disagree. Written as C, with the disagreement stated.
- If `dW*` has the opposite verdict to `dW`, the paper sentence says so in the appendix
  subsection; the main-text sentence still follows the primary.

## Scope limits that go into every outcome's text

One model (MedGemma-4B); CIPE readout, not the TV-based W of Table 1; one legacy draw vs five
matched draws; post-hoc (the pairing was noticed in review, not planned); legacy placement and
coverage are not stored, so a difference cannot be attributed to foreground coverage
specifically (the two controls differ in placement rule and coverage together); says nothing
about the four general models, whose legacy runs used different QA.

## What the paper says under each outcome

Main text net +0 lines (page gate must pass). Numbers below in angle brackets are filled
from the result JSON only.

**Intro, `main.tex` ~l.137-138.** Current: "MedGemma-4B's legacy run used the matched sequence
but we did not analyse that pairing, and MedGemma-27B never ran under the legacy control."

- A: "MedGemma-4B's legacy run shares the matched run's $229$ QA, and on them its CIPE win rate
  drops from $<W_L>$ (one legacy draw) to $<W_F>$ (five matched draws; difference $<dW>$
  $[<lo>,<hi>]$, Appendix~\ref{app:mg4b-paired}), one model on the CIPE readout; MedGemma-27B
  never ran under the legacy control."
  The paragraph's lead "The difference from the legacy numbers cannot be credited to the
  control:" becomes "For the general models the difference from the legacy numbers cannot be
  credited to the control:".
- B: same frame, "rises from $<W_L>$ ... to $<W_F>$"; lead sentence unchanged.
- C/C': "MedGemma-4B's legacy run shares the matched run's $229$ QA, and on them its CIPE win
  rate does not differ detectably between the one legacy draw and the five matched draws
  ($<W_L>$ vs $<W_F>$, difference $<dW>$ $[<lo>,<hi>]$, Appendix~\ref{app:mg4b-paired}); MedGemma-27B
  never ran under the legacy control." Lead sentence unchanged.

**Appendix S4, `main.tex` ~l.915-916.** Current: "Because it shares the matched run's sequence,
a paired legacy-versus-matched contrast for MedGemma-4B could be formed on those $229$ QA; we
did not analyse it and report none." Replaced in every outcome by: "Because it shares the
matched run's sequence, we formed the paired legacy-versus-matched contrast for MedGemma-4B on
those $229$ QA after this review noticed it (Appendix~\ref{app:mg4b-paired})." A new appendix
subsection `app:mg4b-paired` gives: this registration's path and commit, the estimands, a table
(W_L, W_F, dW [CI], p; dW* [CI]; dD_CIPE [CI]; five single-draw dW_k; tie fractions), the scope
limits above, and the receipt/script paths with SHA-256.

**Abstract** ("historical and matched runs are unpaired, so we do not attribute their
difference to the control"):
- A: "historical and matched runs are unpaired except for MedGemma-4B, whose CIPE win rate on
  the same items is lower under the matched control, so beyond that one model we do not
  attribute their difference to the control;"
- B: "historical and matched runs are unpaired except for MedGemma-4B, whose CIPE win rate on
  the same items is higher under the matched control, so we do not attribute their difference
  to the control;"
- C/C': unchanged (the result does not change what the sentence claims).
If the abstract changes, re-run `6_paper_work/check_abstract_supported.py` and tell the main
session to regenerate FINAL_ABSTRACTS / the OpenReview abstract.

**Conclusion** ("although our unpaired runs cannot say how much of it did"):
- A: "although outside one paired model (MedGemma-4B, lower on CIPE under the matched control)
  our runs cannot say how much of it did"; the later "The historical sample mismatch prevents
  isolating the effect of changing the control" gains "for the general models".
- B, C/C': unchanged.

**Nothing else changes.** No new claim about general models, specialisation, TV, or the
mechanism of Proposition 1 is drawn from this contrast. If the result cannot be integrated
and pass all gates by 2026-09-25 12:00 Beijing, it is recorded in the ELN only (brief rule);
the outcome itself is never a reason not to integrate.

## Receipts

Script `code/scripts/medgemma4b_legacy_vs_matched_20260924.py` (written after this commit),
output `outputs/medgemma4b_paired_20260924/paired_contrast.json` with input SHA-256s (both
receipts, mapping, this registration), script SHA-256, seed, B, and per-item rows.
Checker pins added to `code/scripts/check_paper_numbers.py` (or a new
`check_mg4b_paired_claims_20260924.py`) with a /tmp mutation test.
