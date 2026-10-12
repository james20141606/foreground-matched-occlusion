# PRE-REGISTRATION: the region experiment on a disclosure-clean dataset

Committed BEFORE any dataset is downloaded and before any model sees any item.
This is the experiment sec:exposure says our SLAKE result cannot be: 93.0% of the SLAKE
region items sit on images MedGemma's own report says it trained on, so specialist-vs-
general ranking there is uninterpretable. Here the items come from a dataset absent from
every MedGemma disclosure surface.

## Dataset (fixed now, with a registered fallback)

**Primary: VinDr-CXR** (PhysioNet, credentialed). Verified across the MedGemma technical
report v4 (arXiv:2507.05201), the MedGemma v1 and 1.5 model cards, and the MedSigLIP card:
zero mentions in any capacity. Original Vietnamese hospital collection -- no derivation
chain into anything MedGemma discloses. 22 finding classes, 5,342 train images with >= 1
radiologist box; test split has 5-reader consensus boxes.

**Fallback (only if credentialed access is not available in time): ChestX-Det** (open,
Apache-2.0; 2,967 images, 9,639 box/polygon instances, 13 classes). Its images are a
subset of NIH CXR14, so it inherits a web-crawl exposure risk VinDr does not have, and the
paper must carry that caveat verbatim if the fallback is used. Which dataset ran is stated
in the paper either way; running both is permitted and reported side by side.

## What "exposure-clean" is allowed to mean

"Absent from all disclosed training data." NOT "never seen": MedGemma's card hedges with
"including", its Table 1 folds public sets into "Internal" labels, and the Gemma 3 base
crawl is undisclosed. The paper states this scope sentence next to the result.

## Models (fixed now)

Specialist: **MedGemma 1 4B-it only.** The 27B multimodal variant trained on Chest
ImaGenome "to enable anatomy localization" (report Appendix F) -- region-grounding
supervision -- so it is excluded from the clean arm and this exclusion is stated in the
paper. Generals: the same four as the SLAKE arm (Qwen2-VL-7B, Qwen2.5-VL-7B, Gemma3-4B,
InternVL3-8B).

## Item construction (fixed now)

Closed yes/no items in MedGemma's own published prompt template (report Table A7:
"Is there <CONDITION> in this image? ..."). YES items: (image, finding) pairs with a
radiologist box for that finding. NO items: same question asked of images whose reader
consensus marks that finding absent (box from a different finding on that image serves as
the occlusion target, preserving the geometry of the yes arm). Target n = 200 YES + 200
NO, class-stratified, seed 42; if the dataset supports fewer, the reader reports the
shortfall and refuses scoring below n = 120 total.

## Instrument (frozen -- no changes permitted)

The SLAKE arm's foreground-matched occlusion control, byte-identical logic: occlude the
finding box vs an area- and tissue-matched foreground control region elsewhere; readout is
the CORRECTED total-variation W (the probs() defect fix of PREREG_region_tv_correction.md
stays in). Same seeds, same B for the permutation null, same BH-FDR across models.
[Terminology corrected by AMENDMENT 2, wording only, implementation untouched: the frozen
instrument's inference is an ITEM BOOTSTRAP (two-sided p for mean W against 0.5), not a
permutation null; this sentence misnamed it at registration time.]

## Branches (fixed now; consequences bind)

- **E1 SPECIALIST SIGNAL SURVIVES**: MedGemma-4B's W is significant (q < 0.05) AND exceeds
  every general model's W on disclosure-clean items. The paper reports a specialist-vs-
  general contrast that exposure cannot explain -- scoped to one checkpoint, one dataset,
  with the "disclosed" caveat. The SLAKE withdrawal STANDS regardless (it is about SLAKE).
- **E2 RANKING DISSOLVES**: MedGemma-4B is not significant, or does not exceed the
  generals. The exposure explanation is confirmed as sufficient for the original headline,
  and the paper says the specialist advantage did not survive its first disclosure-clean
  test. This is the costlier branch and is registered first-class.
- **E3 INSTRUMENT FAIL / UNDERPOWERED**: the positive control (black-vs-white patch flip)
  fails on this dataset, or the roster floors below n = 120. Reported as such; no partial
  number is promoted.

Reader: `code/scripts/exposure_clean_read.py`, committed with its selftest firing E1, E2
and E3 on synthetics BEFORE any real receipt exists. The reader refuses partial input and
prints the branch verbatim.

No VinDr-CXR or ChestX-Det file exists on our disks at the time of this registration.

## AMENDMENT 1 (2026-08-11, NON-BLIND, all receipts already seen) — the positive control's premise fails on finding-presence questions

Full disclosure first: at the time of this amendment the five W values, their q values, and
both diagnostics below have ALL been seen. Nothing here is blind. The protection is the
same as the TV-correction precedent (PREREG_region_tv_correction.md): every consequence is
written down before the amended reader runs, and the paper must carry this non-blindness
sentence next to any number this amendment promotes.

**What happened.** The registered E3 fired twice over. First, the committed reader used a
0.5 bar for the positive control -- a transcription error; the frozen instrument's own
committed bar is IMAGE_SENSITIVITY_THRESH = 0.02 (cipe_v0_pilot.py, receipts carry the
flag as image_injection_ok). The reader is corrected to the instrument's bar; that fix
direction is dictated by a committed constant, not by choice. Second, even under 0.02,
medgemma4b fails (0.016) -- but the control's premise does not survive the question type.
The control asks whether p(yes) differs between an all-black and an all-white image. On
SLAKE-style organ questions it does. On "Is there <finding> in this image?" a model that
correctly ingests a BLANK image answers "no" at either colour: colour-insensitivity is the
CORRECT behaviour, not the harness bug the control was built to catch (its docstring:
detect image-never-injected text-only forwards).

**The evidence, from the existing receipts.** A text-only forward gives the same p(yes)
for one question on every image. Within-question SD of p(yes) across the real images:
medgemma4b 0.384 (per-item p(yes) spans 0.000-1.000), qwen25vl_7b 0.135, gemma3_4b 0.129,
internvl3_8b 0.112, qwen2vl_7b 0.019. medgemma4b is unambiguously ingesting images. And
the two diagnostics DISAGREE IN OPPOSITE DIRECTIONS on two models: medgemma4b fails colour
but is image-driven; qwen2vl_7b passes colour (0.208) but is essentially image-INSENSITIVE
on real content -- and its W = 0.535 ~ 0.5 is exactly the coin-flip W an image-insensitive
model must mechanically produce.

**Amended ingestion diagnostic.** A model enters the comparison iff its within-question
mean SD of p(yes) across real images is >= 0.05. Rationale for the bar: the W readout
compares occlusion-induced shifts; a model whose answer distribution barely depends on the
un-occluded image (SD below ~0.05, an order of magnitude above the text-only floor ~0)
drives W to 0.5 mechanically, so below that bar W is uninformative rather than evidence.
The colour control is still reported but no longer gates this question type; its premise
failure IS one of the paper's artifact classes now.

**Amended branches (consequences fixed before the amended reader runs).**
- E1' : medgemma4b passes the amended diagnostic, is significant (q < 0.05), and exceeds
  every general model that passes it. Reported as the disclosure-clean contrast, with
  (i) the non-blindness sentence, (ii) the excluded models named with their SD, and
  (iii) the "disclosed training data" scope sentence from the base registration.
- E2' : medgemma4b passes the diagnostic but is not significant or does not exceed every
  passing general. Same consequence as E2.
- E3' : medgemma4b itself fails the amended diagnostic. Instrument verdict stands; no
  number promoted.

Values already seen imply E1' will fire (0.681 > 0.604/0.595/0.546, all pass; qwen2vl_7b
excluded at 0.019). Stating that here, before the amended reader runs, is the point: the
amendment cannot pretend to predict; it can only bind consequences and disclose.

## AMENDMENT 2 (2026-08-12, NON-BLIND, all values seen) — the NO arm does not match the registered estimand; arm-stratified reporting, seed harmonisation, and corrections to Amendment 1's framing

Full disclosure first: everything in this amendment was written AFTER an external review
verified the defects it fixes and AFTER every number below had been computed and seen.
Nothing here is blind. The protection is the same as Amendment 1: every consequence is
bound here, before the amended receipts pipeline and reader run, and the paper carries
this non-blindness sentence next to any number this amendment promotes.

**Defect A (estimand).** The registration above says a NO item's occlusion target is "box
from a different finding on that image ... preserving the geometry of the yes arm" —
singular, geometry-preserving. The committed loader (chestxdet_loader.py) instead supplies
ALL other findings' boxes on the image. Measured over the committed receipts (315 scored
items, identical across the five models): YES items (n=168) occlude 1.43 boxes on average,
NO items (n=147) occlude 2.73, and 5 NO items carry duplicate boxes (the loader
concatenates every finding's box list and ChestX-Det occasionally repeats a box). The NO
arm therefore does NOT preserve the yes arm's geometry, and what it measures is the
answer's sensitivity to occluding OTHER findings' regions — cross-finding context
sensitivity — not localization of the queried finding. Only the YES arm's construction
matches the registered estimand.

**Registered fix — reporting, not re-running.** No model is re-run and no item changes.
The receipts pipeline (expclean_receipts.py) additionally emits per-arm W, p, q and n
(the arm is encoded in sample_id: cxdet_yes_* / cxdet_no_*), with BH-FDR across the five
models WITHIN each arm, alongside the combined quantities. Reporting is arm-stratified:
the YES arm is PRIMARY (its construction matches the registered estimand); the NO arm is
reported as a LABELLED SECONDARY measuring cross-finding context sensitivity, never
promoted as localization evidence. The reader's verdict is taken from the YES arm; the
registered n >= 120 floor applies to the primary arm and to the combined roster.

**Defect B (seed).** The receipts pipeline ran with SEED=0 where the registration promised
"same seeds" as the frozen analysis (region_tv_corrected.py, SEED=42). Corrected to 42 and
all receipts regenerated. W is deterministic given the receipts and does not move; only
the bootstrap p (and hence q) shifts within Monte-Carlo error (medgemma4b combined q
1.7e-4 -> 2.5e-4).

**Corrections to Amendment 1's framing (supersedes its exclusion mechanics).**
(i) The SD_BAR = 0.05 bar was chosen after the SD values had been seen; it is post-hoc and
is hereby demoted from an exclusion gate to a DIAGNOSTIC LABEL. The ingestion SD is
reported continuously for all five models (medgemma4b 0.384, qwen25vl_7b 0.135, gemma3_4b
0.129, internvl3_8b 0.112, qwen2vl_7b 0.019); qwen2vl_7b is described as image-insensitive
UNDER THIS YES/NO READOUT, its W is reported in the table rather than excluded, and no
exclusion is needed for the ordering because medgemma4b's W exceeds qwen2vl_7b's W in
every stratum regardless. The bar is retained only as the E3 instrument gate on the
specialist itself (a specialist that does not ingest the image has no interpretable W).
(ii) Amendment 1 claimed an image-insensitive model "must mechanically produce" W = 0.5.
That is wrong about the implementation, which scores STRICT wins (TV_organ > TV_control):
a perfectly image-invariant model produces identical distributions under every occlusion,
so every comparison is a tie and W = 0, while a model with a small noisy image effect
approaches the coin-flip 0.5. Either way W stops being evidence about localization; the
mechanics are as stated here, not as Amendment 1 stated them.

**The values already seen (seed 42; W deterministic, p/q by item bootstrap, B=10,000,
BH within stratum across the five models).**

    model          combined W / q (n=315)   YES W / q (n=168)     NO W / q (n=147)
    medgemma4b     0.681 / 2.5e-4           0.742 / 5.0e-4        0.612 / 5.0e-3
    internvl3_8b   0.604 / 2.5e-4           0.626 / 5.0e-4        0.578 / 0.026
    qwen25vl_7b    0.595 / 3.3e-4           0.607 / 1.0e-3        0.581 / 0.023
    qwen2vl_7b     0.535 / 0.087            0.570 / 0.016         0.495 / 0.879
    gemma3_4b      0.546 / 0.038            0.526 / 0.358         0.569 / 0.052

**Amended branches (consequences fixed before the amended reader runs).**
- E1'': medgemma4b passes the ingestion diagnostic, its YES-arm q < 0.05, and its YES-arm
  W exceeds every general's YES-arm W (all four generals — no exclusion). Reported as the
  disclosure-clean contrast with (i) the non-blindness sentence covering BOTH amendments,
  (ii) all five ingestion SDs stated, (iii) the "disclosed training data" scope sentence,
  and (iv) the NO arm carried only as the labelled cross-finding secondary.
- E2'': medgemma4b passes the diagnostic but its YES-arm result is not significant or does
  not exceed every general. Same consequence as E2.
- E3'': medgemma4b fails the ingestion diagnostic, or the primary arm / combined roster
  floors below n = 120. Instrument verdict; no number promoted.

Values already seen imply E1'' will fire (YES arm: 0.742 > 0.626/0.607/0.570/0.526).
As with Amendment 1, stating that here is the point: this amendment cannot pretend to
predict; it can only bind consequences and disclose.
