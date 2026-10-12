# Consequence registration: the region arm sits on images the specialist trained on

Committed BEFORE the paper is edited, after the measurement and before deciding what to
claim. The measurement is in `outputs/region_exposure_audit.json`, produced by
`code/scripts/region_exposure_audit.py`, which reproduces the region arm's deterministic
item selection exactly.

## What was measured (not disputed, and not ours to spin)

- MedGemma technical report arXiv:2507.05201 S3.4, verbatim: **"For SLAKE, we used the
  default train/test split."** The same report states that VQA-RAD's original test set
  "includes some duplicated images in the train and test sets (with different questions)"
  and that the authors built their own splits to avoid that contamination -- for VQA-RAD.
  Not for SLAKE.
- **215 of our 229 region items (93.9%) sit on images that appear in the SLAKE train
  split.** Exactly **1** item is test-only.
- **SLAKE's official splits are not image-disjoint**: 586 + 174 + 180 = 940 image-name
  slots over 642 distinct images; 54 of our items sit on images appearing in *both* train
  and test. So no "restrict to the test split" manoeuvre removes image exposure -- the
  non-overlap experiment the reviewer sweep proposed **cannot be constructed from SLAKE
  at all**.

## The consequence, fixed before the edit

The specialist-vs-general contrast on SLAKE is **confounded with training exposure** by
construction: the specialist family has seen ~94% of these images, the four general models
have not. No subset of SLAKE separates the two. Therefore:

1. The paper **stops claiming a specialist-vs-general effect as evidence about medical
   specialisation**. The corrected numbers stay (they are what was measured), but the
   ranking is reported as *exposure-confounded* and the abstract's
   "medical-specialist family" reading is withdrawn.
2. The finding is reported **as a reversal, in the paper's own artefact catalogue** --
   which is this paper's genre. It is the strongest instance in the catalogue because it
   dissolves the paper's own headline positive rather than someone else's result.
3. What survives as a claim: on a dataset a model family has trained on, an occlusion of
   the annotated organ moves that family's answer distribution more than a tissue-matched
   control does. That is a measurement about **differential answer-distribution
   sensitivity under exposure**, not about where a model looks and not about clinical
   grounding.
4. We do **not** substitute a new dataset five weeks before the deadline. We state the
   experiment that would settle it (a region control on a corpus with image-disjoint
   splits and no specialist exposure) as the named next step.

## What must NOT happen

No search over subsets, readouts or models after seeing this. The corrected TV analysis
(`PREREG_region_tv_correction.md`, branch R1) stands as run; this registration changes what
that result is *called*, not what it is.
