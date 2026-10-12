# PRE-REGISTRATION: corrected region readout (normalization bug + M3 estimand)

Committed BEFORE the corrected analysis is run and before any corrected number enters the
paper. **This registration is NOT blind, and saying so is part of it:** an external
reviewer model ran a read-only normalized diagnostic and reported that the MedGemma
fractions stay near 0.646/0.594 while the four general models' clustered estimates sit
just above 0.5 (0.559-0.595). So the likely outcome is already visible. What this
registration buys is therefore not blindness but *fixed consequences*: the branches below
are written before the official numbers exist, and the paper follows them.

## The defect (verified, not reported)

`cipe_bbox_external_validation.probs()` returns `[exp(lp_yes - m), exp(lp_no - m)]` with
`m = max(lp)`. The max element is exactly 1.0, so the vector sums to (1, 2]. Measured over
the committed receipts: **2378 of 2748 stored vectors do not sum to 1** (range 1.0000 to
2.0000, mean 1.1044). `cipe.py` normalizes internally, so every CIPE number is unaffected.
`cipe_region_metric_comparison.readouts()` computes `tv` and `dp_yes` directly on the
stored weights, so **the published "total variation" is not total variation** and the
headline win rates 0.646/0.594/0.563 inherit the defect.

## The correction (receipt-only; no inference is re-run)

1. Normalize every stored vector to sum 1 before any readout.
2. Report BOTH estimands rather than choosing:
   - **D**: `D_i = TV_organ,i - mean_k TV_control,ik`, item bootstrap CI on mean D.
   - **W**: `W_i = mean_k 1[TV_organ,i > TV_control,ik]`, controls clustered within item.
   The previously published quantity, `1[TV_organ > mean_k TV_control]`, is reported as a
   third, deprecated line: averaging five controls is legitimate variance reduction for D,
   but it does not carry a chance-0.5 interpretation, which is what the paper used it for.
3. Report all five single-draw analyses. No draw is selected.
4. Benjamini-Hochberg across the six models, as before.

## Branches (fixed now)

- **R1 SEPARATION SURVIVES**: both MedGemma checkpoints significant under W (clustered) and
  D after BH, and at least three of the four general models not significant. The
  specialist-vs-general claim stands, restated on the corrected readout.
- **R2 EFFECT SURVIVES, SEPARATION DOES NOT**: MedGemma significant but general models also
  significant. The paper **retires the specialist-vs-general claim** and reports a graded
  occlusion sensitivity present across the panel, with the specialist checkpoints strongest.
  The abstract's "medical-specialist family" framing goes.
- **R3 EFFECT DOES NOT SURVIVE**: MedGemma not significant under the corrected estimands.
  The region result is retracted and joins the paper's artefact catalogue as a fifth
  reversal -- which is the paper's genre, so it is reportable rather than fatal.
- **R4 SPLIT**: only one MedGemma checkpoint survives -> family-level replication is gone;
  report a single-checkpoint effect and say so.

## Bound stated now

Whatever survives is a claim about **differential answer-distribution sensitivity to this
occlusion**, not about where the model looks, not about anatomical understanding, and not
about clinical reliance. Occlusion remains a compound intervention despite tissue matching.

No corrected number exists at the time of this registration.
