# Pre-registration — does CIPE beat cheap readouts at the region control?

**Written and committed BEFORE the runs.** Commit this file first; the results commit
must come after it in `git log`. If the two land together, treat the pre-registration
as void.

## Why this experiment

The item-level comparison already ran and CIPE lost: over 17 certified `n=500`
evaluations, macro-mean AUROC is 0.529 for CIPE against 0.540 for an argmax flip rate
(paired difference +0.010 favouring flip, 9/17 evaluations, Wilcoxon p=0.52 — i.e.
indistinguishable). CIPE also ranks items almost identically to total variation and to
|Δp(yes)| (ρ=0.924 both).

The foreground-matched SLAKE region control is the one place CIPE still has a surviving
positive: under a tissue-matched control the annotated-organ effect vanishes for all
four general VLMs and survives only for MedGemma-4B (organ-lower 0.585,
p=6.9×10⁻⁵). So the honest question is narrow and specific:

> **On the intervention where CIPE still works, does CIPE separate annotated-organ from
> matched-tissue occlusion materially better than TV, |Δp(yes)|, or a flip rate?**

If the answer is no, CIPE's distributional form has no demonstrated advantage anywhere
in this paper, and the paper is a measurement audit that reports its own metric as
unvalidated.

## Design

- **Models (6):** MedGemma-4B-it, MedGemma-27B-it (specialists), plus the four general
  VLMs already run under the foreground-matched control — Qwen2-VL-7B, Qwen2.5-VL-7B,
  InternVL3-8B, Gemma-3-4B.
- **Items:** the identical foreground-matched SLAKE subset already used
  (`cipe_bbox_foreground_control.py`; same seed, same 5 control draws per item, same
  coverage matching, n≈229 of 400 admitting a matched placement).
- **Readouts (4):** CIPE, total variation, |Δp(yes)|, argmax flip rate — all computed
  from the same stored probability vectors, so no readout gets a different intervention.
- **Statistic:** per model, AUROC for separating *organ-occluded* from
  *matched-tissue-occluded* interventions on the same item. Paired bootstrap over items
  (10,000 resamples, seed 42) for each readout and for the paired difference
  CIPE − best-alternative.
- **Multiplicity:** 6 models × 4 readouts. Benjamini–Hochberg across all cells; report
  uncorrected p and q together, as in the contamination probe.

## Pre-declared outcomes

Fixed before seeing any result. No outcome here is a failure of the experiment.

- **Branch A — CIPE materially and significantly better.** CIPE's AUROC exceeds the
  best alternative by ≥0.05 with a paired 95% bootstrap CI excluding zero, and this
  replicates in **both** MedGemma checkpoints. Reported as: the distributional form buys
  a real advantage at the region intervention, in medical-pretrained models. This is the
  only branch in which the paper claims a validated metric advantage.
- **Branch B — CIPE better but not materially, or not replicated.** Positive CI but
  Δ<0.05, or significant in one specialist and not the other. Reported as: a weak,
  single-checkpoint effect insufficient to validate the metric. No advantage claimed.
- **Branch C — indistinguishable (the live outcome).** Paired CI covers zero, as at the
  item level. Reported as: CIPE has **no demonstrated advantage over a flip rate at any
  intervention we tested**, and the paper states plainly that its metric is unvalidated
  and that a flip rate is the cheaper equivalent. Given 0.529 vs 0.540 item-level, we
  regard this as the most likely branch.
- **Branch D — CIPE worse.** A cheap readout beats CIPE with a CI excluding zero.
  Reported as such, and we additionally recommend against CIPE in favour of the cheaper
  readout.

**Commitment:** we report the branch that fires. We will not add models, drop models,
change the item subset, or switch the primary statistic after seeing results. If a run
fails technically (OOM, load error) we report the reduced n rather than substituting a
different model.

## What would make this uninterpretable

- Fewer than both MedGemma checkpoints completing → Branch A becomes unavailable by
  construction (it requires replication); we would report at most Branch B.
- Fewer than ~100 matched items surviving → we report the comparison as underpowered
  rather than as a null.

## Receipts

Runner `code/scripts/cipe_bbox_foreground_control.py` (extended to persist
`p_original` / `p_counterfactual` per intervention), analysis
`code/scripts/cipe_region_metric_comparison.py`, outputs
`results/cipe_scale/bbox_fg_*_n400.json` and
`outputs/cipe_v0/region_metric_comparison.{txt,json}`.
