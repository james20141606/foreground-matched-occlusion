# Pre-registration — the specificity control our contamination probe has not yet passed

**Written and committed BEFORE the runs.** Commit this file first; the results commit
must come after it in `git log`. If the two land together, treat the pre-registration
as void.

## Why this experiment

Xu, Wu & Ryu, *A Controlled Audit of Pretraining Contamination in Public Medical
Vision-Language Benchmarks* (arXiv:2606.10066, 8 Jun 2026) audits the same benchmark
family we use — SLAKE-En, PathVQA, VQA-RAD (plus OmniMedVQA) — with *provenance*
methods: SigLIP nearest-neighbour overlap against PMC-OA-beta, canonical-order
exchangeability, Min-K%++ tail enrichment, and cross-model top-K Jaccard. Two of their
results bear directly on ours.

**Their positive corroborates our setting.** They report ≈19.8% of SLAKE-En images
having an extreme same-view near-neighbour in PMC-OA-beta, against a far lower rate for
VQA-RAD, with manual review indicating source-level rather than pixel-level duplication.
That is independent evidence that contamination in this benchmark family is real. It is
not evidence about *behaviour*: they run no train/test accuracy contrast and no
image-benefit decomposition, and their likelihood proxies came back negligible and
mixed-sign. Their methods and ours measure different things.

**Their negative is the control we owe.** Their Min-K%++ tail-enrichment and top-K
Jaccard signals were statistically indistinguishable between the medical models and a
BLIP-2 external baseline that never saw the medical data. A detector that fires the same
on a model with no exposure is not detecting exposure. That is a specificity failure, and
they report it themselves.

Our probe (`results/contamination/`, 8 models × 3 datasets × 2 splits) flags exactly one
cell: **SmolVLM-256M on VQA-RAD**, accuracy gap +0.344 [+0.274, +0.411], image benefit
collapsing +0.364 → +0.080 (drop +0.284, 4.0 sd above the 24-cell mean, 3.6× the next
largest), while Qwen2.5-VL-32B holds +0.276/+0.308 on the same dataset. Before we claim
that this is contamination detection rather than an artefact, the statistic must stay
null on a model that plausibly never saw these benchmarks.

## Ground truth established before the run (documented provenance, not our measurement)

This is the part that makes the control interpretable, and it was checked before any
control cell was submitted:

1. `HuggingFaceTB/SmolVLM-256M-Instruct`'s model card states its training data "comes
   from The Cauldron and Docmatix datasets".
2. `HuggingFaceM4/the_cauldron` ships a **`vqarad` config, 313 rows, `train` split only**
   (HF datasets-server `/splits`), and ships **no SLAKE and no PathVQA config** among its
   50 subsets.

So the single model our detector flags has VQA-RAD — and only VQA-RAD, and only its train
side — in its documented training mixture. The probe therefore already carries a
known-answer structure we had not exploited:

- **Within-model dataset contrast (zero compute, already in the receipts).** SmolVLM-256M
  is null on SLAKE (+0.020) and PathVQA (+0.012) and fires only on VQA-RAD. If the
  statistic fired on smallness, weakness, or low capacity, it would fire on all three
  datasets for this model. It fires on exactly the one dataset in the model's documented
  training corpus. This is our size/capacity control and it costs nothing; we state it
  rather than run a new tiny model for it.
- **What is still missing is the other axis:** hold the dataset fixed at VQA-RAD, where we
  fired, and vary the model to one whose documented mixture excludes it. That is this
  experiment.

We will also verify (2) directly rather than resting on the config listing: measure what
fraction of the 313 Cauldron `vqarad` questions appear in the VQA-RAD **train** split
versus the **test** split. Pre-declared prediction: overwhelmingly train, ~0 test. If
that prediction fails, the ground-truth framing above is withdrawn and this
pre-registration's interpretation section is void — we would then be unable to claim the
flagged cell is a known positive, and the control becomes a bare null with no anchor.

## Design

Identical harness, identical code path, no second implementation. Runner
`code/scripts/cipe_v0_pilot.py` via `code/scripts/slurm/run_contamination_probe.sbatch`
with the same `--family qwen2-vl` generic chat-template path, same `--seed 42`, same
matched n per dataset (VQA-RAD 250, SLAKE 350, PathVQA 500, n capped by the smaller test
pool). Analysis extends `code/analysis/contamination_summary.py`; it is not forked.

**Datasets: vqa-rad, slake, pathvqa only.** OmniMedVQA is excluded on a stated structural
ground, not a convenience one: it ships no canonical train/val/test division, so
`load_omnimedvqa_subset` now refuses any split other than train/all rather than compare a
pool to itself and report the no-op as a null.

### Arm A — negative control (the control this experiment is for)

`llava-hf/llava-1.5-7b-hf`. 3 datasets × 2 splits = **6 cells**.

*Why this model is defensible, and the limit of that defence.* LLaVA-1.5's instruction
mixture (`llava_v1_5_mix665k`) is enumerated in its paper — LLaVA-Instruct-158K, VQAv2,
GQA, OKVQA, OCRVQA, A-OKVQA, TextCaps, RefCOCO, Visual Genome, ShareGPT — and its
pretraining stage is an LCS-558K caption subset. No medical VQA corpus appears in either
stage, and no medical fine-tuning was performed. The existence of LLaVA-Med as a separate
model, built by continued training of LLaVA on PMC figure-caption data, is itself evidence
that the base model lacked medical data: the adaptation would be redundant otherwise.
It is also a 7B model, so it sits inside the panel's existing 4B–32B capacity band rather
than at an extreme.

**We cannot prove non-exposure.** "Never saw it" is not provable for any model whose
pretraining corpus is web-scale: LCS-558K derives from LAION/CC/SBU, and VQA-RAD's images
come from MedPix, a public teaching archive that a web crawl could in principle touch. Our
claim is narrower and is the strongest one the evidence supports: *no medical VQA
benchmark appears in either documented training stage, and no question–answer supervision
for these benchmarks was applied.* Our statistic requires question-conditioned
image-grounded memorisation, which incidental caption-level image exposure would not
supply. We will state this limit in the paper rather than assert a clean negative.

### Arm B — corpus replication (the positive direction of the same claim)

`HuggingFaceTB/SmolVLM-Instruct` (2.2B). 3 datasets × 2 splits = **6 cells**.

Same documented corpus as the flagged model — its card states the same Cauldron +
Docmatix mixture — at ~9× the capacity. A negative control alone shows only that the
statistic does not fire on a clean model. This arm asks the complementary question: does
the statistic track *corpus membership* rather than one checkpoint's pathology? It is 6
cells of already-cached compute and it is the arm most able to falsify our own reading of
the SmolVLM-256M result.

### Arm C — BLIP-2, the audit's own baseline: attempted, and expected to be structurally
unavailable

For exact comparability we wanted `Salesforce/blip2-opt-2.7b`, the same external baseline
that broke their detector. Checked before submitting, in the run environment
(transformers 4.57.1): `Blip2ForConditionalGeneration.forward` takes `pixel_values` as its
**first positional parameter**, whereas `LlavaForConditionalGeneration.forward` takes
`input_ids` first with `pixel_values` optional. Our image-benefit statistic requires a
**text-only forward pass** — `acc_text_only`, the second term of
IB = acc − acc_text_only. If `pixel_values` has no default, BLIP-2 cannot produce that
term through the shared code path at all.

**Pre-declared handling.** If the receipt confirms `pixel_values` is required, we report
BLIP-2 as *architecturally incompatible with this statistic* and do not run it. This is a
substantive finding, not an excuse: their statistics are pure-likelihood and run on any
model, while ours needs a model that can be conditioned on text alone. We will say so, and
we will not silently swap in a different model and call it "the BLIP-2 control". If
`pixel_values` turns out to be optional, we run Arm C as 6 further cells and report it.

**Receipt, taken before submitting any cell** (rs-qwen-env, transformers 4.57.1):

```
LLaVA  pixel_values  default=None
LLaVA  input_ids     default=None
BLIP2  pixel_values  default=REQUIRED
BLIP2  input_ids     default=REQUIRED
```

Arm C is therefore **not run**, on the stated structural ground: BLIP-2 cannot be
conditioned on text alone in this harness, so `acc_text_only` — and with it the entire
image-benefit statistic — is undefined for it. Arm A is the specificity control.

### Metrics — unchanged, and fixed here

- **Primary: image-benefit transfer drop**, `IB_train − IB_test` where
  `IB = acc_original − acc_text_only`. Flag threshold **drop > 0.20**, the value already
  committed in `contamination_summary.py`; it is not re-tuned for this experiment.
- **Secondary: train − test accuracy gap** with the Newcombe 95% CI already implemented.
- **Reference distribution**: the existing 24-cell panel, mean drop +0.020, sd 0.066.
  Control cells are additionally reported as z against that distribution.

### Validity gate that precedes all interpretation

The harness's own black-vs-white image-sensitivity positive control must pass
(`image_injection_ok`, threshold 0.02) for every control cell. **A cell that fails it is
invalid, not null** — it means the image was never injected, which is exactly the harness
bug that once produced a saturated CIPE cluster in this project. A failed cell is reported
as failed and excluded; it is never counted as evidence of specificity.

## Pre-declared outcomes

Fixed before seeing any result. No outcome here is a failure of the experiment.

- **Branch 1 — control stays null (specificity supported).** Every Arm A cell has
  drop ≤ 0.20, and in particular the VQA-RAD control cell sits within the panel's existing
  non-flagged range (the other 23 cells span −0.064 to +0.080). Reported as: on the axis
  where the competing audit's detector failed, ours holds — the statistic does not fire on
  a model with no documented exposure, on the same dataset and the same split contrast
  where it fires for a model that has it. Paired with the within-model dataset contrast,
  the detector's one positive is the one cell with documented training-set inclusion.
- **Branch 2 — control also flags (our detector is not specific either).** Any Arm A cell
  has drop > 0.20, or the VQA-RAD control cell is a clear outlier against the 24-cell
  reference. Reported plainly as a specificity failure of our own probe, in the same terms
  we use to describe theirs, and the SmolVLM-256M cell is downgraded from "contamination
  shown directly" to "a large split effect of undetermined cause". We will make this move
  if it fires; it is the same honest move their paper made.
- **Branch 3 — ambiguous.** A control cell lands in (0.10, 0.20] — above the panel's
  observed non-flagged maximum of +0.080 but below the committed threshold. Reported as
  ambiguous: the control neither clears nor condemns the detector, the specificity claim is
  stated as unresolved, and we do not move the threshold to rescue either reading.
- **Branch 4 — invalid.** A control cell fails the image-injection gate. Reported as a
  failed cell with the reduced arm size. If the VQA-RAD control cell specifically fails,
  Branch 1 becomes unavailable by construction — the control we care about did not run —
  and we report the specificity claim as untested rather than supported.

**Arm B branches, declared separately** (Arm B cannot rescue or void Arm A):
- **B-replicates.** SmolVLM-Instruct-2.2B shows a VQA-RAD-specific elevation — VQA-RAD
  drop clearly above its own SLAKE/PathVQA drops. Reported as: the statistic tracks corpus
  membership across two checkpoints of different capacity trained on the same mixture.
- **B-null.** SmolVLM-Instruct-2.2B is null on all three. Reported as: the effect is
  confined to the 256M checkpoint; corpus membership is necessary but not sufficient, and
  the most likely reading is that memorisation of 313 images survives training only at
  small capacity. We report this as a limit on our own claim and do **not** drop the arm.

**Commitment:** we report the branch that fires. We will not add models, drop models,
change n, change the seed, or move the 0.20 threshold after seeing results. If a run fails
technically (OOM, load error) we report the reduced arm rather than substituting another
model.

## What would make this uninterpretable

- Arm A's **VQA-RAD** cell failing to run or failing the injection gate — that is the cell
  the whole control rests on; SLAKE and PathVQA control cells cannot substitute for it,
  because the detector never fired there for any model.
- The Cauldron provenance check contradicting the config listing (i.e. the 313 `vqarad`
  rows not being predominantly VQA-RAD train questions) — the known-answer framing is then
  withdrawn.

## Receipts

Runner `code/scripts/cipe_v0_pilot.py`; submitter
`code/scripts/slurm/submit_contamination_control.sh`; sbatch
`code/scripts/slurm/run_contamination_probe.sbatch` (unmodified); provenance check
`code/scripts/cauldron_provenance_check.py`; outputs
`results/contamination/contam_{llava15_7b,smolvlm_2b}_{vqa-rad,slake,pathvqa}_{train,test}_n*.json`
and `outputs/cauldron_vqarad_provenance.json`; analysis
`code/analysis/contamination_summary.py` (extended with a control arm, not forked).
