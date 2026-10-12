# Foreground-Matched Occlusion Controls for Auditing Annotated-Region Sensitivity in Medical VLMs

Xupeng Chen (NYU), Jiayi Cheng (NYU), Chenqian Le (NYU), Runhao Li (USC), Michael Yue (Columbia)

Paper: arXiv:XXXX.XXXXX

## Summary

Occlusion audits of medical vision-language models (VLMs) often read a larger answer shift from
blacking out an annotated region than from a same-sized control box as evidence that the model
uses that region. That reading assumes both occlusions are equally strong. A black fill placed on
air changes almost no pixels, so such a control loses to nearly any tissue occlusion: a replayed
minimum-overlap control covered 0.659 tissue on average against 0.943 for a matched one (229
items), and under it every model run appeared region-sensitive. We introduce a foreground-matched
control (same size, matched tissue coverage, no overlap with any annotated box) and a proposition
showing that, under a monotone foreground-dose model, a comparator covering less tissue biases
the win rate upward and matching bounds that term. On SLAKE the matched control gives graded
sensitivity: both MedGemma checkpoints lead (strict-win rates 0.639 and 0.612) and survive
post-hoc resampling of 175 source images, while four general models are lower (0.525 to 0.563)
and their detection mostly depends on how ties are scored. The ranking is exposure-confounded
(92.1% of the 229 items sit on SLAKE train-split images, and the MedGemma family reports
training on SLAKE), the masked boxes annotate the image rather than the question target, and the
estimate covers only the 229 of 400 items that admit a matched control. An audit should report
and match what its control covers, drop and count unmatched items, and report ties at half credit
beside strict wins.

## What this repository contains

This is the region-audit subset of the paper's artefacts: the subset the paper says is released
now. It is the R1 review supplement with the MedTrust-IN corpus removed (see "Not included").

- Per-item receipts of the six foreground-matched SLAKE region runs, the four legacy
  general-model runs and the legacy MedGemma-4B run (stored answer-probability vectors and box
  coordinates; no images).
- The recovered image mapping (229 QA on 175 images), the legacy-identity mapping, the
  corrected-TV receipt, and the analysis scripts.
- An offline verifier (Python and NumPy only) that re-runs the image-cluster, legacy-identity,
  paired-subset (34 QA) and MedGemma-4B paired-contrast analyses, and route A of the corrected
  exposure audit, from the packaged receipts.
- Hash-checked but not re-run: every registration the paper cites (`PREREG_*.md`,
  `notes/PREREG_*.md`, with first and last commit of each in our project repository in
  `notes/PREREG_COMMITS.json`; these are commits of a private repository, not an external
  registry), the legacy-coverage replay, the ChestX-Det receipts, the receipt-only instrument
  checks, the Monte Carlo seed summary and the receipt-provenance table.

## Running the verifier

Python 3.9 or newer with NumPy. From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 code/scripts/build_audit_supplement_20260923.py --verify-root .
```

It checks every file listed in `manifest.json` (sha256 and size), re-runs the packaged analysis
scripts on the packaged receipts, writes the recomputed files under `replay/`, and compares every
numerical summary and every per-QA and per-image contribution with `audit_expected.json` and the
stored receipts. It exits non-zero on any mismatch and prints a JSON list of the checks on success.

This is a same-code replay: it shows that the shipped numbers follow from the shipped receipts
and code, not that an independent implementation agrees. Model inference cannot be re-run from
this repository, because source images are referenced rather than redistributed. Scripts that
need the SLAKE image tree or the git history (identity recovery, route B of the exposure audit,
the exposure generator, the coverage replay, the provenance table) are included for inspection
only.

The builder script (`build_audit_supplement_20260923.py`) also built the review supplement from
our project repository; its docstrings still call that package "anonymous". Only its
`--verify-root` mode is meaningful here. `manifest.json` lists the files it covers; its
`release_note` field records the differences from the review supplement.

## Directory map

```
code/medtrust/                 CIPE metric and types used by the MedGemma-4B paired contrast
code/scripts/                  analysis scripts, verifier (build_audit_supplement_20260923.py)
results/cipe_scale/            per-item receipts: foreground-matched (bbox_fg_*) and legacy (bbox_val_*) runs
outputs/cipe_v0/               legacy Qwen2-VL-7B receipt
outputs/image_cluster_20260922/   image mapping (229 QA, 175 images) and image-cluster sensitivity
outputs/legacy_mapping_20260923/  legacy identity mapping, paired-34 protocol and result, official SLAKE train.json
outputs/medgemma4b_paired_20260924/  MedGemma-4B legacy vs matched paired contrast
outputs/legacy_coverage_replay_20260925/  minimum-overlap coverage replay (tissue coverage 0.659 vs 0.943)
outputs/region_tv_corrected.json  corrected normalized-TV receipt
results/exposure_audit_corrected_20261004/  corrected exposure audit (211/229 on train images), SLAKE split files
results/exposure_clean/        ChestX-Det run receipts and read-outs
results/region_instrument_20260926/  receipt-only instrument checks (placebo, tolerance)
results/seed_sensitivity_20261004/   Monte Carlo seed sensitivity
results/receipt_provenance_20261007/ receipt provenance table
PREREG_*.md, notes/            registrations cited by the paper, and their commit record
audit_expected.json, manifest.json  verifier expectations and file hashes
LICENSES/                      third-party license texts
```

## Not included

- **MedTrust-IN v0.3** (items, statistics, datasheet and the PathVQA rehydration script), the
  evaluation code, model configurations, prompt templates and the per-record result dumps of the
  40-model panel. The paper states these will be released upon publication.
- **Image pixels** from any dataset, including the SLAKE scan shown in Figure 1 (from the Medical
  Segmentation Decathlon, CC BY-SA 4.0). Figure 1's plotted values come from
  `outputs/image_cluster_20260922/image_cluster_sensitivity.json`,
  `results/region_instrument_20260926/receipt_only.json` and
  `outputs/legacy_coverage_replay_20260925/coverage.json`.
- Our internal lab notebook. Some registration documents refer to it ("ELN") and to internal
  review notes by path; they are shipped unchanged because their hashes are part of the record.

## License

- Code (`code/`): Apache License 2.0, see `LICENSE`.
- Our receipts, mappings, read-outs and registrations: CC BY-NC 4.0.
- Third-party material keeps its own license: SLAKE (CC BY 4.0) question/answer text, split files
  and annotation boxes; ChestX-Det annotations (Apache 2.0). Details and attribution in
  `LICENSE-DATA.md`.

## Citation

```bibtex
@article{chen2026foregroundmatched,
  title   = {Foreground-Matched Occlusion Controls for Auditing Annotated-Region Sensitivity in Medical {VLMs}},
  author  = {Chen, Xupeng and Cheng, Jiayi and Le, Chenqian and Li, Runhao and Yue, Michael},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## Contact

Xupeng Chen, xc1490@nyu.edu
