# Data licenses

This repository contains three kinds of files, under different terms.

## 1. Code: Apache License 2.0

Everything under `code/` is released under the Apache License 2.0 (`LICENSE`).
Copyright 2026 Xupeng Chen, Jiayi Cheng, Chenqian Le, Runhao Li, Michael Yue.

## 2. Our derived data: CC BY-NC 4.0

The receipts, mappings, read-outs and summaries we generated (`results/`, `outputs/`,
`audit_expected.json`, `manifest.json`, `notes/PREREG_COMMITS.json`) and the registration
documents (`PREREG_*.md`, `notes/PREREG_*.md`) are released under Creative Commons
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0),
https://creativecommons.org/licenses/by-nc/4.0/, matching the terms the paper states for data
we create. Where these files embed third-party material (section 3), that material stays under
its own license and attribution terms.

## 3. Third-party material

No image pixels from any dataset are included. Images must be obtained from the original
releases.

- **SLAKE** (Liu et al., 2021, "SLAKE: A Semantically-Labeled Knowledge-Enhanced Dataset for
  Medical Visual Question Answering", ISBI 2021). License: Creative Commons Attribution 4.0
  International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/, as stated by the
  dataset authors at https://huggingface.co/datasets/BoKelvin/SLAKE (checked 2026-10-12).
  Files:
  - `outputs/legacy_mapping_20260923/source_train_a9083ce6.json` and
    `results/exposure_audit_corrected_20261004/slake_official/{validate,test}.json`: the official
    SLAKE 1.0 `train.json`, `validate.json` and `test.json`, unmodified (sha256 in
    `results/exposure_audit_corrected_20261004/slake_official/SOURCES.md`);
  - SLAKE questions, answers, image names and `detection.json` boxes, excerpted and re-keyed,
    inside `outputs/image_cluster_20260922/image_mapping.json` and
    `outputs/legacy_mapping_20260923/general_identity_mapping.json`; SLAKE annotation boxes and
    image sizes inside the per-item receipts in `results/cipe_scale/` and the read-outs derived
    from them.
  Changes: selection, re-keying and added fields (model outputs, scores, hashes).
- **ChestX-Det** (Lian et al., 2021), a re-annotated subset of NIH ChestX-ray14 (Wang et al.,
  2017). Annotations: Apache License 2.0,
  https://github.com/Deepwise-AILab/ChestX-Det-Dataset (copy in
  `LICENSES/ChestX-Det-Apache-2.0.txt`). Files: sample identifiers and box
  coordinates inside `results/exposure_clean/bbox_fg_*_chestxdet.json`. No ChestX-ray14 pixels
  are included.
- **Model outputs.** The receipts store answer-probability vectors and derived scores produced
  by the open-weight models named in the paper (MedGemma, Gemma 3, InternVL3, Qwen2-VL,
  Qwen2.5-VL). Use of those models is governed by their own licenses; no weights are included.
