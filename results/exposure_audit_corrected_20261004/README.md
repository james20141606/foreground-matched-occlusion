# Corrected SLAKE exposure receipt (2026-10-04)

Registered in `notes/PREREG_exposure_recompute_20261004.md` (commit ada8fbd) before computing.
Replaces, for the revision, `outputs/region_exposure_audit.json` (left in place unchanged: it is the
receipt the frozen submitted PDF cites).

| | on train images | both train+test | test-only | membership {test, test+train, test+val, train, train+val, val} |
|---|---|---|---|---|
| submitted (defective pool) | 213/229 = 93.0% | 52 | 4 | 4, 52, 6, 106, 55, 6 |
| Route A, image_mapping.json + official split json | **211/229 = 92.1%** | **62** | **3** | 3, 62, 8, 101, 48, 7 |
| Route B, JPEG size + detection.json box fingerprint | **211/229 = 92.1%** | **62** | **3** | 3, 62, 8, 101, 48, 7 |
| G, repaired generator (identity check per item) | **211/229 = 92.1%** | **62** | **3** | 3, 62, 8, 101, 48, 7 |

Per-item image assignment: A, B and G agree on 229/229 (`COMPARE.json`). Distinct images 175;
splits not image-disjoint (940 slots, 642 images), unchanged.

Cause: the old pool admitted 62 images whose detection.json is `[]`, which the foreground loader
drops; from pool index 32 (xmlab272) the lists were offset and 211/229 retained items were looked up
on the wrong image. Running the old filter on the same data reproduces all six published cells exactly.

Files: `route_a_mapping.json`, `route_b_fingerprint.json`, `region_exposure_audit_corrected.json`,
`COMPARE.json`, `slake_official/` (validate.json, test.json + SOURCES.md; train.json is
`outputs/legacy_mapping_20260923/source_train_a9083ce6.json`).
Scripts: `code/scripts/exposure_route_a_mapping_20261004.py`,
`code/scripts/exposure_route_b_fingerprint_20261004.py`, `code/scripts/region_exposure_audit.py`,
plant test `code/scripts/test_region_exposure_audit_plant_20261004.py`.
Image tree used (local, not committed): a local copy of the SLAKE 1.0 `imgs/` tree [path redacted for public release]; all 1,284
files listed in image_mapping.json `source_hashes` exist there with identical sha256 (checked 2026-10-04).
