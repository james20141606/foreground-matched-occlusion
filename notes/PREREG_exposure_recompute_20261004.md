# PREREG: recomputation of the SLAKE exposure audit (S1), registered 2026-10-04T03:26+08:00

Not blind: the BootLoops audit (`6_paper_work/next_cycle_20260927/bootloops_audit_20261003/medtrust.md`, F1)
already reports 211/229 = 92.1% (both-splits 62, test-only 3) against the submitted 213/229 = 93.0%
(52, 4). What this note fixes in advance is the procedure and the decision rule, not ignorance of the
answer. The submitted PDF (15c4f3d, sha256 d79dd5378480...) is frozen and is not edited by this work.

The recomputation uses two routes that share no code and no ordering assumption, and the repaired
generator must agree with both. **Route A (mapping):** each retained `sample_id` of
`results/cipe_scale/bbox_fg_medgemma4b_n400.json` is joined to its `image_ref` through the 09-22
fingerprint recovery `outputs/image_cluster_20260922/image_mapping.json`, and `image_ref` is looked up
by `img_name` in the official SLAKE 1.0 `train.json` / `validate.json` / `test.json` (train.json
byte-identical to the committed `outputs/legacy_mapping_20260923/source_train_a9083ce6.json`,
sha256 7a9fb81f...). **Route B (JPEG fingerprint):** for each retained receipt row, every SLAKE image
whose decoded JPEG size and integer `detection.json` boxes equal the row's `image_size` and
`organ_boxes` is a candidate; the item is assigned only if exactly one candidate (up to byte-identical
JPEG sha256) exists, and its split membership is read from the same official files. No shuffle, no
candidate pool. **Repaired generator (G):** `code/scripts/region_exposure_audit.py` rebuilds the pool
with the foreground loader's own admission filter (detection.json present, image present, at least one
well-formed 4-number box), then asserts per retained item that the reconstructed image's boxes and
size equal the receipt row's (identity check per entry, replacing the bounds-only guard), and
cross-checks `image_ref` against Route A when the mapping is supplied. It runs on a copy
(`results/exposure_audit_corrected_20261004/`), never overwriting `outputs/region_exposure_audit.json`.

**Decision rule.** If A, B and G give the same per-item image assignment for all 229 items and the
same six-cell membership table, that table is the corrected receipt and the revision carries its
numbers (n on train images, % to one decimal, both-splits count, test-only count). If any two routes
disagree on any item, no corrected number is written anywhere; the disagreeing items are listed and
the discrepancy itself is reported. Either way the conclusion this receipt supports is checked
separately and stated as such: the specialist-vs-general contrast stays exposure-confounded iff the
train share is a large majority and the splits remain not image-disjoint. **Plant:** a synthetic
SLAKE tree with one image whose `detection.json` is `[]` placed before a retained item must make the
old pool logic produce a mis-assignment that its bounds guard does not catch, and must make the
repaired identity check exit non-zero; the repaired filter on the same tree must pass.
