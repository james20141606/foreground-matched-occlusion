# PREREG: tissue coverage of the legacy (minimum-overlap) control, by deterministic replay

Registered 2026-09-25 (Beijing), before any legacy control box was replayed and before any
legacy coverage value was computed. CPU only; no VLM is loaded.

## Question

Did the legacy minimum-overlap control placement (`random_boxes` in
`code/scripts/cipe_bbox_external_validation.py`, run for MedGemma-4B as
`results/cipe_scale/bbox_val_medgemma4b_n400.json`) cover less tissue than the annotated
boxes and the foreground-matched placements? This is the premise of Proposition 1(b); the
paper currently says the legacy coverage "is not stored" and "cannot be measured".

## What is already known (disclosure)

- Aggregate legacy and matched CIPE win rates, the paired MedGemma-4B contrast
  (+0.054 [+0.005,+0.105]) and the stored matched coverages (annotated 0.961, matched 0.943,
  means of per-item means) are in the paper.
- While scoping this replay I counted exact values in the legacy receipt: `cipe_random == 1.0`
  on 73/400 items and `cipe_relevant == 1.0` on 32/400. No box, image or coverage was
  looked at.
- The legacy receipt stores per-item CIPE and box counts only: no control boxes, no answer
  distributions.

## Why the replay is deterministic

`main()` builds `rng = random.Random(args.seed)` (seed 42) after model load and the positive
control, and `random_boxes` is its only consumer. Per item it draws, for each annotated box in
order, up to 30 `(randint(0, W-w), randint(0, H-h))` pairs and stops at the first zero-overlap
placement with that box. Consumption depends only on image size and box geometry. An item whose
forward pass raised before `random_boxes` would not consume the stream, but all 400 rows are in
the receipt and a row is written only on success, so every item consumed it. `random_boxes` is
byte-identical between the run commit (4c4b785) and HEAD.

Inputs: item order, image refs, sizes and ordered boxes from
`outputs/image_cluster_20260922/image_mapping.json` (400 items, seed-42 loader, validated by the
09-22/09-24 identity gates); SLAKE images from the torch copy, each checked against the
mapping's `image_sha256`.

## Identity checks (all must pass; otherwise the verdict is "cannot reproduce", ELN only)

- **I1, pipeline.** Recomputed annotated-box coverage for all 400 items and matched-control
  coverage for the 229 admitted items equal the stored `organ_fg_coverage` /
  `control_fg_coverage` in `bbox_fg_medgemma4b_n400.json` exactly (float equality), using the
  fg script's `foreground_mask`/`_ibox`/`_coverage`.
- **I2, rng semantics.** Replaying the foreground-matched placement (`control_draw`, fresh
  `random.Random(42)`, all 400 items in order, 5 draws each) reproduces every stored matched
  control box exactly and the same 229/171 partition. This checks item order, sizes, boxes, the
  foreground masks and Python's `random` on this machine.
- **I3, legacy fingerprint (CPU-only stand-in for a model-forward gate).** Let U be the items
  where applying the replayed legacy boxes with the legacy `mask_boxes` leaves the RGB image
  byte-identical (the boxes fall on pixels that are already pure black). With an unchanged
  image the forward pass repeats `p_o`, KL is exactly 0 and the legacy `cipe_random` must be
  exactly `1.0`. Pass: |U| >= 10 and every item in U has `cipe_random == 1.0`. (If the replay
  were wrong, items in U would show 1.0 only at about the receipt's base rate, <= 73/400.)
  The model-forward identity gate (rerun MedGemma-4B on replayed boxes and match
  `cipe_random` to the float) is skipped: it needs a VLM.

## Estimands (MedGemma-4B legacy sequence only; general-model legacy runs used different QA and are not replayed)

Coverage phi(r) is the fg script's definition (share of grayscale pixels above
min + 5% of the dynamic range; box sliced `[y:y+h, x:x+w]`). Per item: mean over its boxes.
Summaries are means of per-item means, as for the stored 0.961 / 0.943.

- **Primary.** On the 229 admitted items: phi_L (legacy), with Delta = phi_F - phi_L, where
  phi_F is the item's mean over its 5 matched draws. 95% image-cluster bootstrap CI for Delta
  (175 images, 20,000 draws, seed 20260925, percentile).
- **Secondary.** On all 400 legacy items: phi_L versus annotated phi_A, Delta_A = phi_A - phi_L,
  image-cluster CI (296 images; slake_246 whose image identity is ambiguous is kept with its
  mapped image, and the result is also reported without it if it changes any digit shown).
- **Descriptive.** Share of legacy / matched / annotated boxes with phi < 0.5.
- **Not analysed** (to avoid a forking path): any relation between an item's coverage gap and
  its CIPE win. The replay measures coverage, not the effect of coverage on W.

## Decision rule (primary, 229 items)

- **A (premise holds, large):** CI lower bound of Delta > 0 and Delta >= 0.10.
- **B (premise holds, modest):** CI lower bound > 0 and Delta < 0.10.
- **C (legacy not biased toward air on these items):** CI includes 0, or Delta < 0.

## Sentence to print for each outcome

Main text (in §3 "Two controls", replacing nothing that carries a limit; appendix subsection in
S4.1 with the numbers, receipt and script):

- **A:** "Replayed from its seed on CPU, the legacy placement covered less tissue on the same
  229 items (mean $\phi_L$ against $0.943$ matched and $0.961$ annotated; Appendix~X), so
  Proposition~1(b)'s premise held for our first control; the replay does not say how much of
  the legacy win rate this produced."
- **B:** "Replayed from its seed on CPU, the legacy placement covered somewhat less tissue on
  the same 229 items (mean $\phi_L$ against $0.943$ matched; Appendix~X), a smaller gap than a
  control on air would give; the replay does not say how much of the legacy win rate this
  produced."
- **C:** "Replayed from its seed on CPU, the legacy placement did not cover less tissue on the
  same 229 items (mean $\phi_L$ against $0.943$ matched; Appendix~X): the air mechanism of
  Proposition~1(b) did not operate here, and the MedGemma-4B contrast is not explained by
  coverage." Under C the introduction's "can favour background air" and Fig. 1 caption stay
  (they describe a possibility) but get "(on the replayed items it did not; Appendix~X)", the
  abstract's "Our own first control could favour air" gets the same qualifier if space allows,
  and §3.1 "Part (b) is the failure mode our first design was open to" gains "though on
  replay it did not operate".
- **Cannot reproduce (any of I1-I3 fails):** no paper change; ELN only. The paper keeps
  "legacy coverage is not stored".

Whatever the outcome, App. S4.1 "its foreground coverage cannot be measured" is changed only if
I1-I3 pass, and then to "is not stored but can be replayed from the seed (Appendix~X)".

## Deliverables

`code/scripts/legacy_coverage_replay_20260925.py` (CPU; prints receipts),
`outputs/legacy_coverage_replay_20260925/{identity.json,coverage.json}`.
