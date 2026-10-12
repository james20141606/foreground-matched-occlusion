#!/usr/bin/env python3
"""Build MedTrust's combined anonymous audit supplement and replay it (same-code replay).

This script reads the source repository but writes only the requested ZIP (with
timestamped backup if needed). ``--verify-root`` runs three analyses from an
extracted package and checks both summary values and per-QA/image contributions.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "supplement/medtrust_audit_20260923.zip"
FIXED_DATE = (2026, 9, 23, 0, 0, 0)
SELF = "code/scripts/build_audit_supplement_20260923.py"
FG_SCRIPT = "code/scripts/image_cluster_audit_20260922.py"
IDENTITY_SCRIPT = "code/scripts/compare_legacy_slake_identity_20260923.py"
PAIRED_SCRIPT = "code/scripts/paired_legacy_foreground_34_20260923.py"
FG_MAP = "outputs/image_cluster_20260922/image_mapping.json"
OLD_MAP = "outputs/legacy_mapping_20260923/general_identity_mapping.json"
TRAIN = "outputs/legacy_mapping_20260923/source_train_a9083ce6.json"
PROTOCOL = "outputs/legacy_mapping_20260923/paired34_protocol.md"
FG_REFERENCE = "outputs/image_cluster_20260922/image_cluster_sensitivity.json"
IDENTITY_REFERENCE = "outputs/legacy_mapping_20260923/overlap_identity_check.json"
PAIRED_REFERENCE = "outputs/legacy_mapping_20260923/paired34_stored_mask_descriptive.json"
CORRECTED = "outputs/region_tv_corrected.json"
EXPECTED = "audit_expected.json"
# Compose detector literals so the scanner can inspect this very script
# without mistaking its rule definitions for leaked identities.
_PRIVATE_FOLDER = "/private/" + "var/folders/"
# Account names, institutions and other identity tokens are NOT written into this
# script (it ships inside the anonymous package). The builder reads them from the
# environment variable MEDTRUST_PRIVATE_TOKENS (comma-separated, set only on the
# build machine); the packaged verifier replays the path scan below without them.
_EXTRA_TOKENS = [t.strip() for t in os.environ.get("MEDTRUST_PRIVATE_TOKENS", "").split(",")
                 if t.strip()]
PRIVATE = re.compile(
    r"(?:/(?:Users|home|scratch)/[^\s\"']+|" + re.escape(_PRIVATE_FOLDER)
    + r"[^\s\"']+|[A-Za-z]:\\Users\\[^\s\"']+"
    + "".join(r"|\b" + re.escape(t) + r"\b" for t in _EXTRA_TOKENS) + ")", re.I
)
FG_NAMES = (
    "gemma3_4b", "internvl3_8b", "medgemma27b", "medgemma4b",
    "qwen25vl7b", "qwen2vl7b",
)
OLD_NAMES = ("gemma3_4b", "internvl3_8b", "qwen25vl7b", "qwen2vl7b")
FG_RECEIPTS = tuple(f"results/cipe_scale/bbox_fg_{m}_n400.json" for m in FG_NAMES)
OLD_RECEIPTS = tuple(
    ("outputs/cipe_v0" if m == "qwen2vl7b" else "results/cipe_scale")
    + f"/bbox_val_{m}_n400.json" for m in OLD_NAMES
)
SCRIPTS = (
    SELF, FG_SCRIPT, IDENTITY_SCRIPT, PAIRED_SCRIPT,
    "code/scripts/recover_image_ids_20260922.py",
    "code/scripts/export_legacy_slake_identity_20260923.py",
    "code/scripts/region_tv_corrected.py",
)
# 2026-09-24: MedGemma-4B legacy-vs-matched paired contrast on the same 229 QA
# (App. S4.1). Its legacy receipt shares the matched run's sample sequence.
MG4B_LEGACY = "results/cipe_scale/bbox_val_medgemma4b_n400.json"
MG4B_GATE_SCRIPT = "code/scripts/medgemma4b_legacy_matched_identity_20260924.py"
MG4B_SCRIPT = "code/scripts/medgemma4b_legacy_vs_matched_20260924.py"
MG4B_SAT_SCRIPT = "code/scripts/medgemma4b_paired_saturation_descriptive_20260924.py"
MG4B_SEED_SCRIPT = "code/scripts/medgemma4b_paired_registered_seed_20260924.py"
MG4B_RUN = "outputs/medgemma4b_paired_20260924"
MG4B_RECEIPTS = tuple(f"{MG4B_RUN}/{n}.json" for n in (
    "identity_gate", "paired_contrast", "saturation_descriptive", "signflip_registered_seed"))
MG4B_TEXT = (MG4B_GATE_SCRIPT, MG4B_SCRIPT, MG4B_SAT_SCRIPT, MG4B_SEED_SCRIPT,
             "code/medtrust/__init__.py", "code/medtrust/types.py", "code/medtrust/cipe.py",
             "notes/PREREG_medgemma4b_legacy_vs_matched_20260924.md")
# 2026-09-26 addendum (hash-checked by the verifier, not replayed): every registration the
# paper cites, the legacy-coverage replay and ChestX-Det receipts, and MedTrust-IN v0.3 text.
ADD_TEXT = (
    "PREREG_contamination_specificity_control.md", "PREREG_exposure_clean_region.md",
    "PREREG_region_exposure_consequence.md", "PREREG_region_metric_comparison.md",
    "PREREG_region_tv_correction.md", "notes/PREREG_legacy_coverage_replay_20260925.md",
    "code/scripts/legacy_coverage_replay_20260925.py", "code/scripts/legacy_replay_gate_power_20260925.py",
    "code/scripts/expclean_receipts.py", "code/scripts/exposure_clean_read.py",
    "data/medtrust_in_test/DATASHEET.md", "code/scripts/rehydrate_pathvqa_questions.py",
    "notes/PREREG_region_instrument_20260926.md", "code/scripts/region_instrument_analysis_20260926.py",
)
# 2026-09-26 license pass: PathVQA question text is not redistributed (caption copyright stays
# with the source publishers); those items ship by reference with a question hash.
CORPUS = "data/medtrust_in_test/medtrust_in_v0.jsonl"
LICENSE_FILES = {"LICENSES/ChestX-Det-Apache-2.0.txt": "supplement/licenses/ChestX-Det-Apache-2.0.txt"}
ADD_JSON = (
    "outputs/legacy_coverage_replay_20260925/identity.json",
    "outputs/legacy_coverage_replay_20260925/coverage.json",
    "outputs/legacy_coverage_replay_20260925/gate_power_posthoc.json",
    "data/medtrust_in_test/medtrust_in_stats.json",
    "results/region_instrument_20260926/receipt_only.json",
    *(f"results/exposure_clean/expclean_{m}.json" for m in
      ("gemma3_4b", "internvl3_8b", "medgemma4b", "qwen25vl_7b", "qwen2vl_7b")),
    *(f"results/exposure_clean/bbox_fg_{m}_chestxdet.json" for m in
      ("gemma3_4b", "internvl3_8b", "medgemma4b", "qwen25vl_7b", "qwen2vl_7b")),
)
PREREG_COMMITS = "notes/PREREG_COMMITS.json"
# Revision R1 (2026-10-07): corrected exposure audit (paper App. S1), Monte Carlo seed
# sensitivity (App. S5 / S4.1), receipt provenance table. The two official split files are
# byte-immutable (route A checks their sha256); text files are path-scrubbed if needed.
R1_EXPOSURE = "results/exposure_audit_corrected_20261004"
R1_ROUTE_A = "code/scripts/exposure_route_a_mapping_20261004.py"
R1_RAW = (f"{R1_EXPOSURE}/slake_official/validate.json", f"{R1_EXPOSURE}/slake_official/test.json")
R1_TEXT = (
    "notes/PREREG_exposure_recompute_20261004.md", f"{R1_EXPOSURE}/README.md",
    f"{R1_EXPOSURE}/slake_official/SOURCES.md", R1_ROUTE_A,
    "code/scripts/exposure_route_b_fingerprint_20261004.py", "code/scripts/region_exposure_audit.py",
    "code/scripts/seed_sensitivity_20261004.py", "code/scripts/receipt_provenance_20261007.py",
)
R1_JSON = (
    f"{R1_EXPOSURE}/COMPARE.json", f"{R1_EXPOSURE}/route_a_mapping.json",
    f"{R1_EXPOSURE}/route_b_fingerprint.json", f"{R1_EXPOSURE}/region_exposure_audit_corrected.json",
    "results/seed_sensitivity_20261004/summary.json",
    "results/receipt_provenance_20261007/provenance.json",
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read(root: Path, rel: str) -> bytes:
    return (root / rel).read_bytes()


def canonical(obj: object) -> bytes:
    return (json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def scan(rel: str, data: bytes) -> None:
    try:
        decoded = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"non-UTF8 package member {rel}") from exc
    hit = PRIVATE.search(decoded)
    if hit:
        raise ValueError(f"private path/account marker remains in {rel}: {hit.group(0)[:40]}")


def scrub_object(obj: object) -> object:
    """Redact private metadata only in package copies; numeric arrays stay intact."""
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            safe_key = ("REDACTED_KEY_" + sha(key.encode())[:16]) if PRIVATE.search(key) else key
            if safe_key in result:
                raise ValueError("redaction key collision")
            result[safe_key] = scrub_object(value)
        return result
    if isinstance(obj, list):
        return [scrub_object(x) for x in obj]
    if isinstance(obj, str) and PRIVATE.search(obj):
        return "REDACTED_PRIVATE_VALUE_SHA256_" + sha(obj.encode())
    return obj


def package_json(raw: bytes) -> tuple[bytes, bool]:
    obj = json.loads(raw)
    safe = scrub_object(obj)
    changed = safe != obj
    return (canonical(safe) if changed else raw), changed


def tv(a: object, b: object) -> float:
    x, y = np.asarray(a, float), np.asarray(b, float)
    if x.sum() <= 0 or y.sum() <= 0:
        raise ValueError("nonpositive probability vector")
    return float(0.5 * np.abs(x / x.sum() - y / y.sum()).sum())


def cluster_contributions(root: Path) -> dict:
    """Independent item/image ledger from the stored vectors, not source images."""
    mapping = json.loads(read(root, FG_MAP))
    lookup = {r["sample_id"]: r for r in mapping["items"]}
    out = {}
    for name, rel in zip(FG_NAMES, FG_RECEIPTS):
        receipt = json.loads(read(root, rel))
        qa = []
        by_image: dict[str, list[dict]] = {}
        for row in sorted(receipt["rows"], key=lambda x: int(x["sample_id"].split("_")[-1])):
            item = lookup[row["sample_id"]]
            target = tv(row["p_original"], row["p_organ"])
            ctl = [tv(row["p_original"], p) for p in row["p_control_draws"]]
            if len(ctl) != 5 or item["organ_boxes"] != row["organ_boxes"]:
                raise ValueError(f"control/identity mismatch in {rel}")
            wins = [float(target > x) for x in ctl]
            ties = [float(target == x) for x in ctl]
            record = {
                "sample_id": row["sample_id"], "image_ref": item["image_ref"],
                "image_sha256": item["image_sha256"],
                "W_strict": float(np.mean(wins)),
                "W_half_ties": float(np.mean(np.asarray(wins) + 0.5 * np.asarray(ties))),
                "D": float(target - np.mean(ctl)),
                "tie_fraction": float(np.mean(ties)),
            }
            qa.append(record)
            by_image.setdefault(item["image_sha256"], []).append(record)
        images = [{
            "image_sha256": image, "n_qa": len(rows),
            "sample_ids": [r["sample_id"] for r in rows],
            "W_strict_qa_mean": float(np.mean([r["W_strict"] for r in rows])),
            "W_half_ties_qa_mean": float(np.mean([r["W_half_ties"] for r in rows])),
            "D_qa_mean": float(np.mean([r["D"] for r in rows])),
            "tie_fraction_qa_mean": float(np.mean([r["tie_fraction"] for r in rows])),
        } for image, rows in sorted(by_image.items())]
        if len(qa) != 229 or len(images) != 175:
            raise ValueError(f"foreground cohort changed for {name}")
        out[name] = {"qa_contributions": qa, "image_contributions": images}
    return out


def identity_projection(payload: dict) -> dict:
    keys = (
        "identity_rule", "limit", "old_candidates", "new_candidates",
        "old_unique_qa_keys", "new_unique_qa_keys", "shared_one_to_one_qa",
        "shared_ordered_boxes_equal", "shared_foreground_included",
        "same_sample_id_same_qa", "shared_id_pairs",
    )
    result = {k: payload[k] for k in keys}
    result["model_checks"] = {m: {k: v for k, v in info.items()
                                   if k not in {"old_receipt_sha256", "foreground_receipt_sha256"}}
                              for m, info in payload["model_checks"].items()}
    return result


def paired_projection(payload: dict) -> dict:
    keys = ("scope", "n_shared_candidate_qa", "n_selected_shared_qa",
            "n_images", "qa_counts_by_image", "bootstrap")
    result = {k: payload[k] for k in keys}
    result["models"] = {m: {k: v for k, v in info.items()
                            if k not in {"old_receipt_sha256", "foreground_receipt_sha256"}}
                        for m, info in payload["models"].items()}
    return result


def corpus_by_reference(raw: bytes) -> bytes:
    """Replace each PathVQA item's question with its SHA-256; every other field is kept."""
    out, n = [], 0
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        it = json.loads(line)
        if it["dataset"] == "pathvqa":
            it = {("question_sha256" if k == "question" else k):
                  (sha(v.encode("utf-8")) if k == "question" else v) for k, v in it.items()}
            n += 1
        out.append(json.dumps(it, ensure_ascii=False))
    if n != 916:
        raise ValueError(f"expected 916 PathVQA items, found {n}")
    return ("\n".join(out) + "\n").encode("utf-8")


def prereg_commits(root: Path) -> dict:
    """First and last repository commit (hash, UTC time) of each registration and of the
    receipts it governs. Built from git on the build machine; shipped as data."""
    def log(rel: str) -> list[str]:
        out = subprocess.run(["git", "-C", str(root), "log", "--follow", "--format=%H %ct", "--", rel],
                             capture_output=True, text=True, check=True).stdout.split("\n")
        return [l for l in out if l.strip()]
    def utc(ts: str) -> str:
        return dt.datetime.fromtimestamp(int(ts), dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = {}
    for rel in (*[r for r in ADD_TEXT if "PREREG_" in r], "notes/PREREG_medgemma4b_legacy_vs_matched_20260924.md",
                "outputs/legacy_coverage_replay_20260925/coverage.json",
                "results/exposure_clean/expclean_medgemma4b.json",
                "outputs/medgemma4b_paired_20260924/paired_contrast.json",
                "results/region_instrument_20260926/receipt_only.json"):
        lines = log(rel)
        if not lines:
            raise ValueError(f"no commit history for {rel}")
        first, last = lines[-1].split(), lines[0].split()
        rows[rel] = {"first_commit": first[0], "first_commit_utc": utc(first[1]),
                     "last_commit": last[0], "last_commit_utc": utc(last[1]), "n_commits": len(lines)}
    return {"note": "Commits of the private project repository; hashes identify the committed "
                    "bytes, they are not an external registration. Receipt rows show when the "
                    "governed results were first committed.", "files": rows}


def readme() -> bytes:
    return ("""# MedTrust combined anonymous audit supplement

This is **post-hoc, receipt-only** analysis, not a new blinded registration,
clinical test, or model-inference rerun. No source image pixels or model weights
are packaged. The exact official SLAKE `train.json` revision is included byte
for byte; it is metadata, not an image archive.

There are two distinct populations. The foreground-only normalized-TV audit
uses **229 QA on 175 recovered image hashes** and six models, with strict-win,
half-tie and image-cluster sensitivity. The legacy comparison uses exact
`(image_ref, question, normalized answer)` one-to-one identities: only **62**
candidate QA overlap old/new loaders, and **34 QA on 33 images** are in the
foreground inclusion set, across four common general models. The old claim
that all 229 foreground QA were paired with old same-numbered `slake_i` items
is **withdrawn**. Paired34 describes only stored old one-draw and new five-draw
masks; it cannot identify a general control effect on 229 or 400 QA.
The legacy mapping retains a historical MedGemma4B receipt hash as provenance:
that run followed the matched run's loader branch, so it is not one of the four
common general models. Its legacy receipt IS included (added 2026-09-24) for a
separate, post-hoc paired contrast on the same 229 QA (paper App. S4.1): an
outcome-blind identity gate, the registered analysis
(`notes/PREREG_medgemma4b_legacy_vs_matched_20260924.md`), a post-hoc saturation
description and a registered-seed sign-flip recomputation (the analysis script
used seed 20260925; the registration fixes 20260924). The readout is CIPE because
the legacy receipt stores no answer distributions.

The image map came from later available SLAKE metadata and matching annotation
fingerprints; an excluded ambiguous `slake_246` is unresolved. Historical
receipt QA text/pixels were not stored, so source-revision replay and exact
shared-case CIPE agreement corroborate, but cannot prove, every historical
QA/pixel identity. The exported identity-recovery scripts are included for
inspection; they need the original image and detection trees and **cannot**
rebuild the mapping offline from this ZIP alone.

## Offline replay

Python 3.10+ and NumPy suffice. From the extracted ZIP root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 code/scripts/build_audit_supplement_20260923.py --verify-root .
```

The verifier is a **same-code replay**: it re-runs the packaged scripts on the
packaged receipts and compares the result with their stored outputs. It shows that
the shipped numbers follow from the shipped receipts and code; it is not an
independent implementation. Scripts replayed: `image_cluster_audit_20260922.py`,
`compare_legacy_slake_identity_20260923.py`,
`paired_legacy_foreground_34_20260923.py`, and the four MedGemma-4B scripts
`medgemma4b_*_20260924.py` (identity gate, paired contrast, then the saturation
and registered-seed scripts chained on the replayed paired receipt). It checks all numerical summary
objects and compares every per-QA and per-image contribution. Recomputed files
are written only under `replay/` in the extracted directory. The manifest
records each source original SHA-256 and packaged SHA-256; package copies may
redact private paths/accounts, and mapping receipt hashes are updated to the
packaged copies. The official `source_train_a9083ce6.json` is not reserialized.

For the 175-image analysis, QA-weighted strict-win W is primary; image-equal W,
half ties and image-cluster sign flips are sensitivities. The sign-flip p-values
require symmetry of image-level centered sums; strict-win ties complicate a
chance-0.5 null, so the half-tie analysis is reported separately. Paired34
reports percentile 95% and four-model Bonferroni 98.75% intervals, no p-values.

## Addendum (2026-09-26): registrations, replay and ChestX-Det receipts, corpus

These files are hash-checked by the verifier but not re-run (the replays need source
images, which are not redistributed):

- every registration the paper cites (`PREREG_*.md`, `notes/PREREG_*.md`), with the first and
  last repository commit of each, and of the receipts they govern, in
  `notes/PREREG_COMMITS.json` (hashes of a private repository, not an external registry);
- the legacy-placement coverage replay (paper App. S4.2, Table 5):
  `outputs/legacy_coverage_replay_20260925/` and its two scripts;
- the exploratory ChestX-Det run (paper App. S2): per-item receipts
  `results/exposure_clean/bbox_fg_*_chestxdet.json`, the read-outs `expclean_*.json` and
  the scripts `exposure_clean_read.py`, `expclean_receipts.py`;
- the receipt-only instrument checks (paper App. S5.1, Table 7), registered as descriptive in
  `notes/PREREG_region_instrument_20260926.md` before they ran: a placebo (one matched draw
  against the other four) and tolerance tightening on the stored draws,
  `results/region_instrument_20260926/receipt_only.json` and
  `code/scripts/region_instrument_analysis_20260926.py` (reads the six packaged
  `results/cipe_scale/bbox_fg_*_n400.json`; the GPU analyses of that registration have not run).
  Re-running `--receipt_only` with the packaged script reproduces every value of the receipt;
  the receipt's `generator_sha256` records an earlier revision of the same file;
- MedTrust-IN v0.3 (`data/medtrust_in_test/`): the 1,122 items as JSON lines (answer, source
  split and deterministic `image_ref`; question text for the 206 SLAKE and VQA-RAD items; the
  916 PathVQA items by reference, see below; no image pixels), its statistics and datasheet.

## Addendum (2026-10-07, revision R1): corrected exposure audit, seed sensitivity, provenance

- **Exposure audit (paper App. S1), corrected.** 211 of the 229 region items (92.1%), not 213
  (93.0%), sit on images in the SLAKE train split; 62 (not 52) on images in both train and
  test; 3 (not 4) test-only. `results/exposure_audit_corrected_20261004/` holds both routes
  (A: the image mapping joined to the official split files; B: a box-and-size fingerprint),
  the repaired generator's receipt and `COMPARE.json` (229/229 agreement); `README.md` there
  states the cause. The verifier re-runs route A
  (`code/scripts/exposure_route_a_mapping_20261004.py`, standard library only) on the packaged
  mapping, receipt and official split files (`slake_official/validate.json`, `test.json`,
  byte-identical, sha256-checked; `train.json` is the file above). Route B and the generator
  need the SLAKE image tree and are shipped for inspection only. Registration:
  `notes/PREREG_exposure_recompute_20261004.md`.
- **Monte Carlo seed sensitivity** (paper App. S5 and S4.1):
  `results/seed_sensitivity_20261004/summary.json` and its script (hash-checked, not re-run).
- **Receipt provenance** (paper Table 8): `results/receipt_provenance_20261007/provenance.json`
  and its generator, which needs the git history and is shipped for inspection only.
- **PathVQA rehydration**: `rehydrate_pathvqa_questions.py` now loads each split once and pins the
  mirror revision (`1685832883334b5bb5beaf4e4b333fdeecaa4ad9`, the mirror's last commit,
  2023-06-03). The first run downloads the whole mirror, about 6 GB including images.

## Third-party data in this supplement: licenses and attribution

This supplement redistributes question/answer text and annotation metadata (no image pixels)
from public datasets. Each remains under its original license; our derived files (receipts,
mappings, MedTrust-IN v0.3 item lists) are modified or derived works. Licenses were checked
over https on 2026-09-26 (Hugging Face dataset cards, the OSF API, the GitHub API).

- **SLAKE** (Liu et al., 2021, "SLAKE: A Semantically-Labeled Knowledge-Enhanced Dataset for
  Medical Visual Question Answering", ISBI 2021). License: Creative Commons Attribution 4.0
  International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/, as stated by the
  authors at https://huggingface.co/datasets/BoKelvin/SLAKE. Files:
  `outputs/legacy_mapping_20260923/source_train_a9083ce6.json` (the official `train.json`,
  unmodified); SLAKE questions, answers and `detection.json` boxes inside
  `outputs/image_cluster_20260922/image_mapping.json` and
  `outputs/legacy_mapping_20260923/general_identity_mapping.json` (excerpted and re-keyed); the 121 SLAKE items of
  `data/medtrust_in_test/medtrust_in_v0.jsonl` (selected, with added fields).
- **VQA-RAD** (Lau et al., 2018, Scientific Data 5:180251). License: CC0 1.0 Universal, the
  license of the authors' OSF release (https://doi.org/10.17605/OSF.IO/89KPS). Files: the 85
  VQA-RAD items of `medtrust_in_v0.jsonl` (selected, reformatted, with added fields).
- **PathVQA** (He et al., 2020, arXiv:2003.10286). Recorded as MIT-licensed by the public
  mirror https://huggingface.co/datasets/flaviagiammarino/path-vqa; the official repository
  (github.com/UCSD-AI4H/PathVQA) returned 404 on 2026-09-26, so the original redistribution
  terms could not be verified at source. Per that dataset card, copyright in the source images and in
  the captions from which the questions were generated belongs to the publishers of the two
  source textbooks and the owners of the PEIR digital library. We therefore do **not**
  redistribute PathVQA question text: the 916 PathVQA items of `medtrust_in_v0.jsonl` keep
  `item_id`, `image_ref`, the yes/no answer and the screening fields, and replace `question`
  with `question_sha256`. `code/scripts/rehydrate_pathvqa_questions.py` restores the text from
  the public mirror and stops on any hash or answer mismatch.
- **ChestX-Det** (Lian et al., 2021), a re-annotated subset of NIH ChestX-ray14 (Wang et al.,
  2017). Annotations are released under the Apache License 2.0
  (https://github.com/Deepwise-AILab/ChestX-Det-Dataset; copy in
  `LICENSES/ChestX-Det-Apache-2.0.txt`). Files: sample identifiers, finding labels and box
  coordinates inside `results/exposure_clean/bbox_fg_*_chestxdet.json`. No ChestX-ray14 pixels
  are included.

No image pixels from any source are redistributed; images must be obtained from the original
releases.
""").encode("utf-8")


def route_a_summary(root: Path) -> dict:
    """Run route A of the exposure audit (standard library only) on `root` in a subprocess
    and return its summary without per-item rows. Used at build time on the repository and
    at verify time on the extracted package; the two must agree."""
    r = subprocess.run([sys.executable, "-c",
                        "import json,sys; sys.path.insert(0, 'code/scripts'); "
                        "import exposure_route_a_mapping_20261004 as a; r = a.compute('.'); "
                        "r.pop('per_item'); print(json.dumps(r, sort_keys=True))"],
                       cwd=root, capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def build_entries(root: Path) -> tuple[dict[str, bytes], dict]:
    source_bytes: dict[str, bytes] = {}
    packaged: dict[str, bytes] = {}
    origins: dict[str, dict] = {}

    def add(rel: str, data: bytes, original: bytes, redacted: bool = False) -> None:
        packaged[rel] = data
        origins[rel] = {"original_unscrubbed_sha256": sha(original),
                        "packaged_sha256": sha(data), "redacted_or_rebased": redacted,
                        "bytes": len(data)}

    core = (*SCRIPTS, CORRECTED, TRAIN, PROTOCOL)
    for rel in core:
        raw = read(root, rel)
        source_bytes[rel] = raw
        # The official source revision is byte-immutable by design.
        if rel == TRAIN:
            expected_train_sha = "7a9fb81f9d7bb145c3d7e9bc8d3b9016ba18b6539ea5031a53bae8fd3cd991c3"
            if sha(raw) != expected_train_sha:
                raise ValueError("official SLAKE train revision changed")
            add(rel, raw, raw)
        elif rel.endswith(".json"):
            safe, changed = package_json(raw)
            add(rel, safe, raw, changed)
        else:
            add(rel, raw, raw)

    for rel in MG4B_TEXT:
        raw = read(root, rel)
        source_bytes[rel] = raw
        add(rel, raw, raw)
    for rel in (*FG_RECEIPTS, *OLD_RECEIPTS, MG4B_LEGACY, *MG4B_RECEIPTS):
        raw = read(root, rel)
        source_bytes[rel] = raw
        safe, changed = package_json(raw)
        add(rel, safe, raw, changed)

    for rel in ADD_TEXT:
        raw = read(root, rel)
        add(rel, raw, raw)
    for rel in ADD_JSON:
        raw = read(root, rel)
        safe, changed = package_json(raw)
        add(rel, safe, raw, changed)
    for rel in R1_RAW:
        raw = read(root, rel)
        add(rel, raw, raw)
    for rel in R1_TEXT:
        raw = read(root, rel)
        txt = raw.decode("utf-8")
        safe = PRIVATE.sub("<private path redacted>", txt).encode("utf-8")
        add(rel, safe, raw, safe != raw)
    for rel in R1_JSON:
        raw = read(root, rel)
        safe, changed = package_json(raw)
        add(rel, safe, raw, changed)
    raw = read(root, CORPUS)
    add(CORPUS, corpus_by_reference(raw), raw, True)
    for rel, src in LICENSE_FILES.items():
        raw = read(root, src)
        add(rel, raw, raw)
    commits = prereg_commits(root)
    add(PREREG_COMMITS, canonical(commits), canonical(commits))

    fg_raw, old_raw = read(root, FG_MAP), read(root, OLD_MAP)
    source_bytes[FG_MAP], source_bytes[OLD_MAP] = fg_raw, old_raw
    fg_map, old_map = json.loads(fg_raw), json.loads(old_raw)
    fg_checks = {r["file"]: r for r in fg_map["receipt_checks"]}
    if set(fg_checks) != set(FG_RECEIPTS):
        raise ValueError("foreground mapping receipt roster changed")
    for rel in FG_RECEIPTS:
        if fg_checks[rel]["sha256"] != sha(source_bytes[rel]):
            raise ValueError(f"foreground source hash mismatch: {rel}")
        fg_checks[rel]["sha256"] = sha(packaged[rel])
    if fg_map["n_candidates"] != 400 or fg_map["ambiguous_candidate_ids"] != ["slake_246"]:
        raise ValueError("foreground mapping cohort changed")
    fg_map = scrub_object(fg_map)
    fg_bytes = canonical(fg_map)
    add(FG_MAP, fg_bytes, fg_raw, fg_bytes != fg_raw)

    old_checks = old_map["receipt_checks"]
    for rel in OLD_RECEIPTS:
        if old_checks[rel]["sha256"] != sha(source_bytes[rel]):
            raise ValueError(f"legacy source hash mismatch: {rel}")
        old_checks[rel]["sha256"] = sha(packaged[rel])
    if old_map["train_sha256"] != sha(packaged[TRAIN]) or len(old_map["items"]) != 400:
        raise ValueError("legacy mapping source revision or cohort changed")
    old_map = scrub_object(old_map)
    old_bytes = canonical(old_map)
    add(OLD_MAP, old_bytes, old_raw, old_bytes != old_raw)

    original_fg = json.loads(read(root, FG_REFERENCE))
    if original_fg["generator_sha256"] != sha(source_bytes[FG_SCRIPT]):
        raise ValueError("foreground generator/reference drift")
    if original_fg["mapping_sha256"] != sha(fg_raw):
        raise ValueError("foreground mapping/reference drift")
    for rel in FG_RECEIPTS:
        if original_fg["receipt_sha256"][rel] != sha(source_bytes[rel]):
            raise ValueError(f"foreground reference receipt drift: {rel}")
    identity = json.loads(read(root, IDENTITY_REFERENCE))
    paired = json.loads(read(root, PAIRED_REFERENCE))
    if identity["old_mapping_sha256"] != sha(old_raw) or identity["new_mapping_sha256"] != sha(fg_raw):
        raise ValueError("identity reference mapping drift")
    if paired["old_mapping_sha256"] != sha(old_raw) or paired["new_mapping_sha256"] != sha(fg_raw):
        raise ValueError("paired reference mapping drift")
    if paired["source_train_sha256"] != sha(source_bytes[TRAIN]) or paired["protocol_sha256"] != sha(source_bytes[PROTOCOL]):
        raise ValueError("paired reference source/protocol drift")
    if identity["shared_one_to_one_qa"] != 62 or identity["shared_foreground_included"] != 34:
        raise ValueError("identity reference cohort drift")
    if paired["n_selected_shared_qa"] != 34 or paired["n_images"] != 33:
        raise ValueError("paired reference cohort drift")

    expected = {
        "status": "source-derived expected values; receipt hashes rebased in package",
        "foreground": {"results": original_fg["results"], "n_qa": original_fg["n_qa"],
                       "n_images": original_fg["n_images"],
                       "qa_per_image": original_fg["qa_per_image"],
                       "seed": original_fg["seed"],
                       "resamples": original_fg["resamples"],
                       "primary_estimand": original_fg["primary_estimand"],
                       "test_assumption": original_fg["test_assumption"],
                       "qa_image_contributions": cluster_contributions(root)},
        "identity": identity_projection(identity),
        "paired34": paired_projection(paired),
        "exposure_route_a": route_a_summary(root),
    }
    if (expected["exposure_route_a"]["n_items"], expected["exposure_route_a"]["n_on_train_images"],
            expected["exposure_route_a"]["n_test_and_train"], expected["exposure_route_a"]["n_test_only"]) \
            != (229, 211, 62, 3):
        raise ValueError("exposure route A no longer gives the corrected 229/211/62/3")
    add(EXPECTED, canonical(expected), canonical(expected))
    add("README.md", readme(), readme())

    source_reference_sha = {rel: sha(read(root, rel)) for rel in
                            (FG_REFERENCE, IDENTITY_REFERENCE, PAIRED_REFERENCE)}
    manifest = {
        "package": "MedTrust combined anonymous region-audit supplement",
        "status": "post-hoc; 229 QA/175 images foreground internal, 34 QA/33 images paired legacy intersection",
        "official_train_sha256": sha(source_bytes[TRAIN]),
        "original_reference_sha256": source_reference_sha,
        "files": {rel: origins[rel] for rel in sorted(packaged)},
        "no_raw_images_or_model_weights": True,
    }
    add("manifest.json", canonical(manifest), canonical(manifest))
    for rel, data in packaged.items():
        scan(rel, data)
    return packaged, manifest


def verify_root(root: Path) -> dict:
    manifest = json.loads(read(root, "manifest.json"))
    expected = json.loads(read(root, EXPECTED))
    if manifest["official_train_sha256"] != sha(read(root, TRAIN)):
        raise ValueError("official source train bytes changed in extracted package")
    for rel, info in manifest["files"].items():
        data = read(root, rel)
        if sha(data) != info["packaged_sha256"] or len(data) != info["bytes"]:
            raise ValueError(f"extracted member mismatch: {rel}")
        scan(rel, data)
    scan("manifest.json", read(root, "manifest.json"))
    fg_map = json.loads(read(root, FG_MAP))
    old_map = json.loads(read(root, OLD_MAP))
    for check in fg_map["receipt_checks"]:
        if check["sha256"] != sha(read(root, check["file"])):
            raise ValueError(f"packaged foreground map hash mismatch: {check['file']}")
    for rel in OLD_RECEIPTS:
        if old_map["receipt_checks"][rel]["sha256"] != sha(read(root, rel)):
            raise ValueError(f"packaged legacy map hash mismatch: {rel}")
    replay = root / "replay"
    replay.mkdir(exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    commands = (
        (FG_SCRIPT, ("--root", str(root), "--mapping", str(root / FG_MAP)), "image_cluster.json"),
        (IDENTITY_SCRIPT, ("--repo", str(root)), "legacy_identity.json"),
        (PAIRED_SCRIPT, ("--repo", str(root)), "paired34.json"),
    )
    for script, args, name in commands:
        subprocess.run([sys.executable, str(root / script), *args, "--out", str(replay / name)],
                       cwd=root, env=env, check=True, capture_output=True, text=True)
    fg = json.loads(read(root, "replay/image_cluster.json"))
    identity = json.loads(read(root, "replay/legacy_identity.json"))
    paired = json.loads(read(root, "replay/paired34.json"))
    if fg["results"] != expected["foreground"]["results"]:
        raise ValueError("foreground six-model numerical replay differs")
    for field in ("n_qa", "n_images", "qa_per_image", "seed", "resamples",
                  "primary_estimand", "test_assumption"):
        if fg[field] != expected["foreground"][field]:
            raise ValueError(f"foreground cohort/statistic drift: {field}")
    if cluster_contributions(root) != expected["foreground"]["qa_image_contributions"]:
        raise ValueError("foreground per-QA/image contribution replay differs")
    if identity_projection(identity) != expected["identity"]:
        raise ValueError("legacy identity and per-QA pair replay differs")
    if paired_projection(paired) != expected["paired34"]:
        raise ValueError("paired34 numbers or per-QA/image contributions differ")
    mg4b_replay(root, replay, env)
    if route_a_summary(root) != expected["exposure_route_a"]:
        raise ValueError("exposure route A replay differs (App. S1: 211/229, 62, 3)")
    return {"checks": ["exposure route A (229 items: 211 on train images, 62 train+test, 3 test-only)","manifest hashes", "identity scan", "official train raw hash",
                       "six-model image-cluster numerical + 229QA/175image contributions",
                       "legacy 62/34 identity + per-QA pairs",
                       "paired34 four-model numerical + 34QA/33image contributions",
                       "MedGemma-4B 229-QA identity gate + paired contrast (all rows) + "
                       "saturation + registered-seed sign-flip"],
            "foreground_n_qa": fg["n_qa"], "foreground_n_images": fg["n_images"],
            "shared_candidate_qa": identity["shared_one_to_one_qa"],
            "paired_n_qa": paired["n_selected_shared_qa"],
            "paired_n_images": paired["n_images"]}


def _no_sha(obj: object) -> object:
    if isinstance(obj, dict):
        # "inputs" maps paths to SHA-256 values of (possibly re-serialised) inputs
        return {k: _no_sha(v) for k, v in obj.items()
                if not k.endswith("sha256") and k != "inputs"}
    if isinstance(obj, list):
        return [_no_sha(x) for x in obj]
    return obj


def mg4b_replay(root: Path, replay: Path, env: dict) -> None:
    """Re-run the MedGemma-4B chain from packaged scripts and compare every
    non-hash field with the packaged receipts (hashes differ only where a
    package copy was redacted)."""
    def run(script: str, *args: str) -> None:
        subprocess.run([sys.executable, str(root / script), *args], cwd=root, env=env,
                       check=True, capture_output=True, text=True)
    gate_out, pc_out = replay / "mg4b_identity_gate.json", replay / "mg4b_paired_contrast.json"
    run(MG4B_GATE_SCRIPT, "--repo", str(root), "--out", str(gate_out))
    run(MG4B_SCRIPT, "--repo", str(root), "--out", str(pc_out))
    tree = replay / "mg4b_tree"
    for rel in (MG4B_SAT_SCRIPT, MG4B_SEED_SCRIPT, MG4B_LEGACY):
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / rel, tree / rel)
    (tree / MG4B_RUN).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pc_out, tree / MG4B_RUN / "paired_contrast.json")
    run(MG4B_SAT_SCRIPT, str(tree))
    run(MG4B_SEED_SCRIPT, str(tree))
    pairs = ((gate_out, MG4B_RECEIPTS[0]), (pc_out, MG4B_RECEIPTS[1]),
             (tree / MG4B_RUN / "saturation_descriptive.json", MG4B_RECEIPTS[2]),
             (tree / MG4B_RUN / "signflip_registered_seed.json", MG4B_RECEIPTS[3]))
    for got, rel in pairs:
        a, b = _no_sha(json.loads(got.read_text())), _no_sha(json.loads(read(root, rel)))
        if a != b:
            diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            raise ValueError(f"MedGemma-4B replay differs for {rel}: {diff}")


def write_zip(out: Path, packaged: dict[str, bytes]) -> Path | None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="medtrust-audit-build-", dir="/private/tmp") as tmp:
        temp = Path(tmp) / out.name
        with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for rel, data in sorted(packaged.items()):
                entry = zipfile.ZipInfo(rel, date_time=FIXED_DATE)
                entry.create_system = 3
                entry.external_attr = 0o644 << 16
                entry.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(entry, data, compress_type=zipfile.ZIP_DEFLATED,
                           compresslevel=9)
        extracted = Path(tmp) / "extracted"
        extracted.mkdir()
        with zipfile.ZipFile(temp) as z:
            if z.testzip() is not None:
                raise ValueError("ZIP CRC validation failed")
            for member in z.namelist():
                if PurePosixPath(member).is_absolute() or ".." in PurePosixPath(member).parts:
                    raise ValueError(f"unsafe ZIP path: {member}")
            z.extractall(extracted)
        verify_root(extracted)
        backup = None
        if out.exists():
            stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
            backup = out.with_name(f"{out.stem}.backup-{stamp}{out.suffix}")
            suffix = 2
            while backup.exists():
                backup = out.with_name(f"{out.stem}.backup-{stamp}-{suffix}{out.suffix}")
                suffix += 1
            out.replace(backup)
        shutil.copyfile(temp, out)
    return backup


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUTPUT)
    ap.add_argument("--verify-root", type=Path,
                    help="re-run the packaged scripts on an already extracted package "
                         "(same-code replay, not an independent implementation)")
    args = ap.parse_args()
    if args.verify_root is not None:
        print(json.dumps(verify_root(args.verify_root.resolve()), indent=2))
        return
    packaged, manifest = build_entries(ROOT)
    backup = write_zip(args.out, packaged)
    print(json.dumps({"zip": str(args.out), "sha256": sha(args.out.read_bytes()),
                      "entries": len(packaged), "source_files": len(manifest["files"]),
                      "backup": str(backup) if backup else None,
                      "replay": "image-cluster, identity, paired34 and MedGemma-4B chain: numerical and per-QA/image PASS"},
                     indent=2))


if __name__ == "__main__":
    main()
