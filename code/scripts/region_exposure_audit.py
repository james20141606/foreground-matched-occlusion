#!/usr/bin/env python3
"""Are the region-arm items on images the specialist family trained on?

Motivated by a reviewer-style audit of our own pipeline. MedGemma's technical
report (arXiv:2507.05201) lists SLAKE (450 examples) among its training data
(its Table 1) and, for its SLAKE evaluation, states "For SLAKE, we used the
default train/test split". That the 450 training examples came from the train
split is OUR INFERENCE; the report does not say which split they came from.

This script reconstructs the region arm's candidate pool (closed English yes/no
items, shuffled with random.Random(42), admitted by the SAME filter as the
foreground loader cipe_bbox_external_validation.load_slake_with_bbox) and
reports, for the items the foreground receipt RETAINED, how many sit on images
that appear in each official SLAKE split.

CORRECTION (2026-10-04, BootLoops audit F1; notes/PREREG_exposure_recompute_20261004.md).
The previous version admitted every item whose image directory had a
detection.json. The foreground loader also drops items whose detection.json
holds no well-formed box (62 SLAKE images have `[]`) or whose image is missing.
From the first such item (pool position 32) the two lists were offset, and 211
of the 229 retained sample_ids were looked up on the wrong image. Its guard
(`len(sel) != len(kept_idx)`) checked index bounds only, so it could not fire.
The published 213/229 (93.0%) came from that version; the corrected value is
211/229 (92.1%). The guard is now an identity check per retained item: the
reconstructed image's decoded size and detection.json boxes must equal the
receipt row's image_size and organ_boxes, and (with --mapping) its img_name
must equal the 09-22 fingerprint mapping's image_ref. Any mismatch exits 2.

  python code/scripts/region_exposure_audit.py --imgs <SLAKE imgs> \
      --split-source official --splits-dir <dir with train/validate/test.json> \
      --mapping outputs/image_cluster_20260922/image_mapping.json --out <json>
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import random
import subprocess
from collections import Counter
from pathlib import Path

IMGS_DEFAULT = "<private path redacted>"
POOL_N = 400


class IdentityError(SystemExit):
    def __init__(self, msg):
        super().__init__(2)
        self.msg = msg


def load_closed(imgs: Path, seed: int):
    recs = []
    for qf in sorted(imgs.glob("*/question.json")):
        try:
            recs.extend(json.loads(qf.read_text()))
        except Exception:
            continue
    closed = [r for r in recs if r.get("q_lang") == "en"
              and str(r.get("answer")).strip().lower() in {"yes", "no"}]
    random.Random(seed).shuffle(closed)
    return closed


def detection_boxes(imgs: Path, img_name: str):
    """The foreground loader's parse: list of 4-number boxes, or None if unreadable."""
    det = imgs / Path(img_name).parent / "detection.json"
    try:
        boxes = json.loads(det.read_text())
        pairs = [list(d.values())[0] for d in boxes if isinstance(d, dict) and d]
        return [b for b in pairs if isinstance(b, (list, tuple)) and len(b) == 4]
    except Exception:
        return None


def build_pool(imgs: Path, seed: int, legacy_filter: bool = False, n: int = POOL_N):
    """Candidate pool in the foreground loader's order.

    legacy_filter=True reproduces the pre-2026-10-04 defect (admit on detection.json
    existence alone); it is kept only so the plant test can show the identity check
    catches it. Never use it for a receipt."""
    cand = []
    for r in load_closed(imgs, seed):
        img_dir = imgs / Path(r["img_name"]).parent
        if not (img_dir / "detection.json").exists():
            continue
        if not legacy_filter:
            if not (imgs / r["img_name"]).exists():
                continue
            bbs = detection_boxes(imgs, r["img_name"])
            if not bbs:            # None (unparseable) or [] (no box): the loader drops both
                continue
        cand.append(r)
        if len(cand) >= n:
            break
    return cand


def identity_check(imgs: Path, cand, rows, mapping=None):
    """Per retained row: the pool entry at its sample_id must BE that row's image."""
    from PIL import Image
    by_id = {it["sample_id"]: it for it in (mapping or {}).get("items", [])}
    bad = []
    sel = []
    for row in rows:
        sid = row["sample_id"]
        i = int(sid.split("_")[1])
        if i >= len(cand):
            bad.append(f"{sid}: index {i} outside the {len(cand)}-item pool"); continue
        r = cand[i]
        boxes = detection_boxes(imgs, r["img_name"]) or []
        with Image.open(imgs / r["img_name"]) as im:
            size = list(im.size)
        if [list(map(float, b)) for b in boxes] != [list(map(float, b)) for b in row["organ_boxes"]] \
                or size != list(row["image_size"]):
            bad.append(f"{sid}: pool entry {r['img_name']} (size {size}, {len(boxes)} boxes) is not the "
                       f"receipt's image (size {row['image_size']}, {len(row['organ_boxes'])} boxes)")
            continue
        if by_id:
            ref = (by_id.get(sid) or {}).get("image_ref")
            if ref != r["img_name"]:
                bad.append(f"{sid}: pool entry {r['img_name']} != fingerprint mapping {ref}"); continue
        sel.append((sid, r))
    if bad:
        raise IdentityError(f"{len(bad)} of {len(rows)} retained items fail the identity check; "
                            f"first: {bad[:3]}")
    return sel


def load_splits(source: str, splits_dir: str | None):
    splits = {}
    if source == "official":
        files = {"train": "train.json", "validation": "validate.json", "test": "test.json"}
        for sp, fn in files.items():
            p = Path(splits_dir) / fn
            splits[sp] = {it["img_name"] for it in json.loads(p.read_text())}
        return splits, {sp: hashlib.sha256((Path(splits_dir) / fn).read_bytes()).hexdigest()
                        for sp, fn in files.items()}
    from datasets import load_dataset
    for sp in ("train", "validation", "test"):
        # no silent empty split: an unavailable split would undercount membership
        splits[sp] = {it["img_name"] for it in load_dataset("BoKelvin/Slake", split=sp)}
    return splits, {}


def _git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent).stdout.strip() or None
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs", default=IMGS_DEFAULT)
    ap.add_argument("--receipt",
                    default="results/cipe_scale/bbox_fg_medgemma4b_n400.json",
                    help="the run whose RETAINED items define the cohort")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-source", choices=["hf", "official"], default="official")
    ap.add_argument("--splits-dir", default=None)
    ap.add_argument("--mapping", default=None,
                    help="image_mapping.json; if given, image_ref must match per item")
    ap.add_argument("--legacy-filter", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--out", required=True,
                    help="never the published outputs/region_exposure_audit.json in place")
    a = ap.parse_args(argv)

    imgs = Path(a.imgs)
    cand = build_pool(imgs, a.seed, legacy_filter=a.legacy_filter)
    rows = json.loads(Path(a.receipt).read_text())["rows"]
    mapping = json.loads(Path(a.mapping).read_text()) if a.mapping else None
    try:
        sel = identity_check(imgs, cand, rows, mapping)
    except IdentityError as e:
        print(f"[FATAL] {e.msg}")
        raise
    splits, split_sha = load_splits(a.split_source, a.splits_dir)

    all_imgs = set().union(*splits.values())
    memberships = Counter()
    per_item = []
    in_train = 0
    for sid, r in sel:
        hit = tuple(k for k in ("test", "train", "validation") if r["img_name"] in splits[k])
        m = ",".join(hit) if hit else "none"
        memberships[m] += 1
        per_item.append([sid, r["img_name"], m])
        if "train" in hit:
            in_train += 1

    out = {
        "n_items": len(sel), "seed": a.seed, "cohort_source": a.receipt,
        "pool_filter": "legacy (DEFECTIVE)" if a.legacy_filter else "fg-loader-identical",
        "identity_check": "per item: decoded size + detection.json boxes == receipt row"
                          + (" ; img_name == image_mapping.json image_ref" if mapping else ""),
        "split_source": a.split_source, "split_file_sha256": split_sha,
        "split_sizes_by_imgname": {k: len(v) for k, v in splits.items()},
        "n_distinct_images_across_splits": len(all_imgs),
        "splits_are_image_disjoint": sum(len(v) for v in splits.values()) == len(all_imgs),
        "membership_counts": dict(sorted(memberships.items())),
        "n_on_train_images": in_train,
        "frac_on_train_images": in_train / len(sel) if sel else None,
        "n_test_and_train": memberships.get("test,train", 0),
        "n_test_only": memberships.get("test", 0),
        "n_distinct_images_in_cohort": len({r["img_name"] for _, r in sel}),
        "generator": "code/scripts/region_exposure_audit.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_head_at_run": _git_head(),
        "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": "MedGemma tech report arXiv:2507.05201: Table 1 lists SLAKE (450) in training; "
                  "'For SLAKE, we used the default train/test split' refers to evaluation. "
                  "Train-split provenance of the 450 is an inference.",
        "per_item": per_item,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("source", "per_item")}, indent=2))
    print(f"\n[SAVED] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
