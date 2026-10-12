"""POST HOC (not in the registration): power of the I3 fingerprint and seed-robustness of phi_L.

Written after the registered replay passed, because mutation-testing I3 showed it catches a
wrong seed but not a one-item offset in the rng stream. Two questions:
  (1) over 200 wrong seeds, and over one-item stream offsets, how often does I3 still pass?
  (2) how much does the 229-item mean legacy coverage phi_L move across seeds? If every seed
      gives a similar phi_L, the coverage conclusion does not hinge on the exact replay.
CPU only; reuses the registered script's extraction and inputs.

Usage: /usr/bin/python3 code/scripts/legacy_replay_gate_power_20260925.py --imgs <SLAKE imgs>
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import legacy_coverage_replay_20260925 as R


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--imgs", required=True)
    args = ap.parse_args(); imgs = Path(args.imgs)
    ns = {"np": np, "Image": Image, "ImageDraw": ImageDraw, "random": random}
    fg_src = (R.SCRIPTS / "cipe_bbox_foreground_control.py").read_text()
    import ast
    for node in ast.parse(fg_src).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and \
                node.targets[0].id == "FG_THRESH_FRAC":
            exec(compile(ast.get_source_segment(fg_src, node), "fg", "exec"), ns)
    R.extract(R.SCRIPTS / "cipe_bbox_external_validation.py", ["random_boxes", "mask_boxes"], ns)
    R.extract(R.SCRIPTS / "cipe_bbox_foreground_control.py", ["foreground_mask", "_coverage"], ns)
    random_boxes, mask_boxes = ns["random_boxes"], ns["mask_boxes"]
    foreground_mask, _coverage = ns["foreground_mask"], ns["_coverage"]

    items = json.loads(R.MAPPING.read_text())["items"]
    leg = json.loads(R.LEGACY_RECEIPT.read_text())["rows"]
    rep = json.loads((R.OUT / "coverage.json").read_text())
    admitted = [r["admitted"] for r in rep["items"]]
    cr1 = np.array([r["cipe_random"] == 1.0 for r in leg])
    pre = []
    for it in items:
        img = Image.open(imgs / it["image_ref"]).convert("RGB")
        arr = np.asarray(img)
        pre.append((img, arr, foreground_mask(img), it["organ_boxes"]))

    def run(seed, skip_at=None):
        rng = random.Random(seed); phis, unch = [], []
        for i, (img, arr, fg, boxes) in enumerate(pre):
            if skip_at is not None and i == skip_at:
                lb = random_boxes(img, boxes, random.Random(-1))   # this item does not consume the stream
            else:
                lb = random_boxes(img, boxes, rng)
            phis.append(float(np.mean([_coverage(fg, b) for b in lb])))
            unch.append(bool(np.array_equal(np.asarray(mask_boxes(img, lb)), arr)))
        unch = np.array(unch); phis = np.array(phis)
        U = int(unch.sum()); ok = int((unch & cr1).sum())
        return {"U": U, "U_ok": ok, "I3_pass": bool(U >= 10 and ok == U),
                "phi_legacy_229": float(phis[np.array(admitted)].mean()),
                "phi_legacy_400": float(phis.mean())}

    base = run(R.SEED_RUN)
    wrong = {s: run(s) for s in range(1000, 1200)}
    offs = {k: run(R.SEED_RUN, skip_at=k) for k in range(0, 400, 20)}
    phis = np.array([v["phi_legacy_229"] for v in wrong.values()])
    out = {
        "post_hoc": True,
        "registered_seed_42": base,
        "wrong_seeds_1000_1199": {
            "n": len(wrong), "I3_pass": int(sum(v["I3_pass"] for v in wrong.values())),
            "phi_legacy_229_min": float(phis.min()), "phi_legacy_229_max": float(phis.max()),
            "phi_legacy_229_mean": float(phis.mean())},
        "one_item_offsets_every_20th": {
            "n": len(offs), "I3_pass": int(sum(v["I3_pass"] for v in offs.values())),
            "phi_legacy_229_min": float(min(v["phi_legacy_229"] for v in offs.values())),
            "phi_legacy_229_max": float(max(v["phi_legacy_229"] for v in offs.values()))},
    }
    (R.OUT / "gate_power_posthoc.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
