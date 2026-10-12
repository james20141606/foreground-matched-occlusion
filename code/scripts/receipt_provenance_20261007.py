#!/usr/bin/env python3
"""Receipt provenance table for the ICLR revision R1 (2026-10-07; BootLoops audit F7, kit item 28).

For each receipt family the paper cites, record from git and from the receipt itself:
  - first commit that added the file (hash, date),
  - whether the receipt records the hash of the script that generated it,
  - whether that hash equals the sha256 of ANY committed version of ANY .py in the repo,
  - whether the receipt records a git commit and a timestamp.
Nothing is typed by hand except the list of receipt paths. Writes
results/receipt_provenance_20261007/provenance.json and
results/receipt_provenance_20261007/table_receipt_provenance.tex.

  /usr/bin/python3 code/scripts/receipt_provenance_20261007.py
"""
import glob, hashlib, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "receipt_provenance_20261007")
# (label in the table, receipt glob(s); the first match is inspected for fields)
FAMILIES = [
    ("Matched-control runs, six models", ["results/cipe_scale/bbox_fg_*_n400.json"]),
    ("Corrected-TV readout", ["outputs/region_tv_corrected.json"]),
    ("ChestX-Det follow-up", ["results/exposure_clean/expclean_*.json"]),
    ("MedTrust-IN v0.3 statistics", ["data/medtrust_in_test/medtrust_in_stats.json"]),
    ("Image-cluster sensitivity", ["outputs/image_cluster_20260922/image_cluster_sensitivity.json"]),
    ("Legacy identity mapping", ["outputs/legacy_mapping_20260923/general_identity_mapping.json"]),
    ("MedGemma-4B paired contrast", ["outputs/medgemma4b_paired_20260924/paired_contrast.json"]),
    ("Legacy-coverage replay", ["outputs/legacy_coverage_replay_20260925/identity.json"]),
    ("Receipt-only instrument checks", ["results/region_instrument_20260926/receipt_only.json"]),
    ("Corrected exposure audit", ["results/exposure_audit_corrected_20261004/region_exposure_audit_corrected.json"]),
    ("Monte Carlo seed sensitivity", ["results/seed_sensitivity_20261004/summary.json"]),
]
SCRIPT_KEYS = ("generator_sha256", "script_sha256")
COMMIT_KEYS = ("git_head_at_run", "registration_commit", "loader_code_commit")
TIME_KEYS = ("generated_at",)


def git(*a):
    r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"git {' '.join(a)} failed: {r.stderr.strip()}")
    return r.stdout


def committed_py_hashes():
    """sha256 of every committed version of every .py file, over all refs."""
    h = {}
    for line in git("rev-list", "--all", "--objects").splitlines():
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[1].endswith(".py"):
            data = subprocess.run(["git", "cat-file", "-p", parts[0]], cwd=ROOT,
                                  capture_output=True).stdout
            h.setdefault(hashlib.sha256(data).hexdigest(), parts[1])
    return h


def main():
    known = committed_py_hashes()
    if not known:
        sys.exit("no committed .py blobs found -- refusing to report 'no match' for everything")
    rows = []
    for label, globs in FAMILIES:
        files = sorted(f for g in globs for f in glob.glob(os.path.join(ROOT, g)))
        if not files:
            sys.exit(f"{label}: no receipt matches {globs}")
        rel = [os.path.relpath(f, ROOT) for f in files]
        firsts = []
        for r in rel:
            out = git("log", "--diff-filter=A", "--format=%h %ad", "--date=short", "--", r).split("\n")
            out = [x for x in out if x.strip()]
            if not out:
                sys.exit(f"{r}: never committed")
            firsts.append(out[-1])
        d = json.load(open(files[0]))
        sh = next((d[k] for k in SCRIPT_KEYS if isinstance(d, dict) and k in d), None)
        cm = next((d[k] for k in COMMIT_KEYS if isinstance(d, dict) and k in d), None)
        ts = next((d[k] for k in TIME_KEYS if isinstance(d, dict) and k in d), None)
        match = None if sh is None else (known.get(sh) is not None)
        rows.append({
            "label": label, "receipts": rel, "n_files": len(rel),
            "first_commit": sorted(firsts, key=lambda x: x.split()[1])[0],
            "first_commit_latest": sorted(firsts, key=lambda x: x.split()[1])[-1],
            "script_hash_recorded": sh, "script_hash_matches_committed_version": match,
            "matched_script": known.get(sh) if sh else None,
            "commit_recorded": cm, "timestamp_recorded": ts,
        })
    os.makedirs(OUT, exist_ok=True)
    json.dump({"generator": "code/scripts/receipt_provenance_20261007.py",
               "head_at_run": git("rev-parse", "HEAD").strip(),
               "n_committed_py_versions": len(known), "rows": rows},
              open(os.path.join(OUT, "provenance.json"), "w"), indent=1)

    def yn(v):
        return "---" if v is None else ("yes" if v else "\\textbf{no}")
    lines = [r"\begin{tabular}{lcccc}\toprule",
             r"Receipt (files) & First committed & Script hash & Matches a committed script & Commit, time \\\midrule"]
    for r in rows:
        dates = r["first_commit"].split()[1]
        if r["first_commit_latest"].split()[1] != dates:
            dates += "--" + r["first_commit_latest"].split()[1][5:]
        ct = ("both" if r["commit_recorded"] and r["timestamp_recorded"] else
              "commit" if r["commit_recorded"] else "time" if r["timestamp_recorded"] else "none")
        lines.append(f"{r['label']} ({r['n_files']}) & {dates} & "
                     f"{'yes' if r['script_hash_recorded'] else 'no'} & "
                     f"{yn(r['script_hash_matches_committed_version'])} & {ct} \\\\")
    lines.append(r"\bottomrule\end{tabular}")
    open(os.path.join(OUT, "table_receipt_provenance.tex"), "w").write("\n".join(lines) + "\n")
    for r in rows:
        print(f"{r['label']:34s} {r['first_commit']:20s} hash={'Y' if r['script_hash_recorded'] else 'N'} "
              f"match={r['script_hash_matches_committed_version']} commit={r['commit_recorded']} time={r['timestamp_recorded']}")


if __name__ == "__main__":
    main()
