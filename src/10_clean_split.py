"""
src/10_clean_split.py
Image track, step 2b: re-diagnose leakage with a CLASS-AWARE source key
(so generic filenames shared across crops aren't mistaken for the same photo),
then build a guaranteed-clean, stratified, group-aware split:
  - all augmented variants of one source photo stay in ONE split (no leak)
  - train keeps all variants; valid/test keep one image per source (honest eval)
Writes manifest_clean.csv. Run:  python src/10_clean_split.py
"""
from pathlib import Path
import csv, collections, random

ROOT = Path(__file__).resolve().parent.parent
MAN  = ROOT / "data" / "processed" / "multicrop" / "manifest.csv"
OUT  = ROOT / "data" / "processed" / "multicrop" / "manifest_clean.csv"
random.seed(42)

rows = list(csv.DictReader(open(MAN)))
print(f"Loaded {len(rows)} images from manifest.csv")

# ---- 1. re-diagnose leakage two ways ----
def cross_split(keyfn):
    d = collections.defaultdict(set)
    for r in rows: d[keyfn(r)].add(r["split"])
    return {k: v for k, v in d.items() if len(v) > 1}

naive = cross_split(lambda r: r["source"])                       # filename only
aware = cross_split(lambda r: (r["class_name"], r["source"]))    # class + filename
print(f"\nLeakage diagnosis:")
print(f"  filename-only key : {len(naive)} sources cross splits (includes false alarms)")
print(f"  class-aware key   : {len(aware)} sources cross splits  <-- the real suspects")
if aware:
    for (cls, src), sp in list(aware.items())[:8]:
        print(f"      REAL: {src} [{cls}] -> {sorted(sp)}")

# ---- 2. build a clean group split keyed by (class_name, source) ----
groups = collections.defaultdict(list)          # (class, source) -> [rows]
for r in rows:
    groups[(r["class_name"], r["source"])].append(r)

by_class = collections.defaultdict(list)         # class -> [source-groups]
for key, grp in groups.items():
    by_class[key[0]].append(grp)

new_rows = []
for cls, srcgroups in by_class.items():
    random.shuffle(srcgroups)
    n = len(srcgroups)
    n_val = max(1, round(0.15 * n)) if n >= 3 else 0
    n_test = max(1, round(0.15 * n)) if n >= 3 else 0
    n_val = min(n_val, n - 1); n_test = min(n_test, n - 1 - n_val)
    for i, grp in enumerate(srcgroups):
        if i < n_test:                     split = "test"
        elif i < n_test + n_val:           split = "valid"
        else:                              split = "train"
        if split == "train":
            keep = grp                      # keep all augmented variants
        else:
            keep = [grp[0]]                 # one representative per source
        for r in keep:
            r2 = dict(r); r2["split"] = split
            new_rows.append(r2)

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(new_rows)

# ---- 3. verify the clean split & report ----
ver = cross_split_new = collections.defaultdict(set)
for r in new_rows: cross_split_new[(r["class_name"], r["source"])].add(r["split"])
leaks_after = sum(1 for v in cross_split_new.values() if len(v) > 1)

print(f"\nClean split written -> {OUT}  ({len(new_rows)} images)")
print(f"Leaks after rebuild (class-aware): {leaks_after}  (must be 0)")
counts = collections.Counter(r["split"] for r in new_rows)
print(f"Split sizes: " + ", ".join(f"{k}={counts[k]}" for k in ("train","valid","test")))
print("\nPer-crop image counts (clean):")
for crop, c in collections.Counter(r["crop"] for r in new_rows).most_common():
    print(f"   {crop:14s} {c}")