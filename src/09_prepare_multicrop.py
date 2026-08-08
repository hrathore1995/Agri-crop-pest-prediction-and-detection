"""
src/09_prepare_multicrop.py
Image track, step 2: turn the Multi-Crop YOLO dataset into a classification
manifest, one label per image, and check for train/test leakage from Roboflow
augmentation. Extracts the zip once to data/processed/multicrop/.
Run:  python src/09_prepare_multicrop.py
"""
from pathlib import Path
import zipfile, csv, re, collections

ROOT = Path(__file__).resolve().parent.parent
RAW  = ROOT / "data" / "raw"
OUT  = ROOT / "data" / "processed" / "multicrop"
IMG_EXT = (".jpg",".jpeg",".png")

def find_zip():
    for z in RAW.glob("*.zip"):
        if "multi" in z.name.lower(): return z
    raise FileNotFoundError("Multi-Crop zip not found in data/raw/")

def parse_names(yaml_text):
    m = re.search(r"names:\s*\[([^\]]*)\]", yaml_text)
    items = [x.strip().strip("'\"") for x in m.group(1).split(",")]
    return {i: v for i, v in enumerate(items)}

def source_key(fname):
    # Roboflow: "<original>_jpg.rf.<hash>.jpg" -> source id is the part before .rf.
    return re.split(r"\.rf\.", fname)[0]

def label_to_class(label_bytes):
    """One class per image: the most frequent class id among its boxes."""
    ids = []
    for line in label_bytes.decode("utf-8","ignore").splitlines():
        toks = line.split()
        if toks:
            try: ids.append(int(toks[0]))
            except ValueError: pass
    if not ids: return None, False
    counts = collections.Counter(ids)
    top = counts.most_common(1)[0][0]
    mixed = len(counts) > 1
    return top, mixed

def main():
    zip_path = find_zip()
    # 1. extract once
    if not OUT.exists():
        print(f"Extracting {zip_path.name} -> {OUT} (one-time)...")
        OUT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(OUT)
        print("  done.")
    else:
        print(f"Using already-extracted data at {OUT}")

    # 2. locate the folder containing data.yaml
    yaml_path = next(OUT.rglob("data.yaml"))
    base = yaml_path.parent
    names = parse_names(yaml_path.read_text(encoding="utf-8", errors="ignore"))
    print(f"{len(names)} classes; base = {base.relative_to(OUT)}")

    # 3. build manifest across splits
    rows = []
    mixed_count = 0
    for split in ("train","valid","test"):
        img_dir = base / split / "images"
        lbl_dir = base / split / "labels"
        if not img_dir.exists(): continue
        for img in img_dir.iterdir():
            if img.suffix.lower() not in IMG_EXT: continue
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists(): continue
            cid, mixed = label_to_class(lbl.read_bytes())
            if cid is None: continue
            mixed_count += mixed
            rows.append({"split": split, "path": str(img),
                         "class_id": cid, "class_name": names.get(cid, str(cid)),
                         "crop": names.get(cid, "").split("_")[0],
                         "source": source_key(img.name)})

    manifest = OUT / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split","path","class_id","class_name","crop","source"])
        w.writeheader(); w.writerows(rows)
    print(f"\nManifest: {len(rows)} images -> {manifest}")
    print(f"Images with mixed-class boxes (used majority class): {mixed_count}")

    # 4. LEAKAGE CHECK — does any source photo appear in more than one split?
    split_of = collections.defaultdict(set)
    for r in rows:
        split_of[r["source"]].add(r["split"])
    leaked = {s: v for s, v in split_of.items() if len(v) > 1}
    print(f"\n=== LEAKAGE CHECK ===")
    print(f"Source photos appearing in >1 split: {len(leaked)}")
    if leaked:
        for s, v in list(leaked.items())[:10]:
            print(f"   {s}  ->  {sorted(v)}")
        print("   ⚠  These must be resolved before trusting the split.")
    else:
        print("   ✓ clean — no source photo crosses train/valid/test.")

    # 5. per-split and per-class distribution
    print("\nImages per split:")
    for sp, c in collections.Counter(r["split"] for r in rows).items():
        print(f"   {sp}: {c}")
    print("\nSmallest 8 classes (train only) — imbalance watch:")
    tr = collections.Counter(r["class_name"] for r in rows if r["split"]=="train")
    for name, c in sorted(tr.items(), key=lambda x: x[1])[:8]:
        print(f"   {c:5d}  {name}")

if __name__ == "__main__":
    main()