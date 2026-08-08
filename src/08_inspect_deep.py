"""
src/08_inspect_deep.py
Image track, step 1b: deeper look — real class folders for the soybean set,
and the data.yaml class list + class balance for the multi-crop YOLO set.
Run:  python src/08_inspect_deep.py
"""
from pathlib import Path
import zipfile, collections, io, random

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
IMG_EXT = (".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp")

def folder_counts(names, depth):
    c = collections.Counter()
    for n in names:
        parts = n.split("/")
        if len(parts) > depth:
            c["/".join(parts[:depth])] += 1
    return c

def inspect_classification(zf, imgs, name):
    print("\n-- folder structure (image counts by depth) --")
    for depth in (3, 4):
        cc = folder_counts(imgs, depth)
        if not cc: continue
        print(f"\n  depth {depth}:")
        for folder, cnt in cc.most_common(30):
            print(f"    {cnt:6d}  {folder}")

def inspect_yolo(zf, names, imgs, name):
    # 1. class names from data.yaml
    yml = [n for n in names if n.lower().endswith((".yaml",".yml"))]
    class_names = {}
    if yml:
        raw = zf.read(yml[0]).decode("utf-8", "ignore")
        print("\n-- data.yaml (verbatim) --")
        print("   " + raw.replace("\n","\n   ")[:800])
        # naive parse of names: [...] or names:\n - a
        import re
        m = re.search(r"names:\s*\[([^\]]*)\]", raw)
        if m:
            items = [x.strip().strip("'\"") for x in m.group(1).split(",")]
            class_names = {i: v for i, v in enumerate(items)}
        else:
            items = re.findall(r"^\s*-\s*(.+)$", raw, re.M)
            class_names = {i: v.strip().strip("'\"") for i, v in enumerate(items)}
    print(f"\n-- {len(class_names)} classes from yaml --")
    for i, v in class_names.items(): print(f"    {i}: {v}")

    # 2. images per split
    print("\n-- images per split --")
    for split in ("train","valid","test"):
        n = sum(1 for p in imgs if f"/{split}/" in p)
        if n: print(f"    {split}: {n}")

    # 3. class balance from a sample of label files (first token per line = class id)
    labels = [n for n in names if n.lower().endswith(".txt") and "label" in n.lower()]
    sample = labels if len(labels) <= 6000 else random.sample(labels, 6000)
    per_class_imgs = collections.Counter()
    for lp in sample:
        try:
            txt = zf.read(lp).decode("utf-8","ignore").strip()
            ids = {int(line.split()[0]) for line in txt.splitlines() if line.split()}
            for cid in ids: per_class_imgs[cid] += 1
        except Exception:
            pass
    print(f"\n-- class balance (images containing each class; "
          f"sample of {len(sample)} label files) --")
    for cid, cnt in per_class_imgs.most_common():
        print(f"    {cnt:6d}  [{cid}] {class_names.get(cid, '??')}")

def main():
    for z in sorted(RAW.glob("*.zip")):
        print("\n" + "="*72 + f"\nARCHIVE: {z.name}\n" + "="*72)
        with zipfile.ZipFile(z) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            imgs  = [n for n in names if n.lower().endswith(IMG_EXT)]
            is_yolo = any(n.lower().endswith((".yaml",".yml")) for n in names)
            if is_yolo: inspect_yolo(zf, names, imgs, z.name)
            else:       inspect_classification(zf, imgs, z.name)

if __name__ == "__main__":
    main()