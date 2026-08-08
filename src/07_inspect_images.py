"""
src/07_inspect_images.py
Image track, step 1: inspect the downloaded image zips WITHOUT extracting them.
Reports structure, class folders, image counts, annotation format (folders vs
YOLO bbox), and sample image sizes — so we design the model against reality.
Run:  python src/07_inspect_images.py
"""
from pathlib import Path
import zipfile
import collections

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

def top_levels(names, depth=2):
    """Count files grouped by their first `depth` path components (folders)."""
    c = collections.Counter()
    for n in names:
        parts = n.split("/")
        key = "/".join(parts[:depth]) if len(parts) > depth else "/".join(parts[:-1])
        c[key] += 1
    return c

def inspect(zip_path):
    print("\n" + "=" * 72)
    print(f"ARCHIVE: {zip_path.name}   ({zip_path.stat().st_size/1e9:.2f} GB)")
    print("=" * 72)
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        imgs   = [n for n in names if n.lower().endswith(IMG_EXT)]
        txts   = [n for n in names if n.lower().endswith(".txt")]
        yamls  = [n for n in names if n.lower().endswith((".yaml", ".yml"))]
        csvs   = [n for n in names if n.lower().endswith(".csv")]
        print(f"Total files: {len(names)} | images: {len(imgs)} | "
              f".txt: {len(txts)} | .yaml: {len(yamls)} | .csv: {len(csvs)}")

        # annotation-format heuristics
        print("\nLikely format:")
        if yamls or (len(txts) > 0.3*len(imgs) and len(txts) > 50):
            print("  → looks like OBJECT DETECTION (YOLO): per-image .txt label files"
                  + (" + data.yaml" if yamls else ""))
        else:
            print("  → looks like CLASSIFICATION (labels via folder names)")
        for special in ("classes.txt","data.yaml","labels.csv","train","valid","test"):
            hit = [n for n in names if special.lower() in n.lower()][:1]
            if hit: print(f"    found reference: {hit[0]}")

        # folder structure = candidate classes
        print("\nImage counts by folder (top 2 levels), 25 largest:")
        for folder, cnt in top_levels(imgs, depth=2).most_common(25):
            print(f"  {cnt:6d}  {folder}")

        # sample image sizes (needs Pillow; optional)
        print("\nSample image sizes:")
        try:
            from PIL import Image
            import io
            for n in imgs[:5]:
                with zf.open(n) as f:
                    im = Image.open(io.BytesIO(f.read()))
                    print(f"  {im.size[0]}x{im.size[1]}  {im.mode}  {n}")
        except Exception as e:
            print(f"  (skipped — install Pillow to see sizes: {e})")

def main():
    zips = sorted(RAW.glob("*.zip"))
    if not zips:
        print(f"No .zip files found in {RAW}"); return
    print(f"Found {len(zips)} archive(s) in {RAW}:")
    for z in zips: print(f"  - {z.name}")
    for z in zips:
        try:
            inspect(z)
        except Exception as e:
            print(f"\n!! Could not read {z.name}: {e}")

if __name__ == "__main__":
    main()