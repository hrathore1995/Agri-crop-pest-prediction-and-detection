"""
src/11_train_multicrop.py
Image track, step 3: fine-tune a pretrained MobileNetV3-small on the clean
multi-crop split (30 disease classes, 5 crops). Runs on the M4 GPU (MPS).

Handles the heavy class imbalance with inverse-frequency class weights, and
reports MACRO-F1 + per-class metrics (not just accuracy) so the tiny chilli/
cauliflower classes are held honestly accountable.

Set QUICK = True for a fast end-to-end smoke test before the full run.
Run:  python src/11_train_multicrop.py
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # unsupported MPS ops -> CPU
from pathlib import Path
import csv, json, collections, random
import numpy as np

ROOT   = Path(__file__).resolve().parent.parent
MAN    = ROOT / "data" / "processed" / "multicrop" / "manifest_clean.csv"
MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH    = 32
EPOCHS   = 6
LR       = 5e-4
QUICK    = False        # True -> subsample + 1 epoch, just to verify it runs
NUM_WORKERS = 0         # 0 avoids macOS multiprocessing pickling errors
SEED     = 42
random.seed(SEED); np.random.seed(SEED)


# ---------- torch-free data prep (unit-testable) ----------
def load_manifest():
    rows = list(csv.DictReader(open(MAN)))
    classes = sorted({r["class_name"] for r in rows})
    cls2idx = {c: i for i, c in enumerate(classes)}
    return rows, classes, cls2idx

def compute_class_weights(rows, classes):
    counts = collections.Counter(r["class_name"] for r in rows if r["split"] == "train")
    freq = np.array([counts.get(c, 0) for c in classes], dtype=float)
    w = freq.sum() / (len(classes) * np.maximum(freq, 1))    # inverse frequency
    return w.astype(np.float32), freq


def main():
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models, transforms
    from PIL import Image
    from sklearn.metrics import classification_report, f1_score, accuracy_score

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    rows, classes, cls2idx = load_manifest()
    weights, freq = compute_class_weights(rows, classes)
    print(f"{len(classes)} classes | train imbalance: "
          f"min={int(freq.min())} max={int(freq.max())} images/class")

    norm = transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2,0.2,0.2),
        transforms.ToTensor(), norm])
    eval_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(), norm])

    class LeafDS(Dataset):
        def __init__(self, split, tf):
            self.items = [(r["path"], cls2idx[r["class_name"]])
                          for r in rows if r["split"] == split]
            if QUICK and split == "train":
                random.shuffle(self.items); self.items = self.items[:1500]
            self.tf = tf
        def __len__(self): return len(self.items)
        def __getitem__(self, i):
            path, y = self.items[i]
            img = Image.open(path).convert("RGB")
            return self.tf(img), y

    dl = lambda ds, sh: DataLoader(ds, batch_size=BATCH, shuffle=sh,
                                   num_workers=NUM_WORKERS, pin_memory=False)
    train_dl = dl(LeafDS("train", train_tf), True)
    valid_dl = dl(LeafDS("valid", eval_tf), False)
    test_dl  = dl(LeafDS("test",  eval_tf), False)
    print(f"train={len(train_dl.dataset)} valid={len(valid_dl.dataset)} "
          f"test={len(test_dl.dataset)}")

    # pretrained backbone, new 30-way head
    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    model = model.to(device)

    crit = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    epochs = 1 if QUICK else EPOCHS

    def run(dloader, train):
        model.train() if train else model.eval()
        ys, ps, tot = [], [], 0.0
        for xb, yb in dloader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.set_grad_enabled(train):
                out = model(xb); loss = crit(out, yb)
                if train:
                    opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * xb.size(0)
            ys += yb.cpu().tolist(); ps += out.argmax(1).cpu().tolist()
        return tot/len(dloader.dataset), accuracy_score(ys,ps), f1_score(ys,ps,average="macro"), ys, ps

    best = -1
    for ep in range(1, epochs+1):
        tl, ta, tf1, *_ = run(train_dl, True)
        vl, va, vf1, *_ = run(valid_dl, False)
        print(f"epoch {ep}/{epochs}  train loss {tl:.3f} f1 {tf1:.3f} | "
              f"valid loss {vl:.3f} acc {va:.3f} macro-F1 {vf1:.3f}")
        if vf1 > best:
            best = vf1
            torch.save(model.state_dict(), MODELS/"multicrop_mobilenetv3.pt")
            json.dump(classes, open(MODELS/"multicrop_classes.json","w"))

    # ---- honest test evaluation ----
    model.load_state_dict(torch.load(MODELS/"multicrop_mobilenetv3.pt"))
    _, acc, macro, ys, ps = run(test_dl, False)
    print(f"\n=== TEST ===  accuracy {acc:.3f} | macro-F1 {macro:.3f}")
    print("(macro-F1 weights every class equally, so weak rare classes can't hide)\n")
    print(classification_report(ys, ps, target_names=classes, zero_division=0, digits=3))
    print(f"Saved -> {MODELS/'multicrop_mobilenetv3.pt'}")


if __name__ == "__main__":
    main()