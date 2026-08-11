"""
src/12_build_ood.py
Image track, step 4: build an out-of-distribution (OOD) detector for the leaf
model so it can say "this isn't a supported leaf" instead of confidently
misclassifying an elephant. Reuses the trained model — no retraining.

Method: extract each image's feature vector (the network's internal fingerprint),
fit per-class means + a shared covariance, and flag any input whose Mahalanobis
distance to every leaf class is larger than a threshold set on real leaves.
Run:  python src/12_build_ood.py   (after training the leaf model)
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from pathlib import Path
import csv, json
import numpy as np

ROOT   = Path(__file__).resolve().parent.parent
MAN    = ROOT / "data" / "processed" / "multicrop" / "manifest_clean.csv"
MODELS = ROOT / "models"
LEAF   = MODELS / "multicrop_mobilenetv3.pt"
CLS    = MODELS / "multicrop_classes.json"
OOD    = MODELS / "multicrop_ood.npz"
IMG_SIZE, BATCH = 224, 64


def main():
    import torch, torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models, transforms
    from PIL import Image

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)
    classes = json.load(open(CLS)); c2i = {c: i for i, c in enumerate(classes)}

    m = models.mobilenet_v3_small()
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(classes))
    m.load_state_dict(torch.load(LEAF, map_location=device)); m.eval().to(device)
    extractor = nn.Sequential(m.features, m.avgpool).to(device).eval()  # -> feature map

    norm = transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(IMG_SIZE),
                             transforms.ToTensor(), norm])
    rows = list(csv.DictReader(open(MAN)))

    class DS(Dataset):
        def __init__(self, split):
            self.items = [(r["path"], c2i[r["class_name"]]) for r in rows if r["split"]==split]
        def __len__(self): return len(self.items)
        def __getitem__(self, i):
            p, y = self.items[i]; return tf(Image.open(p).convert("RGB")), y

    def embed(split):
        dl = DataLoader(DS(split), batch_size=BATCH, shuffle=False, num_workers=0)
        X, Y = [], []
        with torch.no_grad():
            for xb, yb in dl:
                e = extractor(xb.to(device)).flatten(1).cpu().numpy()
                X.append(e); Y.append(yb.numpy())
        return np.concatenate(X), np.concatenate(Y)

    print("Extracting train embeddings (this pass takes a few minutes)...")
    Xtr, ytr = embed("train")
    print("Extracting valid embeddings...")
    Xva, _ = embed("valid")
    d = Xtr.shape[1]
    print(f"embedding dim = {d}, train N = {len(Xtr)}")

    means = np.stack([Xtr[ytr==i].mean(0) for i in range(len(classes))])
    centered = Xtr - means[ytr]
    cov = centered.T @ centered / len(Xtr) + 1e-3*np.eye(d)
    cov_inv = np.linalg.inv(cov).astype(np.float32)

    def maha(X):
        diffs = X[:, None, :] - means[None, :, :]
        return np.einsum("ncd,de,nce->nc", diffs, cov_inv, diffs).min(1)

    id_scores = maha(Xva)
    thr = float(np.percentile(id_scores, 97.5))   # ~2.5% real leaves flagged
    print(f"In-distribution Mahalanobis (valid): median={np.median(id_scores):.1f}, "
          f"97.5th pct (threshold)={thr:.1f}")

    np.savez(OOD, means=means.astype(np.float32), cov_inv=cov_inv,
             threshold=np.float32(thr), classes=np.array(classes))
    print(f"Saved OOD detector -> {OOD}")


if __name__ == "__main__":
    main()