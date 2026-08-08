"""
src/01_load_data.py
Track A, step 1: load the Assam tomato sensor CSV and inspect it.
Loading and looking only — no labeling or modeling yet.
"""
from pathlib import Path
import pandas as pd

# Project root is one level up from this src/ file
ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "raw" / "tomato_irrigation_dataset.csv"

# Load
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]  # trim stray spaces in headers

# Inspect
print("=" * 60)
print(f"Loaded: {CSV_PATH.name}")
print("=" * 60)
print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns\n")
print("Columns (raw names):")
for c in df.columns:
    print(f"  - {c!r}")
print("\nFirst 5 rows:")
print(df.head().to_string())
print("\nMissing values per column:")
print(df.isna().sum().to_string())
print("\nSummary statistics:")
print(df.describe().round(2).to_string())