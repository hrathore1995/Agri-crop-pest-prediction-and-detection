"""
src/04_load_crida.py
Track B, step 1: load the CRIDA pest-weather archive and inspect it.
Loading, cleaning, and looking only — no modeling yet.
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "crida_pest_weather_data.csv"
OUT = ROOT / "data" / "processed" / "crida_clean.csv"

df = pd.read_csv(RAW)
print("Raw shape:", df.shape)

# --- 1. Clean pest_value: fix known typos, coerce to numeric --------------
# Two bad entries were found earlier: "9..9" and "56.o"
typo_fix = {"9..9": np.nan, "56.o": np.nan}   # unrecoverable -> treat as missing
df["pest_value"] = df["pest_value"].replace(typo_fix)
df["pest_value"] = pd.to_numeric(df["pest_value"], errors="coerce")

# --- 2. Parse year + standard_week into an approximate weekly date --------
df["date"] = (pd.to_datetime(df["year"].astype(str), format="%Y")
              + pd.to_timedelta((df["standard_week"] - 1) * 7, unit="D"))

# --- 3. Split monitored (has a reading) vs unmonitored (blank) rows -------
monitored = df[df["pest_value"].notna()].copy()
unmonitored = df[df["pest_value"].isna()]

# presence/absence label on the monitored rows
monitored["present"] = (monitored["pest_value"] > 0).astype(int)

# --- 4. Inspection --------------------------------------------------------
print("\n" + "=" * 60)
print("CRIDA INSPECTION")
print("=" * 60)
print(f"\nTotal rows: {len(df)}")
print(f"Monitored (usable) rows: {len(monitored)}")
print(f"Unmonitored (blank pest_value) rows: {len(unmonitored)}")
print(f"\nCrops:\n{df['crop'].value_counts().to_string()}")
print(f"\nYear range: {df['year'].min()}–{df['year'].max()}")
print(f"Distinct pests: {df['pest'].nunique()}")

print("\nPresence/absence on monitored rows (1=pest present, 0=none):")
print(monitored["present"].value_counts().to_string())

print("\nWeather columns — missing values (should be 0):")
wcols = ["maxt_c","mint_c","rh1_pct","rh2_pct","rf_mm","ws_kmph","ssh_hrs","evp_mm"]
print(df[wcols].isna().sum().to_string())

# --- 5. Which crop/location/pest series have enough data to model? --------
grp = (monitored.groupby(["crop","location","pest"])
       .agg(n_obs=("present","size"),
            n_present=("present","sum"))
       .reset_index())
grp["n_absent"] = grp["n_obs"] - grp["n_present"]
grp["pct_present"] = (grp["n_present"] / grp["n_obs"] * 100).round(1)

# "modelable" = decent sample AND both classes represented
modelable = grp[(grp["n_obs"] >= 100) &
                (grp["n_present"] >= 20) &
                (grp["n_absent"] >= 20)].sort_values("n_obs", ascending=False)

print(f"\nCrop/location/pest series total: {len(grp)}")
print(f"Series with >=100 obs and both classes >=20: {len(modelable)}")
print("\nTop 15 modelable series (most data):")
print(modelable.head(15).to_string(index=False))

monitored.to_csv(OUT, index=False)
print(f"\nSaved cleaned monitored rows -> {OUT}")