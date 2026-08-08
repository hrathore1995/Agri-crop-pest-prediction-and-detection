"""
src/06_pooled_model.py
Track B, step 3: POOLED severity model across all viable series.
Trains, evaluates vs per-series seasonal climatology (endemic vs episodic),
and saves BOTH the model and an app bundle (feature columns, climatology,
risk-band cutoffs, endemic labels) for the Streamlit front end.
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

import pipeline as P

MODEL_OUT  = P.ROOT / "models" / "track_b_pooled_reg.json"
BUNDLE_OUT = P.ROOT / "models" / "app_bundle.pkl"

df = P.load_clean()
data, X = P.build_pooled(df)
data["y"] = np.log1p(data["pest_value"])
print(f"Pooled rows: {len(data)}  |  features: {X.shape[1]}")

tr = (data["year"] < P.SPLIT_YEAR).values
te = ~tr
print(f"Train rows: {tr.sum()}  Test rows: {te.sum()}")

# endemic/episodic per series (train present-rate)
pres = (data[tr].assign(p=(data.loc[tr,"pest_value"]>0).astype(int))
        .groupby(["location","pest"])["p"].mean())
data["series"] = list(zip(data.location, data.pest))
data["endemic"] = data["series"].map(lambda s: pres.get(s, np.nan) >= P.ENDEMIC_THR)

# per-series seasonal climatology baseline
clim = data[tr].groupby(["location","pest","standard_week"])["y"].mean()
glob = data.loc[tr,"y"].mean()
data["seas"] = data.apply(
    lambda r: clim.get((r.location, r.pest, r.standard_week), np.nan), axis=1).fillna(glob)

# model
reg = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.04,
                   subsample=0.9, colsample_bytree=0.8, random_state=42)
reg.fit(X[tr], data.loc[tr,"y"])
data["pred"] = reg.predict(X)

def block(mask, label):
    y = data.loc[mask,"y"]
    print(f"  {label:14s} n={mask.sum():5d} | "
          f"seasonal R2={r2_score(y, data.loc[mask,'seas']):+.3f} "
          f"MAE={mean_absolute_error(y, data.loc[mask,'seas']):.3f}  ||  "
          f"model R2={r2_score(y, data.loc[mask,'pred']):+.3f} "
          f"MAE={mean_absolute_error(y, data.loc[mask,'pred']):.3f}")

print("\n=== POOLED SEVERITY: seasonal vs weather model (TEST) ===")
block(te, "ALL")
block(te &  data["endemic"].fillna(False).values, "ENDEMIC")
block(te & ~data["endemic"].fillna(True).values,  "EPISODIC")

# ---- build & save the app bundle -----------------------------------------
feat_clim = {}
gb = data.groupby(["location","standard_week"])[P.BASE_FEATS].mean()
for (loc, wk), r in gb.iterrows():
    feat_clim[(loc, int(wk))] = r.values.astype(float)

band_cutoffs = {}
for (loc, pest), g in data.groupby(["location","pest"]):
    pos = g["pest_value"][g["pest_value"] > 0]              # actual outbreak levels
    if len(pos) >= 8:
        q1, q2 = pos.quantile([1/3, 2/3])
    elif len(pos) > 0:                                       # too few to tier
        q1 = q2 = float(pos.median())
    else:
        q1 = q2 = 1.0
    band_cutoffs[(loc, pest)] = (float(q1), float(q2))

bundle = {
    "feature_cols": list(X.columns),
    "base_feats": P.BASE_FEATS,
    "feat_clim": feat_clim,
    "feat_clim_global": data[P.BASE_FEATS].mean().values.astype(float),
    "band_cutoffs": band_cutoffs,
    "endemic": {k: bool(v) for k, v in
                data.dropna(subset=["endemic"]).groupby("series")["endemic"].first().items()},
    "series_list": sorted(set(zip(data.crop, data.location, data.pest))),
    "active_weeks": {k: sorted(int(w) for w in g["standard_week"].unique())
                     for k, g in data.groupby(["location","pest"])},
    "units": {(loc, pest): " ".join(str(u.iloc[0]).split())
              for (loc, pest), u in
              df.dropna(subset=["pest_value"]).groupby(["location","pest"])["unit"]},
    "obs_series": {(loc, pest): (
                       g.groupby("week_id")["pest_value"].mean().sort_index().index.astype(int).tolist(),
                       g.groupby("week_id")["pest_value"].mean().sort_index().values.astype(float).tolist())
                   for (loc, pest), g in
                   df.assign(week_id=df["year"]*52+(df["standard_week"]-1))
                     .dropna(subset=["pest_value"]).groupby(["location","pest"])},
}
reg.save_model(MODEL_OUT)
with open(BUNDLE_OUT, "wb") as f:
    pickle.dump(bundle, f)
print(f"\nSaved model  -> {MODEL_OUT}")
print(f"Saved bundle -> {BUNDLE_OUT}  ({len(bundle['series_list'])} series)")