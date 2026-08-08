"""
src/05_flagship_model.py  (Option A: severity, done right)
Track B flagship: forecast Yellow stem borer PRESSURE (how much), one week
ahead, from prior weeks' weather. Rice / Rajendranagar.

Why severity, not presence: this pest is present ~77-87% of weeks, so binary
"will it appear?" is near-degenerate. "How heavy?" is the well-posed question.

Honest evaluation:
  - TIME split (train early years, test later years).
  - Two baselines: (1) flat train mean, (2) SEASONAL CLIMATOLOGY = predict the
    historical average for that week-of-year. Beating (2) is the real bar:
    it proves weather adds signal BEYOND knowing the calendar.
  - Metrics: MAE & R^2 on log1p scale; median abs error in real counts;
    and a low/med/high risk-band accuracy for farmer-facing interpretability.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "crida_pest_weather_data.csv"
MODEL_OUT = ROOT / "models" / "track_b_flagship_reg.json"

LOCATION, PEST, CROP = "Rajendranagar", "Yellowstemborer", "Rice"
WEATHER = ["maxt_c","mint_c","rh1_pct","rh2_pct","rf_mm","ws_kmph","ssh_hrs","evp_mm"]

df = pd.read_csv(RAW)
df["pest_value"] = pd.to_numeric(
    df["pest_value"].replace({"9..9": np.nan, "56.o": np.nan}), errors="coerce")
for c in WEATHER:
    df[c] = pd.to_numeric(df[c].replace("-", np.nan), errors="coerce")
df["week_id"] = df["year"] * 52 + (df["standard_week"] - 1)

wx = df[df["location"] == LOCATION].groupby("week_id")[WEATHER].mean()
wx = wx.reindex(range(int(wx.index.min()), int(wx.index.max()) + 1))
wx["standard_week"] = (wx.index % 52) + 1

feat = pd.DataFrame(index=wx.index)
feat["standard_week"] = wx["standard_week"]
for c in WEATHER:
    feat[f"{c}_lag1"]  = wx[c].shift(1)
    feat[f"{c}_roll4"] = wx[c].rolling(4).mean().shift(1)
    feat[f"{c}_trend"] = wx[c].shift(1) - wx[c].shift(4)

tgt = (df[(df["location"] == LOCATION) & (df["pest"] == PEST)]
       .dropna(subset=["pest_value"]).groupby("week_id")["pest_value"].mean())
data = feat.join(tgt.rename("pest_value"), how="inner")
data = data[data["pest_value"].notna() & data["maxt_c_roll4"].notna()].copy()
data = data.sort_index()

feat_cols = list(feat.columns)
data["y"] = np.log1p(data["pest_value"])           # model on log counts

cut = int(len(data) * 0.75)
train, test = data.iloc[:cut], data.iloc[cut:]
Xtr, Xte = train[feat_cols], test[feat_cols]
ytr, yte = train["y"], test["y"]
print(f"Flagship severity: {CROP} / {LOCATION} / {PEST}")
print(f"Rows: {len(data)} (train {len(train)}, test {len(test)})")

# --- baselines -------------------------------------------------------------
b_flat = np.full_like(yte, ytr.mean(), dtype=float)

season = train.groupby("standard_week")["y"].mean()          # week-of-year avg
b_seas = test["standard_week"].map(season).fillna(ytr.mean()).to_numpy()

# --- model -----------------------------------------------------------------
reg = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.04,
                   subsample=0.9, colsample_bytree=0.9, random_state=42)
reg.fit(Xtr, ytr)
pred = reg.predict(Xte)

def report(name, yhat):
    print(f"  {name:22s}  MAE(log)={mean_absolute_error(yte, yhat):.3f}   "
          f"R2(log)={r2_score(yte, yhat):+.3f}")

print("\n=== SEVERITY FORECAST (log1p counts) ===")
report("flat-mean baseline", b_flat)
report("seasonal climatology", b_seas)      # <- the bar that matters
report("weather model (XGB)", pred)

# real-count robust error
med_ae = np.median(np.abs(np.expm1(pred) - np.expm1(yte.to_numpy())))
med_ae_seas = np.median(np.abs(np.expm1(b_seas) - np.expm1(yte.to_numpy())))
print(f"\nMedian abs error (real counts):  model={med_ae:.1f}   "
      f"seasonal={med_ae_seas:.1f}")

# --- risk banding (low/med/high from TRAIN terciles) -----------------------
q1, q2 = train["pest_value"].quantile([1/3, 2/3])
def band(v): return np.where(v <= q1, 0, np.where(v <= q2, 1, 2))
true_b = band(np.expm1(yte.to_numpy()))
pred_b = band(np.expm1(pred))
band_acc = (true_b == pred_b).mean()
print(f"\nRisk-band accuracy (low/med/high, cutoffs {q1:.0f}/{q2:.0f}): {band_acc:.3f}")
print("Band confusion [rows=true, cols=pred] (0=low,1=med,2=high):")
print(pd.crosstab(pd.Series(true_b, name="true"),
                  pd.Series(pred_b, name="pred")).reindex(
                  index=[0,1,2], columns=[0,1,2], fill_value=0).to_string())

print("\nTop features:")
imp = pd.Series(reg.feature_importances_, index=feat_cols).sort_values(ascending=False)
print(imp.head(8).round(4).to_string())

reg.save_model(MODEL_OUT)
print(f"\nSaved regressor -> {MODEL_OUT}")