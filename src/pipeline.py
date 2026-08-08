"""
src/pipeline.py
Shared preprocessing + prediction logic used by BOTH the training script
(06_pooled_model.py) and the Streamlit app (app.py). Keeping it in one place
guarantees the app builds features identically to how the model was trained.
No streamlit/xgboost imports here, so it's easy to test in isolation.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "crida_pest_weather_data.csv"

WEATHER = ["maxt_c","mint_c","rh1_pct","rh2_pct","rf_mm","ws_kmph","ssh_hrs","evp_mm"]
MIN_OBS, SPLIT_YEAR, ENDEMIC_THR = 150, 2006, 0.70

# fixed order of the 24 engineered weather features
BASE_FEATS = [f"{c}_{k}" for c in WEATHER for k in ("lag1","roll4","trend")]
FEAT_COLS_BASE = BASE_FEATS + ["standard_week"]     # before one-hot IDs


def load_clean():
    df = pd.read_csv(RAW)
    df["pest_value"] = pd.to_numeric(
        df["pest_value"].replace({"9..9": np.nan, "56.o": np.nan}), errors="coerce")
    for c in WEATHER:
        df[c] = pd.to_numeric(df[c].replace("-", np.nan), errors="coerce")
    df["week_id"] = df["year"]*52 + (df["standard_week"]-1)
    return df


def _location_features(df, loc):
    wx = df[df["location"] == loc].groupby("week_id")[WEATHER].mean()
    wx = wx.reindex(range(int(wx.index.min()), int(wx.index.max())+1))
    f = pd.DataFrame(index=wx.index)
    f["standard_week"] = (wx.index % 52) + 1
    for c in WEATHER:
        f[f"{c}_lag1"]  = wx[c].shift(1)
        f[f"{c}_roll4"] = wx[c].rolling(4).mean().shift(1)
        f[f"{c}_trend"] = wx[c].shift(1) - wx[c].shift(4)
    return f


def build_pooled(df):
    """Return (data, X) where data has metadata + target and X is the model matrix."""
    obs = df.dropna(subset=["pest_value"])
    cnt = obs.groupby(["crop","location","pest"]).size()
    keep = cnt[cnt >= MIN_OBS].index
    feat_by_loc = {loc: _location_features(df, loc) for loc in obs["location"].unique()}

    rows = []
    for crop, loc, pest in keep:
        tgt = (obs[(obs.location==loc) & (obs.pest==pest)]
               .groupby("week_id")["pest_value"].mean())
        d = feat_by_loc[loc].join(tgt.rename("pest_value"), how="inner")
        d = d[d["pest_value"].notna() & d["maxt_c_roll4"].notna()].copy()
        d["crop"], d["location"], d["pest"] = crop, loc, pest
        rows.append(d)
    data = pd.concat(rows).reset_index().rename(columns={"index":"week_id"})
    data["year"] = data["week_id"] // 52
    data["y"] = np.log1p(data["pest_value"])

    X = pd.get_dummies(data[FEAT_COLS_BASE + ["crop","location","pest"]],
                       columns=["crop","location","pest"])
    return data, X


# ---------------------------------------------------------------------------
# Prediction helpers (used by the app; model-agnostic — any .predict works)
# ---------------------------------------------------------------------------
def make_feature_row(bundle, crop, location, pest, week,
                     temp_off=0.0, rain_mult=1.0, humid_off=0.0):
    """Build a single-row feature frame from seasonal climatology + user offsets."""
    clim = bundle["feat_clim"].get((location, int(week)))
    if clim is None:
        clim = bundle["feat_clim_global"]
    row = dict(zip(BASE_FEATS, clim))
    row["standard_week"] = int(week)

    for f in list(row.keys()):
        if f.endswith(("lag1","roll4")):
            if f.startswith(("maxt_c","mint_c")): row[f] += temp_off
            elif f.startswith("rf_mm"):           row[f] = max(0.0, row[f]*rain_mult)
            elif f.startswith(("rh1_pct","rh2_pct")):
                row[f] = float(np.clip(row[f]+humid_off, 0, 100))

    row[f"crop_{crop}"] = 1
    row[f"location_{location}"] = 1
    row[f"pest_{pest}"] = 1
    return pd.DataFrame([row]).reindex(columns=bundle["feature_cols"], fill_value=0)


def predict_severity(model, bundle, crop, location, pest, week, **offsets):
    x = make_feature_row(bundle, crop, location, pest, week, **offsets)
    y_log = float(model.predict(x)[0])
    count = float(np.expm1(y_log))
    q1, q2 = bundle["band_cutoffs"].get((location, pest), (np.nan, np.nan))
    band = "Low" if count <= q1 else ("Medium" if count <= q2 else "High")
    endemic = bundle["endemic"].get((location, pest), None)
    return {"count": max(0.0, count), "band": band, "y_log": y_log,
            "endemic": endemic, "q1": q1, "q2": q2}


# ===========================================================================
# Track A (tomato) — transparent health check from live sensor readings.
# Thresholds mirror src/02_label_data.py. pH is excluded on purpose (the
# dataset's pH column had no sensor and impossible values). soil_moisture is
# a raw ADC value shown for reference only, not thresholded.
# ===========================================================================
TOMATO_THRESHOLDS = {
    "temperature_c": (18.0, 30.0),   # well-established comfort band
    "humidity_pct":  (50.0, 85.0),   # >85% raises fungal-disease risk
    "nitrogen":   (50.0, 250.0),     # rougher nutrient ranges
    "phosphorus": (20.0, 120.0),
    "potassium":  (40.0, 250.0),
}
TOMATO_LABELS = {
    "temperature_c": "Temperature (°C)",
    "humidity_pct":  "Humidity (%)",
    "nitrogen":      "Nitrogen (mg/kg)",
    "phosphorus":    "Phosphorus (mg/kg)",
    "potassium":     "Potassium (mg/kg)",
}


def tomato_health_check(readings: dict):
    """readings: {var: value}. Returns per-variable status + overall verdict."""
    rows, out_of_range = [], 0
    for var, (lo, hi) in TOMATO_THRESHOLDS.items():
        v = readings.get(var)
        ok = (v is not None) and (lo <= v <= hi)
        if not ok:
            out_of_range += 1
        rows.append({"variable": TOMATO_LABELS[var], "value": v,
                     "healthy_range": f"{lo:g} – {hi:g}", "ok": ok})
    status = "Healthy" if out_of_range == 0 else "At risk"
    return {"status": status, "out_of_range": out_of_range, "rows": rows}


# ===========================================================================
# Hidden-state estimation (Track B, subtask 5).
# A local-linear-trend Kalman filter + RTS smoother, run in log1p space, that
# recovers a smooth latent "true pest pressure" from the noisy weekly counts
# and returns a ±2σ uncertainty band. Standard Kalman filtering (the same thing
# filterpy would do); kept in plain numpy so it's portable and testable.
# ===========================================================================
def kalman_smooth(week_ids, values, q_level=0.02, q_slope=0.002, r_obs=0.35):
    """Return (est, lo, hi) latent pest-pressure estimates in COUNT units.
    week_ids: continuous weekly index (gaps allowed) · values: observed counts."""
    z = np.log1p(np.asarray(values, dtype=float))
    wk = np.asarray(week_ids, dtype=float)
    n = len(z)
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    H = np.array([[1.0, 0.0]]); R = np.array([[r_obs**2]])
    x = np.array([z[0], 0.0]); Pc = np.eye(2)
    xf = np.zeros((n, 2)); Pf = np.zeros((n, 2, 2))
    for i in range(n):
        if i > 0:                                   # predict (dt-aware for gaps)
            dt = max(1.0, wk[i] - wk[i-1])
            F = np.array([[1.0, dt], [0.0, 1.0]])
            Q = np.array([[q_level*dt, 0.0], [0.0, q_slope*dt]])
            x = F @ x; Pc = F @ Pc @ F.T + Q
        y = z[i] - (H @ x)[0]                        # update
        S = H @ Pc @ H.T + R
        K = (Pc @ H.T) @ np.linalg.inv(S)
        x = x + (K.flatten() * y)
        Pc = (np.eye(2) - K @ H) @ Pc
        xf[i] = x; Pf[i] = Pc
    # RTS smoother (backward pass) for the best retrospective estimate
    xs = xf.copy(); Ps = Pf.copy()
    for i in range(n-2, -1, -1):
        dt = max(1.0, wk[i+1] - wk[i])
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = np.array([[q_level*dt, 0.0], [0.0, q_slope*dt]])
        Ppred = F @ Pf[i] @ F.T + Q
        C = Pf[i] @ F.T @ np.linalg.inv(Ppred)
        xs[i] = xf[i] + C @ (xs[i+1] - F @ xf[i])
        Ps[i] = Pf[i] + C @ (Ps[i+1] - Ppred) @ C.T
    level = xs[:, 0]
    var_level = np.clip(Ps[:, 0, 0], 0, None)
    resid_var = float(np.var(z - level))            # observation-noise estimate
    std_pred = np.sqrt(var_level + resid_var)       # predictive std (calibrated)
    est = np.expm1(level)
    return est, np.expm1(level - 2*std_pred), np.expm1(level + 2*std_pred)