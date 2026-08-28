"""
src/14_train_netherlands.py
Track D, step 2: XGBoost harvest-rate model on the Autonomous Greenhouse
Challenge compartments, plus the app bundle for the "Tomato Netherlands" tab.

Validation is LEAVE-ONE-COMPARTMENT-OUT: train on five greenhouses, predict the
sixth, rotate. That is the honest test for this data — a random split would let
the model see the same greenhouse on both sides and flatter itself.

Headline finding, reported rather than buried: crop stage carries the signal and
weather/climate features do not improve on it. Every richer feature set was
tested below and scored the same or worse, so the shipped model is deliberately
the small one. See the note in the app's "limits" section.

Run:  python src/14_train_netherlands.py   (after 13_prepare_netherlands.py)
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

ROOT   = Path(__file__).resolve().parent.parent
DATA   = ROOT / "data" / "processed" / "netherlands_harvest.csv"
MODEL_OUT  = ROOT / "models" / "track_d_nl_reg.json"
BUNDLE_OUT = ROOT / "models" / "nl_bundle.pkl"

TARGET  = "ProdA_rate"                       # kg/m2 of class-A fruit per day
FEATURES = ["days_into_season", "interval_days"]          # chosen by the comparison below

# climate variables the app benchmarks the user against (reference panel, not model inputs)
ENVELOPE = ["Tair", "Rhair", "CO2air", "HumDef", "Tot_PAR"]

PARAMS = dict(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.9,
              colsample_bytree=0.8, reg_lambda=2.0, random_state=42)
N_SEEDS = 15          # feature-set gaps here are small; one seed cannot settle them


def loco_predict(d, feats, y, seed=None):
    """Leave-one-compartment-out predictions across all six greenhouses."""
    params = dict(PARAMS) if seed is None else {**PARAMS, "random_state": seed}
    pred = np.zeros(len(d))
    for team in d["team"].unique():
        te = (d["team"] == team).to_numpy()
        m = XGBRegressor(**params)
        m.fit(d.loc[~te, feats], y[~te])
        pred[te] = m.predict(d.loc[te, feats])
    return pred


def main():
    d = pd.read_csv(DATA)
    y = d[TARGET].to_numpy()
    print(f"Harvest events: {len(d)} across {d['team'].nunique()} compartments")

    # ---- feature-set comparison (documents why the model is this small) ------
    stage  = ["days_into_season", "interval_days"]
    light  = ["Iglob_mean_rec", "Iglob_mean_dev", "Tot_PAR_mean_rec", "Tot_PAR_mean_dev"]
    indoor = ["Tair_mean_rec", "Rhair_mean_rec", "CO2air_mean_rec", "HumDef_mean_rec"]
    every  = [c for c in d.columns if c not in ("team", "time", "ProdA", TARGET)]

    flat = np.zeros(len(d))
    for team in d["team"].unique():
        te = (d["team"] == team).to_numpy()
        flat[te] = y[~te].mean()

    print("\n=== leave-one-compartment-out (target: kg/m2/day) ===")
    print(f"  {'flat mean baseline':36s} k= 0  MAE={mean_absolute_error(y, flat):.4f}  "
          f"R2={r2_score(y, flat):+.3f}")
    for name, feats in [("crop stage only", stage),
                        ("stage + light", stage + light),
                        ("stage + light + indoor climate", stage + light + indoor),
                        ("stage + indoor climate", stage + indoor),
                        ("everything", every)]:
        p = loco_predict(d, feats, y)
        mark = "  <- shipped" if feats == FEATURES else ""
        print(f"  {name:36s} k={len(feats):2d}  MAE={mean_absolute_error(y, p):.4f}  "
              f"R2={r2_score(y, p):+.3f}{mark}")

    # ---- seed stability -----------------------------------------------------
    # The gap between the top two sets is a few thousandths of R2, which a single
    # seed cannot resolve. Repeating over N_SEEDS shows crop-stage-only ties
    # 'stage + light' on R2 but wins on MAE and is ~4x more stable, which is what
    # actually decides the shipped set. Indoor climate loses outright either way.
    print(f"\n=== seed stability: LOCO R2 over {N_SEEDS} seeds ===")
    print(f"  {'feature set':32s} {'mean':>7} {'sd':>6} {'MAE':>8}")
    for name, feats in [("crop stage only", stage),
                        ("stage + light", stage + light),
                        ("stage + light + indoor climate", stage + light + indoor),
                        ("stage + indoor climate", stage + indoor),
                        ("everything", every)]:
        preds = [loco_predict(d, feats, y, seed=s_) for s_ in range(N_SEEDS)]
        r2s = [r2_score(y, pp) for pp in preds]
        mae = float(np.mean([mean_absolute_error(y, pp) for pp in preds]))
        mark = "  <- shipped" if feats == FEATURES else ""
        print(f"  {name:32s} {np.mean(r2s):+7.3f} {np.std(r2s):6.3f} {mae:8.4f}{mark}")

    # ---- honest metrics for the shipped feature set --------------------------
    pred = loco_predict(d, FEATURES, y)
    mae, r2 = mean_absolute_error(y, pred), r2_score(y, pred)
    print(f"\n=== shipped model, per compartment ===")
    for team in d["team"].unique():
        te = (d["team"] == team).to_numpy()
        print(f"  {team:16s} n={te.sum():3d}  MAE={mean_absolute_error(y[te], pred[te]):.4f}  "
              f"R2={r2_score(y[te], pred[te]):+.3f}")
    print(f"\n  overall MAE={mae:.4f} kg/m2/day  ({mae / y.mean() * 100:.0f}% of the "
          f"{y.mean():.3f} mean)   R2={r2:+.3f}")

    # ---- final model: fit on all six compartments ----------------------------
    model = XGBRegressor(**PARAMS).fit(d[FEATURES], y)

    # ---- app bundle ----------------------------------------------------------
    med_interval = float(d["interval_days"].median())
    stage_grid = list(range(int(d["days_into_season"].min()),
                        int(d["days_into_season"].max()) + 1, 2))
    curve = model.predict(pd.DataFrame({"days_into_season": stage_grid,
                                        "interval_days": med_interval}))

    # climate envelope actually held by the six compartments, by crop-stage bin
    d["stage_bin"] = (d["days_into_season"] // 10 * 10).astype(int)
    envelope = {}
    for var in ENVELOPE:
        col = f"{var}_mean_rec"
        g = d.groupby("stage_bin")[col].quantile([0.1, 0.5, 0.9]).unstack()
        envelope[var] = {int(b): (float(r[0.1]), float(r[0.5]), float(r[0.9]))
                         for b, r in g.iterrows()}

    # "behind / on track / ahead" from the spread of LOCO residuals
    resid = y - pred
    r1, r2c = np.quantile(resid, [1/3, 2/3])

    bundle = {
        "feature_cols": FEATURES,
        "target": TARGET,
        "median_interval": med_interval,
        "stage_grid": stage_grid,
        "stage_curve": [float(v) for v in curve],
        "stage_min": int(d["days_into_season"].min()),
        "stage_max": int(d["days_into_season"].max()),
        "envelope": envelope,
        "envelope_vars": ENVELOPE,
        "resid_cutoffs": (float(r1), float(r2c)),
        "metrics": {"mae": float(mae), "r2": float(r2),
                    "mean_rate": float(y.mean()), "n": int(len(d)),
                    "n_compartments": int(d["team"].nunique())},
        "season_totals": {t: float(g["ProdA"].sum()) for t, g in d.groupby("team")},
        "observed": {t: (g["days_into_season"].tolist(), g[TARGET].tolist())
                     for t, g in d.groupby("team")},
        "teams": sorted(d["team"].unique().tolist()),
    }

    MODEL_OUT.parent.mkdir(exist_ok=True)
    model.save_model(MODEL_OUT)
    with open(BUNDLE_OUT, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nSaved model  -> {MODEL_OUT}")
    print(f"Saved bundle -> {BUNDLE_OUT}")


if __name__ == "__main__":
    main()
