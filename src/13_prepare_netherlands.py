"""
src/13_prepare_netherlands.py
Track D, step 1: turn the Autonomous Greenhouse Challenge (2nd edition) archive
into one harvest-level table.

The archive holds six greenhouse compartments (five autonomous-AI teams plus a
Reference compartment run by Dutch commercial growers) that each grew cherry
tomato in rockwool from 2019-12-16 to 2020-05-30. Climate is logged every five
minutes; fruit is harvested every three to five days. One row here = one harvest
event, with the climate that preceded it.

Two look-back windows are built per harvest:
  "rec"  since the previous harvest        -> conditions during ripening
  "dev"  35-55 days before the harvest     -> conditions while the truss was
                                              setting (the archive's own
                                              "truss development time" column
                                              runs 40-55 days)

Run:  python src/13_prepare_netherlands.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW  = ROOT / "data" / "raw" / "netherlands_agc"
OUT  = ROOT / "data" / "processed" / "netherlands_harvest.csv"

TEAMS = ["AICU", "Automatoes", "Digilog", "IUACAAS", "TheAutomators", "Reference"]
START_T = 43815.0          # first logged timestamp; Excel serial for 2019-12-16

# indoor climate + irrigation (process computer), rockwool slab, outside weather
CLIM = ["Tair", "Rhair", "CO2air", "HumDef", "Tot_PAR", "EC_drain_PC", "water_sup"]
SLAB = ["EC_slab1", "WC_slab1", "t_slab1"]
WX   = ["Tout", "Rhout", "Iglob", "Windsp"]

DEV_FROM, DEV_TO = 55, 35   # fruit-development window, days before harvest


def num(path):
    """Read a challenge CSV and coerce everything numeric ('NaN' strings included)."""
    return pd.read_csv(path, low_memory=False).apply(pd.to_numeric, errors="coerce")


def window_stats(src, cols, t0, t1, tag):
    """mean/min/max of `cols` over the half-open interval (t0, t1]."""
    w = src.loc[(src.index > t0) & (src.index <= t1), cols]
    if len(w) == 0:
        return {f"{c}_{s}_{tag}": np.nan for c in cols for s in ("mean", "min", "max")}
    out = {}
    for c in cols:
        out[f"{c}_mean_{tag}"] = w[c].mean()
        out[f"{c}_min_{tag}"]  = w[c].min()
        out[f"{c}_max_{tag}"]  = w[c].max()
    return out


def main():
    wx = num(RAW / "Weather" / "Weather.csv").set_index("%time").sort_index()

    rows = []
    for team in TEAMS:
        gh   = num(RAW / team / "GreenhouseClimate.csv").set_index("%time").sort_index()
        slab = num(RAW / team / "GrodanSens.csv").set_index("%time").sort_index()
        prod = num(RAW / team / "Production.csv").dropna(subset=["ProdA"]).sort_values("%time")

        times = prod["%time"].to_numpy()
        for i, t in enumerate(times):
            prev = times[i - 1] if i > 0 else t - 4.0     # first harvest: assume 4-day gap
            interval = t - prev
            r = {
                "team": team,
                "time": t,
                "dap": t - START_T,                       # crop stage, days into the season
                "interval_days": interval,
                "ProdA": prod["ProdA"].iloc[i],
                # kg/m2 PER DAY — harvest gaps vary 3-5 days, so the raw figure
                # would partly just measure how long it had been since the last pick
                "ProdA_rate": prod["ProdA"].iloc[i] / interval,
            }
            r.update(window_stats(gh,   CLIM, prev, t, "rec"))
            r.update(window_stats(slab, SLAB, prev, t, "rec"))
            r.update(window_stats(wx,   WX,   prev, t, "rec"))
            r.update(window_stats(gh,   CLIM, t - DEV_FROM, t - DEV_TO, "dev"))
            r.update(window_stats(wx,   WX,   t - DEV_FROM, t - DEV_TO, "dev"))
            rows.append(r)

    d = pd.DataFrame(rows)

    # Reference/Production.csv carries one stray record from 2019-03-05, months
    # before this crop was planted and with no climate behind it. Drop it.
    stray = (d["dap"] <= 0).sum()
    d = d[d["dap"] > 0].reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)

    print(f"Harvest events: {len(d)}  (dropped {stray} pre-season record)")
    print(f"Features built: {d.shape[1] - 6}")
    print("\nRows per compartment:")
    print(d["team"].value_counts().to_string())
    print("\nSeason totals (kg/m2 marketable, class A):")
    print(d.groupby("team")["ProdA"].sum().round(2).to_string())
    print("\nHarvest rate (kg/m2/day):")
    print(d["ProdA_rate"].describe().round(4).to_string())
    miss = d.columns[d.isna().any()].tolist()
    print(f"\nColumns with missing values: {len(miss)}")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
