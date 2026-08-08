"""
src/02_label_data.py
Track A, step 2: derive a transparent health label from the REAL sensor
columns. pH is dropped for cause (no sensor + noise values). This is a
proxy label built from agronomic ranges — document it as such.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "tomato_irrigation_dataset.csv"
OUT = ROOT / "data" / "processed" / "tomato_labeled.csv"

df = pd.read_csv(RAW)
df.columns = [c.strip() for c in df.columns]

rename_map = {
    "Temperature [_ C]": "temperature_c",
    "Humidity [%]": "humidity_pct",
    "Soil moisture": "soil_moisture_raw",
    "Reference evapotranspiration": "ref_et",
    "Evapotranspiration": "et",
    "Crop Coefficient": "crop_coeff",
    "Crop Coefficient stage": "growth_stage",
    "Nitrogen [mg/kg]": "nitrogen",
    "Phosphorus [mg/kg]": "phosphorus",
    "Potassium": "potassium",
    "Solar Radiation ghi": "solar_radiation",
    "Wind Speed": "wind_speed",
    "Days of planted": "days_planted",
    "pH": "ph",
}
df = df.rename(columns=rename_map)

# Drop pH for cause (see analysis: not a real sensor, noise values).
df = df.drop(columns=["ph"], errors="ignore")

# --- Editable agronomic ranges for tomato. Validate for your writeup. -------
# (low, high). A row is "at risk" if ANY of these falls outside its range.
THRESHOLDS = {
    "temperature_c": (18.0, 30.0),   # well-established; >30 heat stress
    "humidity_pct":  (50.0, 85.0),   # well-established; >85% fungal-disease risk
    "nitrogen":   (50.0, 250.0),     # rougher — widen/narrow as you validate
    "phosphorus": (20.0, 120.0),
    "potassium":  (40.0, 250.0),
}
# soil_moisture_raw is a raw ADC value (needs calibration) -> not thresholded.

present = {k: v for k, v in THRESHOLDS.items() if k in df.columns}
flags = pd.DataFrame(index=df.index)
for col, (lo, hi) in present.items():
    flags[f"{col}_ok"] = df[col].between(lo, hi)

df["stress_score"] = (~flags).sum(axis=1)
df["health_label"] = (df["stress_score"] == 0).astype(int)  # 1 = healthy, 0 = at risk

print("Thresholded variables:", list(present.keys()))
print("\nHealth label (1 = healthy, 0 = at risk):")
print(df["health_label"].value_counts().to_string())
print("\nAs proportions:")
print((df["health_label"].value_counts(normalize=True).round(3)).to_string())
print("\nStress-score distribution (# vars out of range):")
print(df["stress_score"].value_counts().sort_index().to_string())

df.to_csv(OUT, index=False)
print(f"\nSaved -> {OUT}")