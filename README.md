# Tatva Silicon Agri v1

Early detection and prediction of plant diseases and pests in Indian agriculture, with a focus on greenhouse monitoring. Built with free tools, deployed as a Streamlit app.

**Live app:** `<your streamlit.app URL here>`

## Project structure

```
models/     trained model artifacts (XGBoost, regression, classifier, MobileNetV3)
src/        pipeline scripts and Streamlit app
data/       raw + processed data (not tracked in git — see Data below)
```

## Track A — Tomato sensor health

XGBoost classifier trained on Assam University sensor data (temperature, humidity, soil moisture, N-P-K, evapotranspiration). Labels are derived from agronomic thresholds on the same features used for prediction, so the reported 100% accuracy is a check of threshold-recovery, not a real-world validation metric — documented here rather than treated as a result. pH was excluded due to implausible sensor values.

## Track B — Pest severity forecasting

Pooled severity regression across 38 viable series from the ICAR-CRIDA archive: rice and cotton, 12 Indian stations, ~1959–2011, 39 pests, 8 weather variables. Uses lag/rolling weather features per location, a global time-based train/test split (train ≤2005, test 2006+), and per-series seasonal climatology as the baseline. Weather features meaningfully improve prediction for episodic pests but add little over the seasonal baseline for endemic pests. A Kalman filter (local-linear-trend model with RTS smoother) adds a denoised latent pest-pressure trajectory with a calibrated ±2σ uncertainty band.

## Track C — Multi-crop disease image classification

MobileNetV3-small fine-tuned on the Tamil Nadu Multi-Crop Disease Dataset (21,875 images, 30 disease classes across 5 crops, originally YOLO-format via Roboflow). A class-aware leakage check confirmed the train/valid/test split has no real photo leakage across splits (apparent duplicates were generic camera filenames colliding across different crops, not the same photo). Trained with inverse-frequency class weighting to handle imbalance. Test accuracy 96.0%, macro-F1 0.939 after 6 epochs on the M4 GPU (MPS backend, training only — inference runs on CPU).

**Non-leaf / out-of-distribution inputs:** the classifier is closed-set — it always returns its best-ranked guess among the 30 trained classes, even for images that aren't leaves at all. There's no "I don't recognize this" output. As a rough safety net, the app shows a low-confidence warning whenever the top prediction is below 50% confidence, which usually (not always) catches non-leaf or unsupported-crop photos. This threshold is a reasonable default, not a validated cutoff — it hasn't been tested against a labeled set of should-warn / shouldn't-warn cases.

The Maharashtra soybean leaf dataset is downloaded but not yet incorporated into this track.

## Track D — Tomato Netherlands (harvest benchmark)

A **separate** model from Track A, on a separate dataset, deployed as its own tab. Track A is an Indian soil-grown tomato rule check; this is a Dutch hydroponic yield benchmark. Neither model touches the other and neither was overwritten.

Source: the **Autonomous Greenhouse Challenge, 2nd edition** (Wageningen UR, Bleiswijk) — six greenhouse compartments growing cherry tomato in rockwool from 2019-12-16 to 2020-05-30, five run by autonomous-AI teams and one Reference compartment run by Dutch commercial growers. Climate is logged every 5 minutes (47,809 rows per compartment); fruit is harvested every 3–5 days. `13_prepare_netherlands.py` reduces this to 139 harvest events with two look-back windows per harvest (since the previous pick, and the 35–55 day fruit-development window).

XGBoost regression on class-A harvest rate (kg/m²/day — normalised because harvest gaps vary 3–5 days). Validated **leave-one-compartment-out**: train on five greenhouses, predict the sixth, rotate. **MAE 0.0311 kg/m²/day against a 0.134 mean (~23% error), R² +0.57** over 139 harvests.

`days_into_season` counts from 2019-12-16, when logging began and the plants went into the greenhouse. It is **not** days since sowing — the plants were raised in a propagator first, and the archive never states a sowing date.

**The model uses crop stage only, and that is a finding rather than a shortcut.** Every richer feature set was tested and none beat it:

Averaged over 15 random seeds, because the gap between the top two sets is a few thousandths of R² and one seed cannot resolve it:

| features | k | MAE | R² mean | R² sd |
|---|---|---|---|---|
| flat mean baseline | 0 | 0.0451 | −0.004 | — |
| **crop stage only** | **2** | **0.0310** | **+0.571** | **0.002** |
| stage + light | 6 | 0.0324 | +0.573 | 0.008 |
| stage + light + indoor climate | 10 | 0.0320 | +0.550 | 0.007 |
| stage + indoor climate | 6 | 0.0333 | +0.505 | 0.011 |
| everything | 77 | 0.0330 | +0.527 | 0.014 |

Indoor climate loses outright. Outside light ties crop-stage-only on R² (+0.573 vs +0.571, inside its own 0.008 spread) but is worse on average error and four times less stable, so the two-feature model ships.

After accounting for crop stage there is **no statistically detectable difference between the six compartments at all** (ANOVA on residuals, F=0.80, p=0.55). All six were run by expert controllers inside a narrow, near-optimal envelope — 2.3 °C, 5.4% RH and 156 ppm CO₂ separated the extremes, and season totals spanned just 11% (12.89–14.36 kg/m²). There is very little variation in either climate or yield for a model to learn from, so this ships as an honest benchmark curve, not a climate-driven yield predictor. Climate is handled in the app as a separate percentile reference check against the band the six compartments actually held, clearly labelled as not feeding the prediction.

Scope limit: a Dutch high-tech glasshouse, hydroponic rockwool, cherry tomato, winter–spring at 52°N with supplemental HPS lighting and CO₂ dosing. It does not transfer directly to a soil-grown Indian greenhouse.

**Why this dataset could not simply retrain Track A:** the challenge greenhouses are hydroponic, so there is no soil-nutrient measurement anywhere in the archive. Three of Track A's five thresholded variables (N, P, K in mg/kg) have no counterpart — the only nutrient data is 10 manual lab samples per compartment in mmol/L of nutrient *solution*. `ref_et`, `et`, `crop_coeff` and `growth_stage` are also absent. Hence a new model rather than a weight update.

## App

`src/app.py` is a four-tool Streamlit app (sidebar radio switcher) sharing `src/pipeline.py` for consistent feature engineering between training and inference.

- **Pest forecast (rice & cotton):** crop → location → pest selectors, week selector bounded to monitored windows, weather scenario sliders, seasonal profile chart, Kalman hidden-state chart, non-zero tercile risk banding (Low/Medium/High), measurement units per series
- **Plant health check (tomato):** live sensor sliders with in-range/out-of-range display per variable
- **Tomato Netherlands (harvest benchmark):** crop-stage slider → expected class-A harvest rate, benchmark curve across the season with every real harvest from the six Dutch compartments overlaid, a behind/on-track/ahead standing if you enter what you actually picked, and a climate-envelope reference check against the p10–p90 band the expert compartments held at that stage
- **Leaf disease ID (photo):** upload a leaf photo, get the predicted crop + disease with a confidence score, top-3 breakdown, and a low-confidence warning (see Track C note above)

## Setup (local development)

```bash
conda env create -f environment.local.yml
conda activate agri
streamlit run src/app.py
```

## Deployment

Deployed on **Streamlit Community Cloud**, connected to this GitHub repo (`main` branch, entrypoint `src/app.py`, Python 3.11).

A couple of deployment-specific notes, in case this ever needs redoing:

- **Dependencies use `requirements.txt` (pip), not `environment.local.yml` (conda).** Streamlit Community Cloud auto-detects a file literally named `environment.yml` and defaults to conda if present, which reliably hangs or gets killed on their build infra ("Solving environment" stuck for hours is a known, widely-reported issue). The local conda env file is deliberately named `environment.local.yml` so it's excluded from that auto-detection while still being available for local dev reproducibility.
- **`requirements.txt` pins a CPU-only torch build** via `--extra-index-url https://download.pytorch.org/whl/cpu`, since Community Cloud runs CPU-only Linux servers and the default PyPI wheel bundles unnecessary CUDA support. This doesn't change model behavior — the app's inference code already runs on CPU locally too (`torch.load(..., map_location="cpu")`); MPS was only ever used during training.
- **`scikit-learn` is required** even though the app doesn't call it directly — `XGBRegressor` is xgboost's sklearn-compatible wrapper class and depends on sklearn just to exist.
- Community Cloud's free tier caps around ~1GB memory; worth monitoring given torch + xgboost + pandas all load together.

## Data

`data/raw/` and `data/processed/` are not tracked in this repo (too large for GitHub). Sources:

- Assam University tomato sensor dataset
- ICAR-CRIDA pest-weather archive
- Tamil Nadu multi-crop disease image dataset
- Autonomous Greenhouse Challenge 2nd edition (WUR) — expected at `data/raw/netherlands_agc/`
- Maharashtra soybean leaf image dataset

<!-- TODO: add download links -->

## Status

Deployed and live on Streamlit Community Cloud, all four tracks working. Remaining open items: incorporate the Maharashtra soybean dataset, and add data source download links above.