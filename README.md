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

## App

`src/app.py` is a three-tool Streamlit app (sidebar radio switcher) sharing `src/pipeline.py` for consistent feature engineering between training and inference.

- **Pest forecast (rice & cotton):** crop → location → pest selectors, week selector bounded to monitored windows, weather scenario sliders, seasonal profile chart, Kalman hidden-state chart, non-zero tercile risk banding (Low/Medium/High), measurement units per series
- **Plant health check (tomato):** live sensor sliders with in-range/out-of-range display per variable
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
- Maharashtra soybean leaf image dataset

<!-- TODO: add download links -->

## Status

Deployed and live on Streamlit Community Cloud, all three tracks working. Remaining open items: incorporate the Maharashtra soybean dataset, and add data source download links above.