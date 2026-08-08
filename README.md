# Tatva Silicon Agri v1

Early detection and prediction of plant diseases and pests in Indian agriculture, with a focus on greenhouse monitoring. Built with free tools, deployable as a Streamlit app.

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

MobileNetV3-small fine-tuned on the Tamil Nadu Multi-Crop Disease Dataset (21,875 images, 30 disease classes across 5 crops, originally YOLO-format via Roboflow). A class-aware leakage check confirmed the train/valid/test split has no real photo leakage across splits (apparent duplicates were generic camera filenames colliding across different crops, not the same photo). Trained with inverse-frequency class weighting to handle imbalance. Test accuracy 96.0%, macro-F1 0.939 after 6 epochs on the M4 GPU (MPS backend).

The Maharashtra soybean leaf dataset is downloaded but not yet incorporated into this track.

## App

`src/app.py` is a Streamlit app sharing `src/pipeline.py` for consistent feature engineering between training and inference.

- **Pest page:** crop → location → pest selectors, week selector bounded to monitored windows, weather scenario sliders, seasonal profile chart, Kalman hidden-state chart, non-zero tercile risk banding (Low/Medium/High), measurement units per series
- **Tomato page:** live sensor sliders with in-range/out-of-range display per variable

## Setup

```bash
conda env create -f environment.yml
conda activate agri
streamlit run src/app.py
```

## Data

`data/raw/` and `data/processed/` are not tracked in this repo (too large for GitHub). Sources:

- Assam University tomato sensor dataset
- ICAR-CRIDA pest-weather archive
- Tamil Nadu multi-crop disease image dataset
- Maharashtra soybean leaf image dataset

<!-- TODO: add download links -->

## Status

Nearly feature-complete. Deployment to Hugging Face Spaces planned next.