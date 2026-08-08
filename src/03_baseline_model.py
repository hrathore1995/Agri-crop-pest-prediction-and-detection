"""
src/03_baseline_model.py
Track A, step 3: baseline classifier on the labeled sensor data.
Purpose: validate the full pipeline. High accuracy is EXPECTED here
because the label is a deterministic function of some of the features
(see notes in chat) — this is not evidence of real predictive power.
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "tomato_labeled.csv"
MODEL_OUT = ROOT / "models" / "track_a_xgb.json"

df = pd.read_csv(DATA)

TARGET = "health_label"
LEAK = ["stress_score"]          # derived from the label -> must NOT be a feature
drop_cols = [TARGET] + LEAK

# One-hot encode the categorical growth stage; keep all numeric features.
X = df.drop(columns=drop_cols)
X = pd.get_dummies(X, columns=["growth_stage"], drop_first=True)
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    eval_metric="logloss", random_state=42
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Test accuracy:", round((pred == y_test).mean(), 4))
print("\nClassification report:")
print(classification_report(y_test, pred, target_names=["at risk (0)", "healthy (1)"]))
print("Confusion matrix [rows=true, cols=pred]:")
print(confusion_matrix(y_test, pred))

print("\nTop feature importances:")
imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp.head(10).round(4).to_string())

model.save_model(MODEL_OUT)
print(f"\nSaved model -> {MODEL_OUT}")