"""
Train Linear, Ridge, and Lasso models and save them to disk.
Keeps comments for clarity as requested.
"""
import os
import json
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

from src.train import X_train, X_test, y_train, y_test


# Directory to save artifacts
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Define models to train
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=1.0, max_iter=10000)
}

metrics = {}

for name, m in models.items():
    # Train
    m.fit(X_train, y_train)

    # Predict
    y_pred = m.predict(X_test)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    metrics[name] = {'r2': float(r2), 'rmse': rmse}

    # Save model
    model_path = os.path.join(ARTIFACT_DIR, f"{name}.joblib")
    joblib.dump(m, model_path)
    print(f"Saved {name} to {model_path}")

# Save metrics
metrics_path = os.path.join(ARTIFACT_DIR, 'metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)

print("Training complete. Metrics:")
print(json.dumps(metrics, indent=2))
