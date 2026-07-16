import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from src.train import *

import joblib
import os


# --- 1. RIDGE REGRESSION ---
# alpha=1.0 is the default penalty strength. Higher values = stronger regularization.
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# Predict on test data
y_pred_ridge = ridge_model.predict(X_test)

# Calculate scores
r2_ridge = r2_score(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
print("Ridge Regression model")
print(r2_ridge, rmse_ridge)
