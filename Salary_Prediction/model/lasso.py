import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from src.train import *

lasso_model = Lasso(alpha=1.0, max_iter=10000) # max_iter added to help convergence
lasso_model.fit(X_train, y_train)

# Predict on test data
y_pred_lasso = lasso_model.predict(X_test)

# Calculate scores
r2_lasso = r2_score(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
print("Lasso Regression ")
print(f"RMSE (Lasso): {rmse_lasso:.2f}\nR2 (Lasso): {r2_lasso:.4f}")

