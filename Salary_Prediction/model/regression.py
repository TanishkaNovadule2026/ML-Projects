# Train the model
import sys

from src.train import X_train, X_test, y_train, y_test
from sklearn.linear_model import LinearRegression
import pandas as pd 
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np



# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model training complete!")

# Test the model
y_pred = model.predict(X_test)

r2_reg = r2_score(y_test, y_pred)
rmse_reg = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n--- Model Performance Metrics ---")
print(f"R-squared (R2) Score: {r2_reg:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_reg:.2f}")

# 5. [Optional] Let's see actual vs predicted values side by side
comparison_df = pd.DataFrame({'Actual Salary': y_test, 'Predicted Salary': y_pred})
print("\nSample Predictions:")
print(comparison_df.head())


