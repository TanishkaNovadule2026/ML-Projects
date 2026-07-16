import joblib # Save bestmodel object 

from model.regression import r2_reg, rmse_reg
from model.ridge import r2_ridge, rmse_ridge
from model.lasso import r2_lasso, rmse_lasso

# Here we make a dictionary 
model_data = {
    'Linear Regression': {'r2': r2_reg, 'rmse': rmse_reg},
    'Lasso Regression':  {'r2': r2_lasso, 'rmse': rmse_lasso},
    'Ridge Regression':  {'r2': r2_ridge, 'rmse': rmse_ridge}
}



# Now find maximum linear regression on model data and min error 

best_r2_model = max(model_data, key=lambda x: model_data[x]['r2']) 
best_rmse_model = min(model_data, key=lambda x: model_data[x]['rmse'])

# best model ko object me change karna 
best_model_object = model_data[best_r2_model]['object']
# Save the name of best model 
joblib.dump(best_model_object, 'best_model.pkl')

joblib.dump({'name': best_r2_model, 'r2': model_data[best_r2_model]['r2']}, 'model_info.pkl')


for name, scores in model_data.items():
    print(f"{name}: R2 = {scores['r2']:.4f}, RMSE = {scores['rmse']:.4f}")

print(f"Max R2 Score: {best_r2_model} ({model_data[best_r2_model]['r2']:.4f})")
print(f" Min RMSE Value: {best_rmse_model} ({model_data[best_rmse_model]['rmse']:.4f})")

"""
import sys
import os
def main():
    # Ensure project root is on path
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from src.train import X_train, X_test, y_train, y_test
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np

    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=1.0, max_iter=10000)
    }

    results = []
    for name, m in models.items():
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results.append((name, r2, rmse))

    print("Model Evaluation Results")
    for name, r2, rmse in results:
        print(f"{name}: R2 = {r2:.4f}, RMSE = {rmse:.2f}")



if __name__ == '__main__':
    main()

"""