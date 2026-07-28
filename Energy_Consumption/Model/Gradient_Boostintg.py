from src.preprocess import *
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


X = df.drop('EnergyConsumption', axis=1)
y = df['EnergyConsumption']

print(X, y)

# Split data 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = GradientBoostingRegressor(n_estimators= 100, learning_rate=0.1, max_depth=5, random_state=42)

# train data 
model.fit(X_train, y_train)

# predict 
y_pred = model.predict(X_test)

input_data = pd.DataFrame({
    "Temperature": [45],
    "Hour": [12],
    "DayOfWeek": [3],
    "Appliances": [4]
})

prediction = model.predict(input_data)
print(prediction)

print(f"Prediction Energy consumption:, {prediction[0]:.2f}")

print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
