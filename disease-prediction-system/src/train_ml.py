# train Data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from .preprocess import df

# Features are all columns except the target 'disease'
X = df.drop(columns=['disease'])
y = df['disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
