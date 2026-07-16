from .preprocessing import df
from sklearn.model_selection import train_test_split

# training and test data 

X = df.drop(columns=['EmployeeID', 'Salary'])
y = df['Salary']

#Split data 80% training, 20% Testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""random_state = """
print(f"Total Rows in Dataset: {len(df)}")
print(f"Training Rows (X_train): {X_train.shape[0]}")
print(f"Testing Rows (X_test): {X_test.shape[0]}")
