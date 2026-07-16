# data cleaning and preprocessing

from src.config import DATA_FILE
import pandas as pd
import numpy as np

# Load CSV file
df = pd.read_csv(DATA_FILE)

print("Dataset Loaded!")
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"Null values: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")

# Drop duplicate rows if any
if df.duplicated().any():
	df.drop_duplicates(inplace=True)

# Basic missing value handling: fill categorical with mode, numeric with median
for col in df.columns:
	if df[col].dtype == 'object' or str(df[col].dtype).startswith('category'):
		if df[col].isnull().any():
			df[col].fillna(df[col].mode()[0], inplace=True)
	else:
		if df[col].isnull().any():
			df[col].fillna(df[col].median(), inplace=True)

# Ensure Salary is numeric
if 'Salary' in df.columns:
	df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
	if df['Salary'].isnull().any():
		df['Salary'].fillna(df['Salary'].median(), inplace=True)

# Encode categorical columns (one-hot encoding)
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
if cat_cols:
	df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

print("Preprocessing complete.")
print(f"Final shape: {df.shape}")
