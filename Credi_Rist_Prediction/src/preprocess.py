from .config import _PROJECT_ROOT, DATA_FILE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# Write processed output to a separate file — never overwrite the raw source
PROCESSED_FILE = DATA_FILE.replace(".csv", "_processed.csv")


try:
    df = pd.read_csv(DATA_FILE)
    print("Data Loaded")
    clean_df = df.copy()

    print(df.info())
    print(df.isnull().sum())
    print(df[['person_emp_length', 'loan_intent', 'loan_int_rate']].head(20))


    

    # Clean data:
    # Fill null values
    # In loan_int_rate, fill null with mean grouped by loan_intent
    """df['loan_int_rate'] = (
            df.groupby('loan_intent')['loan_int_rate']
            .transform(lambda x: x.fillna(x.mean()))
        )
    print("Null values in loan_int_rate column is : ", df['loan_int_rate'].isnull().sum())
"""
    """df['person_emp_length'] = df['person_emp_length'].transform(lambda x: x.fillna(x.median()))
    print("Null values in person_emp_length column is : ", df['person_emp_length'].isnull().sum())
"""

        # Now detect outliers
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

        # IMPORTANT: exclude the target column (and any binary/ID columns) from
        # IQR-based outlier treatment — otherwise the minority class gets wiped out.
    exclude_from_outliers = ['loan_status']
    numeric_columns = numeric_columns.drop(
        [c for c in exclude_from_outliers if c in numeric_columns]
    )

    print(df['person_age'].sample(10))

        # Outlier detection
    outlier_summary = []
    for col in numeric_columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr

        outlier_count = ((df[col] < lower_limit) | (df[col] > upper_limit)).sum()

        clean_df = clean_df[
        (clean_df[col] >= lower_limit) &
        (clean_df[col] <= upper_limit)
        ]
        # Save cleaned dataset
        

        outlier_summary.append({
            "column": col,
            "Outlier Count": outlier_count
        })
    # Save cleaned dataset
    OUTPUT_FILE = os.path.join(
        _PROJECT_ROOT,
        "Data",
        "Encoded_Data",
        "credit_risk_dataset.csv"
    )
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    clean_df.to_csv(OUTPUT_FILE, index=False)

    print("Original rows:", len(df))
    print("Rows after removing outliers:", len(clean_df))
    

    """summary_df = pd.DataFrame(outlier_summary)
    print(summary_df)"""


    print(clean_df.isnull().sum())

    # Now Save encoded Data into another folder 
    # Create a copy of cleaned dataset
    encoded_df = clean_df.copy()

    
    # 1. Ordinal Encoding

    # person_home_ownership
    ownership_map = {
        "RENT": 0,
        "MORTGAGE": 1,
        "OWN": 2,
        "OTHER": 3
    }

    encoded_df["person_home_ownership"] = (
        encoded_df["person_home_ownership"].map(ownership_map)
    )
    print("Personal Home Owership ")
    print(encoded_df['person_home_ownership'].head(5))
    # loan_grade
    grade_map = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6
    }

    encoded_df["loan_grade"] = (
        encoded_df["loan_grade"].map(grade_map)
    )

    
    # 2. Label Encoding
    

    default_map = {
        "N": 0,
        "Y": 1
    }

    encoded_df["cb_person_default_on_file"] = (
        encoded_df["cb_person_default_on_file"].map(default_map)
    )

    
    # 3. One-Hot Encoding

    encoded_df = pd.get_dummies(
        encoded_df,
        columns=["loan_intent"],
        drop_first=True,
        dtype=int
    )

    
    # Save Encoded Dataset
    #

    OUTPUT_FILE = os.path.join(
        _PROJECT_ROOT,
        "Data",
        "Encoded_Data",
        "credit_risk_encoded.csv"
    )

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    encoded_df.to_csv(OUTPUT_FILE, index=False)

    print("Encoded dataset saved successfully!")
    print(encoded_df.head())
    
    

except FileNotFoundError:
    print(f"{DATA_FILE} Not found")
except pd.errors.EmptyDataError:
    print(f"File not contain any content")
except KeyError as e:
    print(f"{e} Key Error ")
except Exception as e:
    print(f"{e} Exception occurred")