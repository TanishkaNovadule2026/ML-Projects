# data cleaning and encoding
"""
Education : ordinal encoding
JobRole : one-hot encoding
Location : label encoding
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder

from src.config import DATA_FILE


def preprocess_data(data_file=DATA_FILE):
    """Load the dataset, clean it, and return encoded features ready for training."""
    df = pd.read_csv(data_file)

    print("Dataset Loaded")
    print(df.isnull().sum())
    print("Duplicate rows:", df.duplicated().sum())
    print(df.info())

    df = df.copy()

    # Remove duplicates if any
    if df.duplicated().any():
        df = df.drop_duplicates()

    # Fill missing values
    for col in df.columns:
        if df[col].dtype == "object" or str(df[col].dtype).startswith("category"):
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])
        else:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

    # Convert salary to numeric
    if "Salary" in df.columns:
        df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
        if df["Salary"].isnull().any():
            df["Salary"] = df["Salary"].fillna(df["Salary"].median())

    # Encode Education using ordinal order
    education_order = ["High School", "Bachelor", "Master", "PhD"]
    ordinal_encoder = OrdinalEncoder(categories=[education_order])
    df["Education_Encoded"] = ordinal_encoder.fit_transform(df[["Education"]]).ravel()

    # Encode Location using label encoding
    label_encoder = LabelEncoder()
    df["Location_Encoded"] = label_encoder.fit_transform(df["Location"])

    # One-hot encode JobRole
    oh_encoder = OneHotEncoder(sparse_output=False, drop="first")
    job_encoded_array = oh_encoder.fit_transform(df[["JobRole"]])
    job_encoded_columns = oh_encoder.get_feature_names_out(["JobRole"])
    df_job_encoded = pd.DataFrame(job_encoded_array, columns=job_encoded_columns, dtype=int)

    # Combine and remove original string columns
    df = pd.concat([df.reset_index(drop=True), df_job_encoded.reset_index(drop=True)], axis=1)
    df = df.drop(columns=["Education", "Location", "JobRole"])

    return df


# Create a dataframe object at import time so other modules can use it.
df = preprocess_data()


if __name__ == "__main__":
    print("Preprocessing complete")
    print(df.head())
    print("Shape:", df.shape)