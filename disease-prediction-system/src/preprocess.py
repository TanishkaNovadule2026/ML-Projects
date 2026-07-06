import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .config import DATA_FILE

try:
    # Read dataset
    df = pd.read_csv(DATA_FILE)
    print("Dataset loaded successfully.")

    # Check required column
    if 'disease' not in df.columns:
        raise KeyError("'disease' column not found in the dataset.")

    # Duplicate count
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate rows: {duplicate_count}")

    # Remove duplicates
    df = df.drop_duplicates()

    # Dataset information
    print(df.info())

    # Top 3 diseases
    print("Top 3 Diseases:")
    top_disease = df['disease'].value_counts().head(3)
    print(top_disease)

    # Label Encoding
    le = LabelEncoder()
    Disease_data = le.fit_transform(df['disease'])
    print("Encoded Disease Labels:")
    print(Disease_data)

except FileNotFoundError:
    print(f"Error: Dataset file '{DATA_FILE}' was not found.")

except pd.errors.EmptyDataError:
    print("Error: The CSV file is empty.")

except pd.errors.ParserError:
    print("Error: The CSV file is corrupted or has an invalid format.")

except KeyError as e:
    print(f"Column Error: {e}")

except Exception as e:
    print(f"Unexpected Error: {e}")