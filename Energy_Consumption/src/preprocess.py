import pandas as pd
from .config import DATA_FILE

try:
    df = pd.read_csv(DATA_FILE)
    print(df.info())



  
except Exception as e: 
    print(f"{e}")