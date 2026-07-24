import os 
# Go to the parent folder of the parent folder of the current file 
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_FILE = os.path.join(_PROJECT_ROOT, "Data", "credit_risk_dataset.csv")


