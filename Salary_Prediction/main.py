import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("Loading data...")
from src.preprocessing import df


print("Training model...")
from model.regression import *
print()
from model.evaluate_models import *
print()

from model.ridge import *
print()

from model.lasso import *




