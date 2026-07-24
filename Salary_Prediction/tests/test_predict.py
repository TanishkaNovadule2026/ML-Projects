import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src import predict as predict_module
from src.predict import _build_input, predict_salary
from src.train import X_train


class PredictInputTests(unittest.TestCase):
    def test_build_input_matches_training_columns(self):
        X_input = _build_input(5, "Bachelor", "Data Analyst", "Delhi", 6, 23)
        # Is the returned object a dataframe?
        self.assertIsInstance(X_input, pd.DataFrame)
        # this check one row should be created 
        self.assertEqual(X_input.shape[0], 1)
        #Both list are identical like feature in test data same a training dataset feature. If one column is missing then prediction fails 
        self.assertListEqual(list(X_input.columns), list(X_train.columns))
# 
    def test_prediction_is_not_negative(self):
        pred, _ = predict_salary(5, "Bachelor", "Data Analyst", "Delhi", 6, 23)
        self.assertGreaterEqual(pred, 0)
# instead of using real ML model. It create a fake model 
    def test_prediction_returns_loaded_model_name(self):
        class DummyModel:
            def predict(self, _input_df):
                return np.array([500000.0])

        original_model = predict_module.model
        original_model_name = getattr(predict_module, "model_name", None)
        predict_module.model = DummyModel()
        predict_module.model_name = "Ridge Regression"

        try:
            _, model_name = predict_salary(5, "Bachelor", "Data Analyst", "Delhi", 6)
            self.assertEqual(model_name, "Ridge Regression")
        finally:
            predict_module.model = original_model
            if original_model_name is None:
                delattr(predict_module, "model_name")
            else:
                predict_module.model_name = original_model_name

    def test_negative_model_prediction_uses_fallback_salary(self):
        class DummyModel:
            def predict(self, _input_df):
                return np.array([-1000.0])

        original_model = predict_module.model
        original_model_name = getattr(predict_module, "model_name", None)
        predict_module.model = DummyModel()
        predict_module.model_name = "Ridge Regression"

        try:
            pred, model_name = predict_salary(5, "Bachelor", "Data Analyst", "Delhi", 6)
            self.assertGreater(pred, 0)
            self.assertIn("fallback", model_name.lower())
        finally:
            predict_module.model = original_model
            if original_model_name is None:
                delattr(predict_module, "model_name")
            else:
                predict_module.model_name = original_model_name


if __name__ == "__main__":
    unittest.main()
