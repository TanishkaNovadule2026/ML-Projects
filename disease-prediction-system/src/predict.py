from .train_ml import rf_model, X_test, y_test
import numpy as np

try:
    # Example new patient features (shape: 1 x n_features)
    new_patient = np.array([[1, 1, 1, 0, 0, 1, 1]])

    #[0] -> Extract the probability/prediction value for the first (and only) patient
    predicted_disease = rf_model.predict(new_patient)[0]

    # Find confidence level (probabilities for all possible classes)
    all_probabilities = rf_model.predict_proba(new_patient)[0]

    # List of all class labels the RF model learned during training
    class_labels = rf_model.classes_

    # print("class labels are : \n", class_labels)

    # Map each disease with its probability
    confidence_dict = dict(zip(class_labels, all_probabilities))

    # Get confidence of predicted disease
    highest_confidence = confidence_dict[predicted_disease]

except AttributeError as e:
    print(f"Model Error: {e}")

except ValueError as e:
    print(f"Input Error: {e}")

except IndexError as e:
    print(f"Prediction Error: {e}")

except Exception as e:
    print(f"Unexpected Error: {e}")