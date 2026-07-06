# Evaluate Model accuracy
from .train_ml import rf_model, X_test, y_test
from sklearn.metrics import accuracy_score
from .predict import predicted_disease, all_probabilities, class_labels

try:
    # Predict on test dataset
    y_pred = rf_model.predict(X_test)

    # Calculate and display model accuracy
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")

    # Map classes to their respective probabilities
    confidence_dict = dict(zip(class_labels, all_probabilities))

    # Get confidence of predicted disease
    highest_confidence = confidence_dict[predicted_disease]

    print(f"Predicted Disease: {predicted_disease}")
    print(f"Confidence Level: {highest_confidence * 100:.2f}%")

    # print(f"\nFull Probability Breakdown: {confidence_dict}")

except AttributeError as e:
    print(f"Model Error: {e}")

except ValueError as e:
    print(f"Value Error: {e}")

except KeyError as e:
    print(f"Prediction Error: {e}")

except ImportError as e:
    print(f"Import Error: {e}")

except Exception as e:
    print(f"Unexpected Error: {e}")