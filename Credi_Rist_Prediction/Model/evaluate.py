"""
evaluate.py

Evaluates the trained Random Forest model from Model/Random_Forest.py.

Usage:
    python -m Model.evaluate
    (run as a module from the project root, same way you run Random_Forest.py)

Assumes Model/Random_Forest.py exposes:
    rf, X_train, X_test, y_train, y_test, y_pred, risk_prob, positive_class_idx
"""
import matplotlib.plotly as plt
from Model.Random_Forest import (
    rf,
    X_train,
    X_test,
    y_train,
    y_test,
    y_pred,
    risk_prob,
    positive_class_idx,
)

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

# 1. Core Classification Metrics

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, risk_prob)

print("=" * 40)
print("Random Forest — Evaluation Metrics")
print("=" * 40)
print(f"Accuracy   : {accuracy:.4f}")
print(f"Precision  : {precision:.4f}")
print(f"Recall     : {recall:.4f}")
print(f"F1 Score   : {f1:.4f}")
print(f"ROC-AUC    : {roc_auc:.4f}")


# ============================
# 2. Full Classification Report
# ============================
 
print("\nClassification Report")
print("-" * 40)
print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))
 
# ============================
# 3. Confusion Matrix
# ============================
 
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print("-" * 40)
print(pd.DataFrame(
    cm,
    index=["Actual: No Default", "Actual: Default"],
    columns=["Predicted: No Default", "Predicted: Default"]
))
 
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["No Default", "Default"])
ax.set_yticklabels(["No Default", "Default"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.colorbar(im)
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("\nSaved: confusion_matrix.png")
 
# ============================
# 4. ROC Curve
# ============================
 
fpr, tpr, _ = roc_curve(y_test, risk_prob)
 
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()
print("Saved: roc_curve.png")
 
# ============================
# 5. Precision-Recall Curve
# ============================
# More informative than ROC when classes are imbalanced (defaults are the minority class)
 
precisions, recalls, _ = precision_recall_curve(y_test, risk_prob)
avg_precision = average_precision_score(y_test, risk_prob)
 
plt.figure(figsize=(5, 4))
plt.plot(recalls, precisions, label=f"PR Curve (AP = {avg_precision:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("precision_recall_curve.png")
plt.close()
print("Saved: precision_recall_curve.png")
 
# ============================
# 6. Feature Importance
# ============================
 
importances = pd.Series(rf.feature_importances_, index=X_train.columns)
importances = importances.sort_values(ascending=False)
 
print("\nTop 10 Feature Importances")
print("-" * 40)
print(importances.head(10))
 
plt.figure(figsize=(6, 5))
importances.head(10).sort_values().plot(kind="barh")
plt.xlabel("Importance")
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()
print("Saved: feature_importance.png")
 
# ============================
# 7. Cross-Validation (optional sanity check)
# ============================
# Checks whether performance is stable across different train/test splits,
# rather than relying on a single split.
 
from sklearn.model_selection import cross_val_score
 
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="roc_auc")
print("\n5-Fold Cross-Validation ROC-AUC")
print("-" * 40)
print("Scores :", np.round(cv_scores, 4))
print(f"Mean   : {cv_scores.mean():.4f}")
print(f"Std    : {cv_scores.std():.4f}")
 
print("\nEvaluation complete.")