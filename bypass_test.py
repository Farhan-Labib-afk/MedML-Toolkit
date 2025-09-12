#!/usr/bin/env python3
"""
Demo script for AIpmt
Run this file directly to test AIpmt on a toy dataset.
Perfect for showcasing results (e.g., LinkedIn screenshots).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve, average_precision_score
from tqdm import tqdm

# Import your AIpmt class here (or paste inside if testing standalone)
from aipmt.main import AIpmt   # ✅ Uses your main.py directly

print("🎯 AIpmt Demo Starting...\n")

# -------------------------------------------------
# 1. Create Toy Dataset
# -------------------------------------------------
print("📊 Generating toy dataset...")
X, y = make_classification(
    n_samples=200,
    n_features=8,
    n_informative=4,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)
X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
y_series = pd.Series(y, name="target")
print("✅ Dataset ready (200 samples, 8 features)\n")

# -------------------------------------------------
# 2. Initialize AIpmt
# -------------------------------------------------
print("🤖 Initializing AIpmt...")
classifier = AIpmt(X_df, y_series)
print("✅ AIpmt initialized!\n")

# -------------------------------------------------
# 3. Feature Selection
# -------------------------------------------------
print("🔍 Running ANOVA feature selection...")
classifier.fit(fs_method="ANOVA")
print("✅ Features ranked by importance!\n")

# Show top features
print("📈 Top 5 Features by ANOVA:")
top5 = classifier.X_sorted.columns[:5]
for i, feat in enumerate(top5, 1):
    print(f"   {i}. {feat}")

# -------------------------------------------------
# 4. Train & Evaluate
# -------------------------------------------------
print("\n⚡ Training Logistic Regression on top 5 features...")
best_features = classifier.X_sorted.iloc[:, :5]
model = classifier.train(best_features, y_series, method='LR')
results = classifier.cv_test(best_features, y_series, model, cv=5)

print("✅ Model trained & evaluated!\n")

print("📊 Performance Metrics:")
print(f"   Accuracy   : {results['ACC']}")
print(f"   Sensitivity: {results['SN']}")
print(f"   Specificity: {results['SP']}")
print(f"   MCC        : {results['MCC']}")
print(f"   F1-Score   : {results['F1']}")

# -------------------------------------------------
# 5. Visualization
# -------------------------------------------------
print("\n📉 Plotting results...")

# --- Confusion Matrix ---
disp = ConfusionMatrixDisplay(results['cm'])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix (Logistic Regression, Top 5 Features)")
plt.show()

# --- ROC Curve ---
y_proba = model.predict_proba(best_features)[:, 1]
fpr, tpr, _ = roc_curve(y_series, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# --- Precision-Recall Curve ---
precision, recall, _ = precision_recall_curve(y_series, y_proba)
avg_prec = average_precision_score(y_series, y_proba)

plt.figure()
plt.plot(recall, precision, color="green", lw=2, label=f"AP = {avg_prec:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

print("\n🎉 Demo complete! You can screenshot the metrics + plots above.")
