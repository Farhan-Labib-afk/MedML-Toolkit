#!/usr/bin/env python3
"""Toy dataset demo for MedML Toolkit."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

from medml_toolkit import MedMLToolkit


def main() -> None:
    print("MedML Toolkit toy demo starting...")
    X, y = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=4,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y, name="target")
    print("Dataset ready (200 samples, 8 features)")

    classifier = MedMLToolkit(X_df, y_series)
    classifier.fit(fs_method="ANOVA")

    top5 = classifier.X_sorted.columns[:5]
    print("Top 5 features:")
    for i, feat in enumerate(top5, 1):
        print(f"  {i}. {feat}")

    best_features = classifier.X_sorted.iloc[:, :5]
    model = classifier.train(best_features, y_series, method="LR")
    results = classifier.cv_test(best_features, y_series, model, cv=5)

    print("\nPerformance Metrics:")
    print(f"  Accuracy   : {results['ACC']}")
    print(f"  Sensitivity: {results['SN']}")
    print(f"  Specificity: {results['SP']}")
    print(f"  MCC        : {results['MCC']}")
    print(f"  F1-Score   : {results['F1']}")

    disp = ConfusionMatrixDisplay(results["cm"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (Logistic Regression, Top 5 Features)")
    plt.show()

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

    print("\nDemo complete.")


if __name__ == "__main__":
    main()
