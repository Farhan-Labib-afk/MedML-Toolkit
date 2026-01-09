#!/usr/bin/env python3
"""Run MedML Toolkit on a real, built-in dataset (no downloads)."""

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

from medml_toolkit import MedMLToolkit


def main() -> None:
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    print("MedML Toolkit demo: Breast Cancer Wisconsin")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")

    classifier = MedMLToolkit(X, y)

    # Rank features and run incremental feature selection with LR.
    classifier.fit(fs_method="ANOVA", ifs_method="LR", ifs_grid=False, ifs_cv=5)

    best_features = classifier.transform(X, evaluate="F1")
    model = classifier.train(best_features, y, grid=False, method="LR", cv=5)
    results = classifier.cv_test(best_features, y, model, cv=5)

    print("\nBest CV Metrics (5-fold):")
    print(f"  Accuracy   : {results['ACC']}")
    print(f"  Sensitivity: {results['SN']}")
    print(f"  Specificity: {results['SP']}")
    print(f"  F1-Score   : {results['F1']}")
    print(f"  MCC        : {results['MCC']}")

    output_dir = Path("outputs") / "breast_cancer"
    output_dir.mkdir(parents=True, exist_ok=True)

    classifier.plot_correlation(X, y, out=str(output_dir / "correlations.png"))
    classifier.plot_cluster(X, y, cluster="PCA", out=str(output_dir / "pca.png"))
    classifier.plot_lines(classifier.ifs_results, out=str(output_dir / "feature_selection.png"))
    classifier.plot_roc(best_features, y, model, cv=5, out=str(output_dir / "roc.png"))
    classifier.plot_prc(best_features, y, model, cv=5, out=str(output_dir / "prc.png"))

    print("\nSaved plots:")
    print(f"  {output_dir / 'correlations.png'}")
    print(f"  {output_dir / 'pca.png'}")
    print(f"  {output_dir / 'feature_selection.png'}")
    print(f"  {output_dir / 'roc.png'}")
    print(f"  {output_dir / 'prc.png'}")


if __name__ == "__main__":
    main()
