"""Core MedML Toolkit model and plotting utilities."""

from copy import deepcopy
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    average_precision_score,
    auc,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm


def _round_or_none(value: Optional[float], ndigits: int = 3) -> Optional[float]:
    return None if value is None else round(value, ndigits)


class MedMLToolkit:
    """Precision medicine modeling toolkit with feature selection and plots."""

    def __init__(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.X = X
        self.y = y
        self.feature_importance = {
            "ANOVA": f_classif,
            "Chi2": chi2,
        }
        self.train_model_list = {
            "LR": LogisticRegression(solver="liblinear", max_iter=5000, random_state=42),
            "SVM": svm.SVC(kernel="rbf", probability=True, random_state=42),
        }
        self.param_grid = {
            "LR": {"C": [0.001, 0.01, 0.1, 1, 10], "penalty": ["l1", "l2"]},
            "SVM": {
                "C": list(2 ** i for i in range(-5, 15 + 1, 2)),
                "gamma": list(2 ** i for i in range(-15, 3 + 1, 2)),
            },
        }

    def fit(
        self,
        fs_method: str = "ANOVA",
        ifs_method: Optional[str] = None,
        ifs_grid: bool = False,
        ifs_cv: int = 5,
        show_progress: bool = True,
    ) -> None:
        if fs_method not in self.feature_importance:
            raise ValueError(f"Unknown fs_method: {fs_method}")
        if ifs_method is not None and ifs_method not in self.train_model_list:
            raise ValueError(f"Unknown ifs_method: {ifs_method}")

        scaler = MinMaxScaler() if fs_method == "Chi2" else StandardScaler()
        self.scaler = scaler
        self.X = pd.DataFrame(scaler.fit_transform(self.X), columns=self.X.columns)

        self.correlations = self.X.corrwith(self.y)
        self.fs_scores = self.feature_importance[fs_method](self.X, self.y)[0]

        sorted_indices = self.fs_scores.argsort()
        self.X_sorted = self.X.iloc[:, sorted_indices[::-1]]

        if ifs_method:
            results = []
            for i in tqdm(
                range(1, len(self.X_sorted.columns) + 1), disable=not show_progress
            ):
                selected_features = self.X_sorted.iloc[:, :i]
                model = self.train(
                    selected_features, self.y, grid=ifs_grid, method=ifs_method, cv=ifs_cv
                )
                cv_result = self.cv_test(selected_features, self.y, model, cv=ifs_cv)
                results.append(
                    {
                        "Num Features": i,
                        "ACC": cv_result["ACC"],
                        "SN": cv_result["SN"],
                        "SP": cv_result["SP"],
                        "MCC": cv_result["MCC"],
                        "F1": cv_result["F1"],
                    }
                )
            self.ifs_results = pd.DataFrame(results)

    def transform(self, X: pd.DataFrame, evaluate: str = "F1") -> pd.DataFrame:
        if hasattr(self, "scaler"):
            X = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        else:
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        if not hasattr(self, "X_sorted"):
            return X

        if not hasattr(self, "ifs_results") or evaluate not in self.ifs_results.columns:
            return X[self.X_sorted.columns]

        metric_series = self.ifs_results[evaluate].dropna()
        if metric_series.empty:
            return X[self.X_sorted.columns]

        best_num = self.ifs_results["Num Features"][metric_series.idxmax()]
        return X[self.X_sorted.iloc[:, :best_num].columns]

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        grid: bool = False,
        method: str = "LR",
        cv: int = 5,
    ):
        model = deepcopy(self.train_model_list[method])
        if grid:
            model = GridSearchCV(
                model,
                self.param_grid[method],
                cv=cv,
                scoring="accuracy",
                return_train_score=False,
                n_jobs=1,
            ).fit(X, y)
        else:
            model.fit(X, y)
        return model

    def cv_test(self, X: pd.DataFrame, y: pd.Series, model, cv: int = 5) -> dict:
        y_pred = cross_val_predict(model, X, y, cv=cv)
        cm, acc, sn, sp, mcc, f1 = self.calculate_metrics(y, y_pred)
        return {
            "ACC": _round_or_none(acc),
            "SN": _round_or_none(sn),
            "SP": _round_or_none(sp),
            "MCC": _round_or_none(mcc),
            "F1": _round_or_none(f1),
            "cm": cm.values,
        }

    def predict(self, X: pd.DataFrame, model, threshold: float = 0.5):
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X)
            y_pred = (y_pred_proba[:, 1] > threshold).astype(int)
            return y_pred, y_pred_proba
        y_pred = model.predict(X)
        return y_pred, None

    def calculate_metrics(
        self, y: pd.Series, y_pred: np.ndarray
    ) -> Tuple[pd.DataFrame, float, Optional[float], Optional[float], Optional[float], Optional[float]]:
        unique_labels = np.unique(np.concatenate([np.asarray(y), np.asarray(y_pred)]))
        if unique_labels.size == 2:
            labels_in_y = np.unique(np.asarray(y))
            labels = list(labels_in_y)
            cm_array = confusion_matrix(y, y_pred, labels=labels)
            tn, fp, fn, tp = cm_array.ravel()
            cm = pd.DataFrame(cm_array, columns=labels, index=labels)
        else:
            cm_array = confusion_matrix(y, y_pred)
            labels = np.unique(np.concatenate([np.asarray(y), np.asarray(y_pred)]))
            cm = pd.DataFrame(cm_array, index=labels, columns=labels)
            accuracy = np.trace(cm_array) / cm_array.sum()
            sensitivity = recall_score(y, y_pred, average="macro", zero_division=0)
            f1_macro = f1_score(y, y_pred, average="macro", zero_division=0)
            mcc = matthews_corrcoef(y, y_pred)

            total = cm_array.sum()
            specificities = []
            for i in range(cm_array.shape[0]):
                tp = cm_array[i, i]
                fp = cm_array[:, i].sum() - tp
                fn = cm_array[i, :].sum() - tp
                tn = total - tp - fp - fn
                denom = tn + fp
                specificities.append(tn / denom if denom != 0 else 0)
            specificity = float(np.mean(specificities)) if specificities else 0
            return cm, accuracy, sensitivity, specificity, mcc, f1_macro

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

        mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / mcc_denominator if mcc_denominator != 0 else 0

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        return cm, accuracy, sensitivity, specificity, mcc, f1_score

    def plot_density(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cols: int = 7,
        title: str = "Density Histogram of Each Feature",
        out: str = "Feature Density.png",
    ) -> None:
        rows = (len(X.columns) + cols - 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.7))
        fig.suptitle(title, fontsize=16)

        axs_flat = axs.flatten() if hasattr(axs, "flatten") else [axs]
        labels = pd.Series(y).unique()
        for i, feature in enumerate(X.columns):
            ax = axs_flat[i]
            for label_class in labels:
                data_class = X[pd.Series(y) == label_class]
                data_class[feature].plot(
                    kind="density", ax=ax, title=feature, label=f"Class {label_class}"
                )
            ax.set_xlabel("")
            ax.set_ylabel("Density")
            ax.legend()

        total_axes = rows * cols
        if len(X.columns) < total_axes:
            for i in range(len(X.columns), total_axes):
                fig.delaxes(axs_flat[i])
        plt.tight_layout()
        if out:
            plt.savefig(out, dpi=300, bbox_inches="tight")

    def plot_cluster(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cluster: str = "PCA",
        title: str = "Feature Dimensionality Reduction Cluster",
        out: str = "Feature Cluster.png",
    ) -> None:
        if cluster == "PCA":
            reduced_features = PCA(n_components=2).fit_transform(X)
        elif cluster == "TSNE":
            reduced_features = TSNE(n_components=2, random_state=42).fit_transform(X)
        else:
            raise ValueError("cluster must be 'PCA' or 'TSNE'")

        data_with_labels = pd.DataFrame(
            reduced_features, columns=[f"{cluster}1", f"{cluster}2"]
        )
        data_with_labels["label"] = y

        plt.figure(figsize=(5, 5))
        for label, group in data_with_labels.groupby("label"):
            try:
                color = "#FE8011" if float(label) > 0 else "#2279B5"
            except Exception:
                color = None
            plt.scatter(
                group[f"{cluster}1"],
                group[f"{cluster}2"],
                label=f"Cluster {label}",
                alpha=0.7,
                color=color,
            )

        plt.title(title)
        plt.legend()
        plt.xlabel(f"{cluster} Component 1")
        plt.ylabel(f"{cluster} Component 2")

        if out:
            plt.savefig(out, dpi=300, bbox_inches="tight")

    def plot_correlation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        title: str = "Feature Correlations with Label",
        out: str = "Feature Correlations.png",
    ) -> None:
        correlations = X.corrwith(y)

        plt.figure(figsize=(14, 6))
        plt.bar(
            correlations.index,
            correlations,
            color=["#FE8011" if corr > 0 else "#2279B5" for corr in correlations],
        )
        plt.xticks(rotation=45, ha="right")
        plt.title(title)
        plt.ylabel("Correlation")

        if out:
            plt.savefig(out, dpi=300, bbox_inches="tight")

    def plot_lines(
        self,
        results_df: pd.DataFrame,
        title: str = "Performance Metrics vs. Number of Features",
        out: str = "Feature Selection.png",
    ) -> None:
        plt.figure(figsize=(10, 6))
        plt.plot(results_df["Num Features"], results_df["ACC"], label="Accuracy")
        plt.plot(results_df["Num Features"], results_df["SN"], label="Sensitivity")
        plt.plot(results_df["Num Features"], results_df["SP"], label="Specificity")
        plt.plot(results_df["Num Features"], results_df["MCC"], label="MCC")
        plt.plot(results_df["Num Features"], results_df["F1"], label="F1 Score")

        plt.legend(loc="lower right")
        plt.xlabel("Number of Features")
        plt.ylabel("Metrics Value")
        plt.title(title)
        plt.grid(True)
        if out:
            plt.savefig(out, dpi=300, bbox_inches="tight")

    def plot_cm(self, cm, out: str = "Confusion_Matrix.png") -> None:
        if isinstance(cm, pd.DataFrame):
            cm_values = cm.values
            labels = list(cm.columns)
        else:
            cm_values = np.asarray(cm)
            labels = list(range(cm_values.shape[0]))

        num_classes = cm_values.shape[0]
        plt.figure(figsize=(4, 4))
        plt.imshow(
            cm_values,
            interpolation="nearest",
            cmap=LinearSegmentedColormap.from_list(
                "custom_cmap", ["#FFFFFF", "#2486B9", "#005E91"], N=256
            ),
        )
        plt.colorbar()
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")

        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, labels)
        plt.yticks(tick_marks, labels)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(
                    j,
                    i,
                    str(cm_values[i, j]),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=12,
                )
        if out:
            plt.savefig(out, dpi=300, bbox_inches="tight")

    def plot_roc(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model,
        cv: int = 5,
        title: str = "5-Fold Cross-Validated ROC Curve",
        out: str = "ROC Curve.png",
    ) -> None:
        model = deepcopy(model)
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        all_tpr = []
        mean_fpr = np.linspace(0, 1, 50)
        colors = ["b", "g", "r", "c", "m"]

        plt.figure(figsize=(8, 6))
        for i, (train_index, test_index) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train)
            if not hasattr(model, "predict_proba"):
                raise ValueError("Model must implement predict_proba for ROC plotting.")
            y_pred_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

            all_tpr.append(np.interp(mean_fpr, fpr, tpr))
            plt.plot(
                fpr,
                tpr,
                color=colors[i % len(colors)],
                alpha=0.2,
                label=f"Fold {i + 1} (AUC = {round(auc(fpr, tpr), 2)})",
            )

        mean_tpr = np.mean(all_tpr, axis=0)
        std_tpr = np.std(all_tpr, axis=0)
        roc_auc = auc(mean_fpr, mean_tpr)

        plt.plot(mean_fpr, mean_tpr, color="black", label=f"Mean ROC (AUC = {roc_auc:.2f})")
        plt.fill_between(
            mean_fpr,
            mean_tpr - std_tpr,
            mean_tpr + std_tpr,
            color="gray",
            alpha=0.3,
            label="+/- 1 std. dev.",
        )
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True)
        if out:
            plt.savefig(out, dpi=300, bbox_inches="tight")

    def plot_prc(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model,
        cv: int = 5,
        title: str = "5-Fold Cross-Validated PRC Curve",
        out: str = "PRC Curve.png",
    ) -> None:
        model = deepcopy(model)
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        all_recall = []
        mean_precision = np.linspace(0, 1, 50)
        colors = ["b", "g", "r", "c", "m"]

        plt.figure(figsize=(8, 6))
        for i, (train_index, test_index) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train)
            if not hasattr(model, "predict_proba"):
                raise ValueError("Model must implement predict_proba for PRC plotting.")
            y_pred_prob = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)

            all_recall.append(np.interp(mean_precision, precision[::-1], recall[::-1]))
            plt.plot(
                recall,
                precision,
                color=colors[i % len(colors)],
                alpha=0.6,
                label=f"Fold {i + 1} (AUC = {round(average_precision_score(y_test, y_pred_prob), 2)})",
            )

        mean_recall = np.mean(all_recall, axis=0)
        model.fit(X, y)
        avg_precision = average_precision_score(y, model.predict_proba(X)[:, 1])

        plt.plot(
            mean_recall,
            mean_precision,
            color="black",
            label=f"Mean PRC (Avg. Precision = {avg_precision:.2f})",
        )
        plt.plot([0, 1], [1, 0], color="gray", linestyle="--")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True)
        if out:
            plt.savefig(out, dpi=300, bbox_inches="tight")
