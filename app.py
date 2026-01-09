#!/usr/bin/env python3
"""Streamlit UI for MedML Toolkit."""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.utils.multiclass import type_of_target

from medml_toolkit import MedMLToolkit


def load_sample_dataset() -> pd.DataFrame:
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def render_plot(title: str, plot_fn, *args, **kwargs) -> None:
    plot_fn(*args, **kwargs)
    st.subheader(title)
    st.pyplot(plt.gcf())
    plt.close()


def split_xy(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def target_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        pd.DataFrame(
            {
                "column": df.columns,
                "dtype": [str(dt) for dt in df.dtypes],
                "unique": [df[col].nunique(dropna=True) for col in df.columns],
                "missing": [int(df[col].isna().sum()) for col in df.columns],
            }
        )
        .sort_values(by="unique")
        .reset_index(drop=True)
    )
    return summary


def encode_labels(y: pd.Series) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
    if pd.api.types.is_numeric_dtype(y):
        return y, None
    codes, uniques = pd.factorize(y.astype(str))
    mapping = pd.DataFrame({"label": uniques, "code": range(len(uniques))})
    return pd.Series(codes, index=y.index, name=y.name), mapping


def main() -> None:
    st.set_page_config(page_title="MedML Toolkit", page_icon=":bar_chart:", layout="wide")
    st.title("MedML Toolkit: Precision Medicine Modeling")
    st.write(
        "Upload a CSV dataset or use a built-in sample to run feature selection, "
        "train a model, and visualize performance."
    )

    with st.sidebar:
        st.header("Data")
        data_source = st.radio("Data source", ["Sample dataset", "Upload CSV"])

        if data_source == "Upload CSV":
            uploaded = st.file_uploader("Upload CSV", type=["csv"])
            df = read_uploaded_csv(uploaded) if uploaded else None
        else:
            df = load_sample_dataset()

        st.header("Model")
        fs_method = st.selectbox("Feature selection", ["ANOVA", "Chi2"], index=0)
        if fs_method == "Chi2":
            st.caption("Chi2 uses Min-Max scaling to keep features non-negative.")
        ifs_method = st.selectbox("IFS model", ["LR", "SVM"], index=0)
        ifs_cv = st.slider("CV folds", min_value=3, max_value=10, value=5)
        ifs_grid = st.checkbox("Grid search", value=False)
        evaluate = st.selectbox("Select best features by", ["F1", "ACC", "MCC"], index=0)
        cluster_method = st.selectbox("Clustering plot", ["PCA", "TSNE"], index=0)
        st.header("Target Handling")
        enable_binning = st.checkbox("Auto-bin continuous targets", value=False)
        bin_count = st.slider("Bins (quantiles)", min_value=2, max_value=10, value=3)

    if df is None:
        st.info("Upload a CSV file to get started.")
        return

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), width="stretch")
    st.subheader("Target Candidate Summary")
    st.dataframe(target_summary(df), width="stretch")

    st.info(
        "Most columns are continuous features. Choose a label column with a small "
        "number of unique values (e.g., 2-10)."
    )

    show_all_targets = st.checkbox("Show all columns in target list", value=False)
    unique_counts = df.nunique(dropna=True)
    candidate_cols = [
        col for col in df.columns if show_all_targets or unique_counts[col] <= 20
    ]

    default_target = "target" if "target" in df.columns else df.columns[-1]
    if default_target not in candidate_cols:
        candidate_cols = [default_target] + candidate_cols
    default_index = candidate_cols.index(default_target)
    target_col = st.selectbox("Target column", candidate_cols, index=default_index)

    if st.button("Run MedML Toolkit"):
        X, y_raw = split_xy(df, target_col)
        y, mapping = encode_labels(y_raw)
        target_type = type_of_target(y)
        if target_type not in {"binary", "multiclass"}:
            if enable_binning and pd.api.types.is_numeric_dtype(y_raw):
                try:
                    y = pd.qcut(y_raw, q=bin_count, labels=False, duplicates="drop")
                    target_type = type_of_target(y)
                    bins_sorted = sorted(int(v) for v in pd.unique(y))
                    mapping = pd.DataFrame(
                        {"label": [f"bin_{i}" for i in bins_sorted], "code": bins_sorted}
                    )
                except Exception as exc:
                    st.error(f"Auto-binning failed: {exc}")
                    return
            else:
                st.error(
                    "Target column must be categorical for classification. "
                    "Select a label column with discrete classes or enable auto-binning."
                )
                return

        non_numeric = X.select_dtypes(exclude=["number"]).columns.tolist()
        if non_numeric:
            st.error(
                "All feature columns must be numeric. Non-numeric columns: "
                + ", ".join(non_numeric)
            )
            return

        if cluster_method == "TSNE" and X.shape[0] > 2000:
            st.warning("TSNE can be slow on large datasets. Consider PCA for speed.")
        if ifs_grid and X.shape[1] > 50:
            st.warning("Grid search can be slow with many features.")

        with st.spinner("Running feature selection and training..."):
            classifier = MedMLToolkit(X, y)
            try:
                classifier.fit(
                    fs_method=fs_method,
                    ifs_method=ifs_method,
                    ifs_grid=ifs_grid,
                    ifs_cv=ifs_cv,
                    show_progress=False,
                )
            except Exception as exc:
                st.error(f"Fit failed: {exc}")
                return

            best_features = classifier.transform(X, evaluate=evaluate)
            model = classifier.train(best_features, y, grid=ifs_grid, method=ifs_method, cv=ifs_cv)
            results = classifier.cv_test(best_features, y, model, cv=ifs_cv)

        st.subheader("Metrics (Cross-Validated)")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("ACC", results["ACC"])
        col2.metric("SN", results["SN"])
        col3.metric("SP", results["SP"])
        col4.metric("F1", results["F1"])
        col5.metric("MCC", results["MCC"])

        if mapping is not None:
            st.subheader("Label Encoding")
            st.dataframe(mapping, width="content")

        if results["SN"] is None:
            st.info("Multi-class labels detected: SN/SP/MCC/F1 are not computed.")

        st.subheader("Top Features")
        if hasattr(classifier, "X_sorted"):
            st.write(list(classifier.X_sorted.columns[:10]))

        render_plot(
            "Feature Correlations",
            classifier.plot_correlation,
            X,
            y,
            out=None,
        )
        render_plot(
            f"{cluster_method} Cluster",
            classifier.plot_cluster,
            X,
            y,
            cluster=cluster_method,
            out=None,
        )

        if hasattr(classifier, "ifs_results"):
            render_plot(
                "Feature Selection Curve",
                classifier.plot_lines,
                classifier.ifs_results,
                out=None,
            )

        try:
            render_plot("ROC Curve", classifier.plot_roc, best_features, y, model, cv=ifs_cv, out=None)
            render_plot("Precision-Recall Curve", classifier.plot_prc, best_features, y, model, cv=ifs_cv, out=None)
        except Exception as exc:
            st.warning(f"ROC/PRC plots skipped: {exc}")


if __name__ == "__main__":
    main()
