import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import f_classif, chi2
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score


class AIpmt:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.feature_importance = {
            "ANOVA": f_classif,
            "Chi2": chi2
        }
        # Use solver that supports 'l1' and set random_state for reproducibility
        self.train_model_list = {
            "LR": LogisticRegression(solver='liblinear', max_iter=5000, random_state=42),
            "SVM": svm.SVC(kernel='rbf', probability=True, random_state=42)
        }
        self.param_grid = {
            "LR": {'C': [0.001, 0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2']},
            "SVM": {
                'C': list(2 ** i for i in range(-5, 15 + 1, 2)),
                'gamma': list(2 ** i for i in range(-15, 3 + 1, 2))
            }
        }
        return

    def fit(self, fs_method="ANOVA", ifs_method=None, ifs_grid=False, ifs_cv=5):

        scaler = StandardScaler()
        self.X = pd.DataFrame(scaler.fit_transform(self.X), columns=self.X.columns)

        self.correlations = self.X.corrwith(self.y)

        self.fs_scores = self.feature_importance[fs_method](self.X, self.y)[0]

        sorted_indices = self.fs_scores.argsort()
        self.X_sorted = self.X.iloc[:, sorted_indices[::-1]]

        if ifs_method:
            results = []
            for i in tqdm(range(1, len(self.X_sorted.columns) + 1)):
                selected_features = self.X_sorted.iloc[:, :i]
                model = self.train(selected_features, self.y, grid=ifs_grid, method=ifs_method, cv=ifs_cv)
                cv_result = self.cv_test(selected_features, self.y, model, cv=ifs_cv)

                results.append({
                    'Num Features': i,
                    'ACC': cv_result['ACC'],
                    'SN': cv_result['SN'],
                    'SP': cv_result['SP'],
                    'MCC': cv_result['MCC'],
                    'F1': cv_result['F1']
                })
            self.ifs_results = pd.DataFrame(results)

    def transform(self, X, evaluate="F1"):

        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        try:
            best_num = self.ifs_results['Num Features'][self.ifs_results[evaluate].idxmax()]
            return X[self.X_sorted.iloc[:, :best_num].columns]
        except Exception:
            return X[self.X_sorted.columns]

    def train(self, X, y, grid=False, method='LR', cv=5):
        model = deepcopy(self.train_model_list[method])
        # If grid search requested, use GridSearchCV around the base estimator
        if grid:
            bestModel = GridSearchCV(model, self.param_grid[method], cv=cv, scoring="accuracy",
                                     return_train_score=False, n_jobs=1)
            model = bestModel.fit(X, y)
        else:
            model.fit(X, y)
        return model

    def cv_test(self, X, y, model, cv=5):

        y_pred = cross_val_predict(model, X, y, cv=cv)
        cm, ACC, SN, SP, MCC, F1 = self.calculate_metrics(y, y_pred)
        return {'ACC': round(ACC, 3),
                'SN': round(SN, 3),
                'SP': round(SP, 3),
                'MCC': round(MCC, 3),
                'F1': round(F1, 3),
                'cm': cm.values
                }

    def predict(self, X, model, threshold=0.5):
        # Safe predict: if model supports predict_proba use it, otherwise fallback to predict
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X)
            y_pred = (y_pred_proba[:, 1] > threshold).astype(int)
            return y_pred, y_pred_proba
        else:
            y_pred = model.predict(X)
            return y_pred, None

    def calculate_metrics(self, y, y_pred):

        # handle binary and multi-class gracefully
        unique_labels = np.unique(np.concatenate([np.asarray(y), np.asarray(y_pred)]))
        if unique_labels.size == 2:
            # preserve order of labels present in y if possible
            labels_in_y = np.unique(np.asarray(y))
            labels = list(labels_in_y)
            # compute cm with explicit labels
            cm_array = confusion_matrix(y, y_pred, labels=labels)
            tn, fp, fn, tp = cm_array.ravel()
            cm = pd.DataFrame(cm_array, columns=labels, index=labels)
        else:
            cm_array = confusion_matrix(y, y_pred)
            labels = np.unique(np.concatenate([np.asarray(y), np.asarray(y_pred)]))
            cm = pd.DataFrame(cm_array, index=labels, columns=labels)
            # for multiclass, compute simple overall accuracy and leave other metrics as placeholders
            accuracy = np.trace(cm_array) / cm_array.sum()
            # Return placeholders for SN, SP, MCC, F1: compute per-class metrics if needed
            sensitivity = None
            specificity = None
            mcc = None
            f1_score = None
            return cm, accuracy, sensitivity, specificity, mcc, f1_score

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

        mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / mcc_denominator if mcc_denominator != 0 else 0

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        return cm, accuracy, sensitivity, specificity, mcc, f1_score

    def plot_density(self, X, y, cols=7, title="Density Histogram of Each Feature", out="Feature Density.png"):

        rows = (len(X.columns) + cols - 1) // cols

        fig, axs = plt.subplots(rows, cols, figsize=(cols*3, rows*2.7))
        fig.suptitle(title, fontsize=16)
        axs_flat = axs.flatten() if hasattr(axs, "flatten") else [axs]
        for i, feature in enumerate(X.columns):
            ax = axs_flat[i]
            for label_class in pd.Series(y).unique():
                data_class = X[pd.Series(y) == label_class]
                data_class[feature].plot(kind='density', ax=ax, title=feature, label=f"Class {label_class}")
            ax.set_xlabel('')
            ax.set_ylabel('Density')
            ax.legend()

        # remove unused axes
        total_axes = rows * cols
        if len(X.columns) < total_axes:
            for i in range(len(X.columns), total_axes):
                fig.delaxes(axs_flat[i])
        plt.tight_layout()
        if out:
            plt.savefig(out, dpi=300, bbox_inches='tight')

    def plot_cluster(self, X, y, cluster='PCA', title="Feature Dimensionality Reduction Cluster", out="Feature Cluster.png"):
        if cluster == 'PCA':
            pca = PCA(n_components=2)
            reduced_features = pca.fit_transform(X)
        elif cluster == 'TSNE':
            tsne = TSNE(n_components=2, random_state=42)
            reduced_features = tsne.fit_transform(X)
        else:
            raise ValueError("cluster must be 'PCA' or 'TSNE'")

        data_with_labels = pd.DataFrame(reduced_features, columns=[f'{cluster}1', f'{cluster}2'])
        data_with_labels['label'] = y

        plt.figure(figsize=(5, 5))
        for label, group in data_with_labels.groupby('label'):
            # color selection: numeric labels handled, non-numeric fallback to default
            try:
                color = '#FE8011' if float(label) > 0 else '#2279B5'
            except Exception:
                color = None
            plt.scatter(group[f'{cluster}1'], group[f'{cluster}2'], label=f'Cluster {label}', alpha=0.7, color=color)

        plt.title(title)
        plt.legend()
        plt.xlabel(f'{cluster} Component 1')
        plt.ylabel(f'{cluster} Component 2')

        if out:
            plt.savefig(out, dpi=300, bbox_inches='tight')

    def plot_correlation(self, X, y, title="Feature Correlations with Label", out="Feature Correlations.png"):

        correlations = X.corrwith(y)

        plt.figure(figsize=(14, 6))
        plt.bar(correlations.index, correlations, color=['#FE8011' if corr > 0 else '#2279B5' for corr in correlations])
        plt.xticks(rotation=45, ha='right')
        plt.title(title)
        plt.ylabel('Correlation')

        if out:
            plt.savefig(out, dpi=300, bbox_inches='tight')

    def plot_lines(self, results_df, title="Performance Metrics vs. Number of Features", out="Feature Selection.png"):
        plt.figure(figsize=(10, 6))
        plt.plot(results_df['Num Features'], results_df['ACC'], label='Accuracy')
        plt.plot(results_df['Num Features'], results_df['SN'], label='Sensitivity')
        plt.plot(results_df['Num Features'], results_df['SP'], label='Specificity')
        plt.plot(results_df['Num Features'], results_df['MCC'], label='MCC')
        plt.plot(results_df['Num Features'], results_df['F1'], label='F1 Score')

        plt.legend(loc='lower right')
        plt.xlabel('Number of Features')
        plt.ylabel('Metrics Value')
        plt.title(title)
        plt.grid(True)
        if out:
            plt.savefig(out, dpi=300, bbox_inches='tight')

    def plot_cm(self, cm, out='Confusion_Matrix.png'):

        num_classes = cm.shape[0]

        plt.figure(figsize=(4, 4))

        plt.imshow(cm, interpolation='nearest',
                   cmap=LinearSegmentedColormap.from_list('custom_cmap', ['#FFFFFF', '#2486B9', '#005E91'], N=256))

        plt.colorbar()

        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')

        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, range(num_classes))
        plt.yticks(tick_marks, range(num_classes))

        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, str(cm.iloc[i, j]), ha='center', va='center', color='black', fontsize=12)
        if out:
            plt.savefig(out, dpi=300, bbox_inches='tight')

    def plot_roc(self, X, y, model, cv=5, title="5-Fold Cross-Validated ROC Curve", out="ROC Curve.png"):
        model = deepcopy(model)

        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        all_fpr = []
        all_tpr = []
        mean_fpr = np.linspace(0, 1, 50)

        colors = ['b', 'g', 'r', 'c', 'm']

        plt.figure(figsize=(8, 6))
        for i, (train_index, test_index) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train)
            if not hasattr(model, "predict_proba"):
                raise ValueError("Model must implement predict_proba for ROC plotting.")
            y_pred_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

            all_fpr.append(fpr)
            # use numpy interp (scipy.interp deprecated)
            all_tpr.append(np.interp(mean_fpr, fpr, tpr))

            plt.plot(fpr, tpr, color=colors[i % len(colors)], alpha=0.2,
                     label=f'Fold {i+1} (AUC = {round(auc(fpr, tpr), 2)})')

        mean_tpr = np.mean(all_tpr, axis=0)
        std_tpr = np.std(all_tpr, axis=0)

        roc_auc = auc(mean_fpr, mean_tpr)

        plt.plot(mean_fpr, mean_tpr, color='black', label='Mean ROC (AUC = {:.2f})'.format(roc_auc))
        plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='gray', alpha=0.3,
                         label='± 1 std. dev.')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(True)
        if out:
            plt.savefig(out, dpi=300, bbox_inches='tight')

    def plot_prc(self, X, y, model, cv=5, title="5-Fold Cross-Validated PRC Curve", out="PRC Curve.png"):
        model = deepcopy(model)

        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        all_precision = []
        all_recall = []
        mean_precision = np.linspace(0, 1, 50)

        colors = ['b', 'g', 'r', 'c', 'm']

        plt.figure(figsize=(8, 6))
        for i, (train_index, test_index) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train)
            if not hasattr(model, "predict_proba"):
                raise ValueError("Model must implement predict_proba for PRC plotting.")
            y_pred_prob = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)

            all_precision.append(precision)
            # interpolate recall values onto a mean_precision grid for averaging
            all_recall.append(np.interp(mean_precision, precision[::-1], recall[::-1]))

            plt.plot(recall, precision, color=colors[i % len(colors)], alpha=0.6,
                     label=f'Fold {i+1} (AUC = {round(average_precision_score(y_test, y_pred_prob), 2)})')

        mean_recall = np.mean(all_recall, axis=0)
        avg_precision = average_precision_score(y, model.predict_proba(X)[:, 1])

        plt.plot(mean_recall, mean_precision, color='black',
                 label='Mean PRC (Avg. Precision = {:.2f})'.format(avg_precision))
        plt.plot([0, 1], [1, 0], color='gray', linestyle='--')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(True)
        if out:
            plt.savefig(out, dpi=300, bbox_inches='tight')
