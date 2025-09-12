# AIpmt: AI Precision Medicine Toolkit

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AIpmt is a comprehensive Python toolkit designed for **predictive modeling and feature analysis** in precision medicine applications. Built with flexibility, visualization, and educational use in mind, it helps students, researchers, and practitioners quickly test classification models, evaluate features, and visualize results with minimal setup.

## 🚀 Features

- **Multiple ML Models**: Logistic Regression, SVM (easily extendable)
- **Advanced Feature Selection**: ANOVA F-test, Chi-squared, incremental feature selection
- **Comprehensive Metrics**: Accuracy, Sensitivity, Specificity, MCC, F1-score
- **Rich Visualizations**: 
  - ROC and Precision-Recall curves with cross-validation
  - Feature correlation analysis
  - PCA/t-SNE clustering plots
  - Confusion matrices
  - Feature selection performance tracking
- **Cross-Validation**: Built-in k-fold cross-validation with stratification
- **Easy Integration**: Clean API designed for research workflows

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Install

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/AIpmt.git
cd AIpmt
```

2. **Create virtual environment:**
```bash
python -m venv aipmt_env

# Windows
aipmt_env\Scripts\activate

# macOS/Linux
source aipmt_env/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
pip install -e .
```

## 🔬 Quick Start

### Basic Usage

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from aipmt import AIpmt

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Initialize AIpmt
classifier = AIpmt(X, y)

# Fit with feature selection
classifier.fit(fs_method="ANOVA", ifs_method="LR", ifs_grid=True, ifs_cv=5)

# Generate visualizations
classifier.plot_correlation(X, y)
classifier.plot_cluster(X, y, cluster='PCA')
classifier.plot_lines(classifier.ifs_results)
```

### Advanced Usage

```python
# Custom feature selection and model training
classifier.fit(
    fs_method="Chi2",           # Feature selection method
    ifs_method="SVM",           # Model for incremental selection
    ifs_grid=True,              # Enable grid search
    ifs_cv=10                   # 10-fold cross-validation
)

# Get best features based on F1 score
best_features = classifier.transform(X, evaluate="F1")

# Train final model
model = classifier.train(best_features, y, grid=True, method='SVM', cv=5)

# Evaluate performance
results = classifier.cv_test(best_features, y, model, cv=5)
print(f"Best F1 Score: {results['F1']}")
```

## 📊 Visualization Gallery

AIpmt provides publication-ready visualizations:

- **ROC Curves**: Cross-validated ROC analysis with confidence intervals
- **Precision-Recall Curves**: Detailed PRC analysis for imbalanced datasets  
- **Feature Correlations**: Bar plots showing feature-target correlations
- **Clustering Plots**: PCA/t-SNE visualizations for data exploration
- **Performance Tracking**: Line plots showing metrics vs. number of features
- **Confusion Matrices**: Customizable confusion matrix heatmaps

## 🛠️ API Reference

### Core Methods

#### `AIpmt(X, y)`
Initialize the AIpmt classifier.
- **X**: Feature matrix (pandas DataFrame)
- **y**: Target variable (pandas Series)

#### `fit(fs_method, ifs_method, ifs_grid, ifs_cv)`
Train the model with feature selection.
- **fs_method**: Feature selection method ('ANOVA', 'Chi2')
- **ifs_method**: Model for incremental selection ('LR', 'SVM')
- **ifs_grid**: Enable hyperparameter tuning
- **ifs_cv**: Cross-validation folds

#### `transform(X, evaluate)`
Transform data using selected features.
- **X**: Input features
- **evaluate**: Metric for feature selection ('F1', 'ACC', 'MCC')

### Visualization Methods

- `plot_correlation()`: Feature correlation analysis
- `plot_cluster()`: PCA/t-SNE clustering
- `plot_density()`: Feature distribution plots
- `plot_lines()`: Performance vs. features
- `plot_roc()`: ROC curve analysis
- `plot_prc()`: Precision-recall curves
- `plot_cm()`: Confusion matrix

## 📈 Example Results

```
Best Performance Results:
Features: 15
Accuracy: 0.956
Sensitivity: 0.947
Specificity: 0.965
F1-Score: 0.954
MCC: 0.912
```

## 🔧 Testing

Run the included test script to verify installation:

```bash
python test_aipmt.py
```

For comprehensive testing with real data:

```bash
python examples/simple_example.py
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Commit changes**: `git commit -m 'Add feature'`
4. **Push to branch**: `git push origin feature-name`
5. **Submit a Pull Request**

### Future Enhancements
- Additional ML models (XGBoost, Neural Networks)
- More feature selection algorithms
- Interactive plotting with Plotly
- Automated hyperparameter optimization
- Support for regression tasks

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎓 Educational Use

AIpmt is designed for educational purposes and research projects. It's perfect for:
- **Machine Learning Courses**: Hands-on feature selection and model evaluation
- **Research Projects**: Quick prototyping and result visualization  
- **Data Science Competitions**: Feature engineering and model comparison
- **Medical AI Applications**: Precision medicine research workflows

## 📞 Support

For questions, issues, or contributions:
- **GitHub Issues**: [Report bugs or request features](https://github.com/yourusername/AIpmt/issues)
- **Discussions**: [Ask questions or share ideas](https://github.com/yourusername/AIpmt/discussions)

## 🙏 Acknowledgments

- Developed as part of undergraduate Computer Science research
- Built on top of scikit-learn, pandas, and matplotlib
- Special thanks to the open-source community

---

**Made with ❤️ for the machine learning community**