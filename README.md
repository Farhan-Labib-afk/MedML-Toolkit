# AIpmt: AI Precision Medicine Toolkit

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AIpmt is a Python toolkit for **predictive modeling, feature selection, and visualization** in precision medicine and related fields.  
It is designed for **students, researchers, and practitioners** who want to quickly test classification models, analyze features, and visualize performance with minimal setup.

---

## 👨‍💻 Author

**Farhan Labib**  
- Undergraduate Computer Science Researcher  
- Passionate about Machine Learning, Data Science, and Precision Medicine  
- [LinkedIn]([https://www.linkedin.com/](https://www.linkedin.com/in/farhan-labib-edu/)) • [GitHub](https://github.com/Farhan-Labib-afk/)  

---

## 🚀 Features

- **Machine Learning Models**  
  - Logistic Regression, Support Vector Machines (SVM)  
  - Easy to extend with other scikit-learn models  

- **Feature Selection**  
  - ANOVA F-test, Chi-Squared  
  - Incremental Feature Selection (IFS) with cross-validation  

- **Evaluation Metrics**  
  - Accuracy, Sensitivity, Specificity, F1-score, MCC  

- **Visualizations**  
  - ROC & PRC curves with cross-validation  
  - Feature correlation plots  
  - PCA / t-SNE clustering  
  - Confusion matrices  
  - Performance vs. number of features  

- **Cross-Validation**  
  - Built-in stratified k-fold CV  

- **Clean API**  
  - Easy to integrate into research workflows  

---

## 📦 Installation

### Prerequisites
- Python **3.8+**
- `pip` package manager

### Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/AIpmt.git
cd AIpmt

# Create virtual environment
python -m venv aipmt_env

# Activate it
# Windows
aipmt_env\Scripts\activate
# macOS/Linux
source aipmt_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
🔬 Quick Start
Example Usage
python
Copy code
import pandas as pd
from sklearn.datasets import load_breast_cancer
from aipmt import AIpmt

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Initialize AIpmt
clf = AIpmt(X, y)

# Train with feature selection
clf.fit(fs_method="ANOVA", ifs_method="LR", ifs_grid=True, ifs_cv=5)

# Visualizations
clf.plot_correlation(X, y)
clf.plot_cluster(X, y, cluster="PCA")
clf.plot_lines(clf.ifs_results)
📊 Visualization Examples
ROC Curves: Cross-validated ROC with confidence intervals

Precision-Recall Curves: Useful for imbalanced data

Feature Correlations: Bar plots of correlations with target

Clustering: PCA/t-SNE scatter plots

Performance Tracking: Metrics vs. feature count

Confusion Matrix: Heatmaps for classification results

🛠️ API Reference
Core Methods
AIpmt(X, y): Initialize with dataset

fit(fs_method, ifs_method, ifs_grid, ifs_cv): Train model + feature selection

transform(X, evaluate): Select best features

train(X, y, method, grid, cv): Train final model

cv_test(X, y, model, cv): Evaluate with cross-validation

Visualization
plot_correlation()

plot_cluster()

plot_density()

plot_lines()

plot_roc()

plot_prc()

plot_cm()

📈 Example Results
yaml
Copy code
Best Features: 12
Performance Metrics:
   Accuracy   : 0.865
   Sensitivity: 0.848
   Specificity: 0.881
   MCC        : 0.73
   F1-Score   : 0.862
🤝 Contributing
Contributions are welcome!

Fork the repo

Create a branch: git checkout -b feature-name

Commit: git commit -m "Add new feature"

Push: git push origin feature-name

Submit a Pull Request

Future Enhancements:

More ML models (XGBoost, NN)

Extra feature selection algorithms

Interactive plotting (Plotly)

Auto hyperparameter optimization

Regression support

📝 License
Licensed under the MIT License.

🎓 Educational Use
AIpmt is great for:

Machine Learning courses

Research projects

Medical AI prototyping

Data science competitions


🙏 Acknowledgments
Developed as part of undergraduate CS research

Built on scikit-learn, pandas, matplotlib

