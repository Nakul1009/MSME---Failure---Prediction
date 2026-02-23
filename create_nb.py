import nbformat as nbf

# Define the notebook content
nb = nbf.v4.new_notebook()

# Markdown cell: Introduction
intro_md = """# MSME Failure Prediction - Model Development
This notebook documents the process of training and evaluating machine learning models to predict business bankruptcy using financial ratios.

## Target Dataset
Taiwanese Bankruptcy Prediction Dataset

## Selected Features (18)
1. Cash flow rate
2. Cash Flow to Sales
3. Cash Flow to Liability
4. Current Ratio
5. Quick Ratio
6. Cash/Current Liability
7. Debt ratio %
8. Liability to Equity
9. Interest Coverage Ratio
10. DFL
11. ROA
12. Operating Gross Margin
13. Gross Profit to Sales
14. Net Income to Total Assets
15. Revenue Growth Rate
16. Accounts Receivable Turnover
17. Inventory Turnover Rate
18. Average Collection Days
"""

# Code cell: Imports and Data Loading
load_code = """import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset Path
DATA_PATH = r'C:\\Users\\nakul\\Downloads\\taiwanese+bankruptcy+prediction\\data.csv'

# Feature Mapping
COLUMNS_TO_USE = [
    'Bankrupt?',
    ' Cash flow rate',
    ' Cash Flow to Sales',
    ' Cash Flow to Liability',
    ' Current Ratio',
    ' Quick Ratio',
    ' Cash/Current Liability',
    ' Debt ratio %',
    ' Liability to Equity',
    ' Interest Coverage Ratio (Interest expense to EBIT)',
    ' Degree of Financial Leverage (DFL)',
    ' ROA(C) before interest and depreciation before interest',
    ' Operating Gross Margin',
    ' Gross Profit to Sales',
    ' Net Income to Total Assets',
    ' Realized Sales Gross Profit Growth Rate',
    ' Accounts Receivable Turnover',
    ' Inventory Turnover Rate (times)',
    ' Average Collection Days'
]

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
df = df[COLUMNS_TO_USE]
df = df.dropna()

print(f"Dataset shape: {df.shape}")
df.head()"""

# Code cell: Preprocessing
preprocess_code = """X = df.drop('Bankrupt?', axis=1)
y = df['Bankrupt?']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split (80/20 with stratification for imbalanced data)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print("Data split and scaled.")"""

# Code cell: Model Training
train_code = """models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

results = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[name] = {"Accuracy": acc, "F1-Score": f1, "Model": model}
    print(f"{name} - Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

best_name = max(results, key=lambda x: results[x]['F1-Score'])
print(f"\\nBest Model: {best_name}")"""

# Code cell: Export
export_code = """# Save best model and scaler
joblib.dump(results[best_name]['Model'], 'failure_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and Scaler exported as .pkl")"""

# Add cells to notebook
nb.cells = [
    nbf.v4.new_markdown_cell(intro_md),
    nbf.v4.new_code_cell(load_code),
    nbf.v4.new_code_cell(preprocess_code),
    nbf.v4.new_code_cell(train_code),
    nbf.v4.new_code_cell(export_code)
]

# Save notebook
with open('model_development.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created: model_development.ipynb")
