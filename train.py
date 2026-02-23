import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Dataset Path
DATA_PATH = r'C:\Users\nakul\Downloads\taiwanese+bankruptcy+prediction\data.csv'

# Feature Mapping (18 features + 1 target)
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

def train_and_evaluate():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # Filter columns
    df = df[COLUMNS_TO_USE]
    
    # Handle missing values if any
    df = df.dropna()
    
    X = df.drop('Bankrupt?', axis=1)
    y = df['Bankrupt?']
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }
    
    best_model = None
    best_f1 = -1
    best_name = ""
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {"Accuracy": acc, "F1-Score": f1}
        print(f"{name} - Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

    print(f"\nBest Model: {best_name} (F1: {best_f1:.4f})")
    
    # Save best model and scaler
    joblib.dump(best_model, 'failure_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model and Scaler saved successfully.")
    
    return results

if __name__ == "__main__":
    train_and_evaluate()
