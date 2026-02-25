import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data
df = pd.read_csv("C:\\Users\\nakul\\Downloads\\taiwanese+bankruptcy+prediction\\data.csv") #Give your path here

# Feature mapping (columns have leading spaces in the dataset)
FEATURE_MAP = {
    'Cash flow rate': ' Cash flow rate',
    'Cash Flow to Sales': ' Cash Flow to Sales',
    'Cash Flow to Liability': ' Cash Flow to Liability',
    'Current Ratio': ' Current Ratio',
    'Quick Ratio': ' Quick Ratio',
    'Cash/Current Liability': ' Cash/Current Liability',
    'Debt ratio %': ' Debt ratio %',
    'Liability to Equity': ' Liability to Equity',
    'Interest Coverage Ratio': ' Interest Coverage Ratio (Interest expense to EBIT)',
    'DFL': ' Degree of Financial Leverage (DFL)',
    'ROA': ' ROA(C) before interest and depreciation before interest',
    'Operating Gross Margin': ' Operating Gross Margin',
    'Gross Profit to Sales': ' Gross Profit to Sales',
    'Net Income to Total Assets': ' Net Income to Total Assets',
    'Revenue Growth Rate': ' Total Asset Growth Rate',
    'Accounts Receivable Turnover': ' Accounts Receivable Turnover',
    'Inventory Turnover Rate': ' Inventory Turnover Rate (times)',
    'Average Collection Days': ' Average Collection Days',
}

features = list(FEATURE_MAP.values())
X = df[features]
y = df['Bankrupt?']

# Split data (stratified for imbalance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define base estimators
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('et', ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
    ('dt', DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42)),
]

# Stacking model
stacking_gbm = StackingClassifier(
    estimators=estimators,
    final_estimator=GradientBoostingClassifier(n_estimators=50, random_state=42),
    cv=3,  # Reduced for faster training; increase if needed
    passthrough=True,
    n_jobs=1  # Set to 1 to avoid parallel issues in some environments
)

# Train
stacking_gbm.fit(X_train_scaled, y_train)

# Evaluate
y_pred = stacking_gbm.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and scaler for backend use
joblib.dump(stacking_gbm, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')