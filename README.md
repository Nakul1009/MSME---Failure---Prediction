# MSME Failure Predictor

An AI-powered financial health analysis tool for Micro, Small & Medium Enterprises (MSMEs). Input financial ratios and receive instant bankruptcy risk predictions backed by a Stacking GBM ensemble model, plus actionable improvement advice from Google Gemini AI.

## Project Structure

```
MSME---Failure---Prediction/
├── backend/                   # Flask REST API
│   ├── app.py                 # Main application & endpoints
│   ├── model_config.py        # Model loader, feature config, utilities
│   └── __init__.py
├── frontend/                  # Vanilla HTML/CSS/JS UI
│   ├── index.html
│   ├── script.js
│   └── styles.css
├── model/                     # ML model training
│   ├── train.py               # Training pipeline (run this to retrain)
│   ├── models/                # Saved model artifacts (gitignored)
│   │   ├── stacking_gbm.pkl
│   │   ├── robust_scaler.pkl
│   │   └── model_metadata.pkl
│   └── __init__.py
├── data/
│   └── data.csv               # Training dataset (~11MB)
├── docs/
│   ├── BACKEND_GUIDE.md
│   └── COMPLETE_SYSTEM_SUMMARY.md
├── .env                       # API keys (gitignored)
├── .gitignore
├── requirements.txt
├── run.py                     # Server entry point
└── README.md
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Train the model (first time only)
```bash
python model/train.py
```
This trains the Stacking GBM model and saves artifacts to `model/models/`.

### 4. Start the backend server
```bash
python run.py
```
Server runs at `http://localhost:5000`.

### 5. Open the frontend
Open `frontend/index.html` in your browser (or serve it with a local server).

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/predict` | Single company prediction |
| POST | `/api/batch-predict` | Batch company predictions |
| POST | `/api/feature-insights` | Detailed feature health breakdown |
| GET | `/api/model-info` | Model metadata and metrics |

### Example: Single Prediction
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"Current Ratio": 2.1, "Debt ratio %": 45, ...}}'
```

---

## Model Architecture

- **Base learners**: Random Forest, Gradient Boosting, Extra Trees, Decision Tree
- **Meta-learner**: Gradient Boosting Classifier
- **Ensemble**: `StackingClassifier` with 5-fold cross-validation
- **Imbalance handling**: SMOTE oversampling
- **Feature scaling**: RobustScaler

## Features Used (18 financial ratios)

| Category | Features |
|----------|----------|
| Liquidity | Cash flow rate, Current Ratio, Quick Ratio, Cash/Current Liability |
| Solvency | Debt ratio %, Liability to Equity, Interest Coverage Ratio, DFL |
| Profitability | ROA, Operating Gross Margin, Gross Profit to Sales, Net Income to Total Assets |
| Efficiency | Accounts Receivable Turnover, Inventory Turnover Rate, Average Collection Days |
| Growth | Revenue Growth Rate, Cash Flow to Sales, Cash Flow to Liability |