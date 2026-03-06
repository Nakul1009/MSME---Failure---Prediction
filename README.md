# MSME Failure Predictor

An AI-powered financial health analysis tool for Micro, Small & Medium Enterprises (MSMEs). Input financial ratios and receive instant bankruptcy risk predictions backed by a Stacking GBM ensemble model, plus actionable improvement advice from HuggingFace AI.

![Dashboard Preview](dashboard.png)

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Step-by-Step Setup Guide](#step-by-step-setup-guide)
  - [Step 1: Clone the Repository](#step-1-clone-the-repository)
  - [Step 2: Create a Virtual Environment](#step-2-create-a-virtual-environment)
  - [Step 3: Install Dependencies](#step-3-install-dependencies)
  - [Step 4: Configure Environment Variables](#step-4-configure-environment-variables)
  - [Step 5: Train the ML Model](#step-5-train-the-ml-model)
  - [Step 6: Start the Server](#step-6-start-the-server)
  - [Step 7: Open the Dashboard](#step-7-open-the-dashboard)
- [Project Structure](#project-structure)
- [How to Use the Dashboard](#how-to-use-the-dashboard)
- [API Endpoints](#api-endpoints)
- [Model Architecture](#model-architecture)
- [Features Used](#features-used-18-financial-ratios)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before you begin, make sure you have the following installed on your system:

| Software | Version | Download Link |
|----------|---------|---------------|
| **Python** | 3.10 or higher (tested with 3.12) | [python.org/downloads](https://www.python.org/downloads/) |
| **pip** | Comes with Python | Included with Python |
| **Git** | Any recent version | [git-scm.com/downloads](https://git-scm.com/downloads) |

> **How to check if Python is installed:**
> Open Command Prompt (Windows) or Terminal (Mac/Linux) and type:
> ```bash
> python --version
> ```
> You should see something like `Python 3.12.x`. If not, install Python first.

> **Important (Windows):** During Python installation, make sure to check ✅ **"Add Python to PATH"**.

---

## Step-by-Step Setup Guide

### Step 1: Clone the Repository

Open Command Prompt (Windows) or Terminal (Mac/Linux) and run:

```bash
git clone https://github.com/Nakul1009/MSME---Failure---Prediction.git
```

Then navigate into the project folder:

```bash
cd MSME---Failure---Prediction
```

> **Note:** If you downloaded the project as a ZIP file instead, extract it and open a terminal inside the extracted folder.

---

### Step 2: Create a Virtual Environment

A virtual environment keeps the project's dependencies isolated from your system Python.

**On Windows (Command Prompt):**
```bash
python -m venv env
env\Scripts\activate
```

**On Windows (PowerShell):**
```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

> **If you get a PowerShell execution policy error**, run this first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

**On macOS / Linux:**
```bash
python3 -m venv env
source env/bin/activate
```

After activation, your terminal prompt should show `(env)` at the beginning, like this:
```
(env) C:\Users\YourName\MSME---Failure---Prediction>
```

---

### Step 3: Install Dependencies

With the virtual environment activated, install all required packages:

```bash
pip install -r requirements.txt
```

This installs:
- **Flask** — Web server framework
- **scikit-learn** — Machine learning library
- **pandas & numpy** — Data processing
- **imbalanced-learn** — SMOTE for handling class imbalance
- **huggingface-hub** — AI-powered advisory suggestions
- **python-dotenv** — Environment variable management

> **This may take 2-5 minutes** depending on your internet speed. Wait until you see `Successfully installed ...` at the end.

---

### Step 4: Configure Environment Variables

The project uses a HuggingFace API key for the AI-powered advisory chatbot. **This step is optional** — the app will work without it (using rule-based suggestions instead of AI suggestions).

**To enable AI advisory (optional):**

1. Go to [huggingface.co](https://huggingface.co/) and create a free account
2. Go to **Settings → Access Tokens → New Token** and create a token
3. Create a file called `.env` in the project root folder with this content:

```env
HF_API_KEY=your_huggingface_api_key_here
```

**To skip AI advisory:**

You can skip this step entirely. The app will still work for predictions and use rule-based suggestions instead.

---

### Step 5: Train the ML Model

> ⚠️ **This step is mandatory on first run.** The trained model files are not included in the repository (they are too large for Git).

With your virtual environment activated, run:

```bash
python model/train.py
```

**What this does:**
1. Loads the financial dataset from `data/data.csv` (~6,800 companies)
2. Preprocesses and denormalizes the data to real-world scale
3. Applies SMOTE oversampling to handle class imbalance (only ~3% of companies are bankrupt)
4. Trains a Stacking GBM ensemble model (Random Forest + Gradient Boosting + Extra Trees + Decision Tree)
5. Calibrates the optimal prediction threshold
6. Saves the trained model to `model/models/`

**Expected output (takes 1-2 minutes):**
```
================================================================================
MSME FAILURE PREDICTOR - MODEL TRAINING
================================================================================
Loading data from .../data/data.csv
Dataset shape: (6819, 96)
Clipping outliers at p1/p99...
Denormalizing to real-world scale...
Splitting data (80/20 train/test)
Applying SMOTE for class imbalance
Training model (this may take several minutes)...
✓ Model training complete
Evaluating model on test set...

Test Set Metrics:
  Accuracy:  0.95+
  ROC-AUC:   0.86+
  Recall:    0.50

================================================================================
TRAINING COMPLETE ✓
================================================================================
```

> **If you see errors**, make sure you're using the virtual environment (`(env)` in your prompt) and all dependencies from Step 3 are installed.

---

### Step 6: Start the Server

With your virtual environment activated, run:

```bash
python run.py
```

**Expected output:**
```
============================================================
  MSME FAILURE PREDICTOR — Backend Server
============================================================
  Server    : http://0.0.0.0:5000
  Frontend  : http://localhost:5000/
  Health    : http://localhost:5000/health
============================================================
 * Serving Flask app 'backend.app'
 * Running on http://127.0.0.1:5000
```

> **Keep this terminal window open!** The server must be running while you use the app. To stop the server, press `Ctrl + C`.

> **If port 5000 is already in use**, close any other terminal windows that might be running the server, or restart your computer.

---

### Step 7: Open the Dashboard

Open your web browser and navigate to:

```
http://localhost:5000/
```

You should see the **MSME Failure Predictor** dashboard with the financial data entry form.

🎉 **The application is now running!**

---

## How to Use the Dashboard

### Making a Prediction

1. **Enter Financial Data:** Fill in the 18 financial ratios in the form. Each field has a tooltip explaining what it means.

2. **Click "Analyze Risk":** The model will predict the bankruptcy risk and display:
   - **Risk Score** (0-100%) — probability of financial failure
   - **Risk Level** — HIGH / MEDIUM / LOW
   - **Status** — AT RISK / STABLE
   - **AI Suggestions** — actionable improvement advice

### Example: High-Risk Company
| Metric | Value |
|--------|-------|
| Current Ratio | 0.3 |
| Debt ratio % | 95 |
| ROA | -0.15 |
| Cash flow rate | -0.2 |

→ Expected result: **HIGH RISK (~90%+ bankruptcy risk)**

### Example: Healthy Company
| Metric | Value |
|--------|-------|
| Current Ratio | 3.0 |
| Debt ratio % | 25 |
| ROA | 0.15 |
| Cash flow rate | 0.5 |

→ Expected result: **LOW RISK (~1% bankruptcy risk)**

### AI Chat Advisor

After making a prediction, switch to the **Chat** tab to ask follow-up questions like:
- "Why is my risk score high?"
- "How can I improve my liquidity?"
- "What should I do about my debt ratio?"

---

## Project Structure

```
MSME---Failure---Prediction/
├── backend/                   # Flask REST API
│   ├── app.py                 # Main application & endpoints
│   ├── model_config.py        # Model loader, feature config, utilities
│   └── __init__.py
├── frontend/                  # Vanilla HTML/CSS/JS UI
│   ├── index.html             # Dashboard page
│   ├── script.js              # Frontend logic & API calls
│   └── styles.css             # Styling & animations
├── model/                     # ML model training
│   ├── train.py               # Training pipeline (run this to retrain)
│   ├── models/                # Saved model artifacts (generated after training)
│   │   ├── stacking_gbm.pkl   # Trained model
│   │   ├── robust_scaler.pkl  # Feature scaler
│   │   └── model_metadata.pkl # Training metrics & config
│   └── __init__.py
├── data/
│   └── data.csv               # Training dataset (~6,800 companies, ~11MB)
├── docs/
│   ├── BACKEND_GUIDE.md       # Backend API documentation
│   └── COMPLETE_SYSTEM_SUMMARY.md
├── .env                       # API keys (create manually, not in Git)
├── .gitignore                 # Files excluded from Git
├── requirements.txt           # Python dependencies
├── run.py                     # Server entry point
└── README.md                  # This file
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check — verify server and model status |
| POST | `/api/predict` | Single company bankruptcy prediction |
| POST | `/api/batch-predict` | Batch predictions for multiple companies |
| POST | `/api/feature-insights` | Detailed feature health breakdown |
| GET | `/api/model-info` | Model metadata, metrics, and configuration |
| POST | `/api/chat` | Conversational AI financial advisor |

### Example API Call (Single Prediction)

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"Cash flow rate": 0.5, "Cash Flow to Sales": 0.3, "Cash Flow to Liability": 0.8, "Current Ratio": 3.0, "Quick Ratio": 2.5, "Cash/Current Liability": 0.8, "Debt ratio %": 25, "Liability to Equity": 0.5, "Interest Coverage Ratio": 10.0, "DFL": 1.2, "ROA": 0.15, "Operating Gross Margin": 0.4, "Gross Profit to Sales": 0.35, "Net Income to Total Assets": 0.12, "Revenue Growth Rate": 0.15, "Accounts Receivable Turnover": 8.0, "Inventory Turnover Rate": 6.0, "Average Collection Days": 30}}'
```

---

## Model Architecture

| Component | Details |
|-----------|---------|
| **Type** | Stacking Ensemble Classifier |
| **Base Learners** | Random Forest, Gradient Boosting, Extra Trees, Decision Tree |
| **Meta-Learner** | Gradient Boosting Classifier |
| **Cross-Validation** | 5-fold Stratified K-Fold |
| **Imbalance Handling** | SMOTE oversampling (80% ratio) |
| **Feature Scaling** | RobustScaler |
| **Threshold** | Calibrated via Precision-Recall curve |
| **Training Data** | ~6,800 companies (97% safe, 3% bankrupt) |

### Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | ~95% |
| ROC-AUC | ~0.87 |
| Recall (Bankruptcy) | ~50% |
| Precision (Bankruptcy) | ~35% |

---

## Features Used (18 Financial Ratios)

| Category | Features |
|----------|----------|
| **Liquidity** | Cash flow rate, Current Ratio, Quick Ratio, Cash/Current Liability |
| **Solvency** | Debt ratio %, Liability to Equity, Interest Coverage Ratio, DFL |
| **Profitability** | ROA, Operating Gross Margin, Gross Profit to Sales, Net Income to Total Assets |
| **Efficiency** | Accounts Receivable Turnover, Inventory Turnover Rate, Average Collection Days |
| **Growth** | Revenue Growth Rate, Cash Flow to Sales, Cash Flow to Liability |

---

## Troubleshooting

### ❌ "python is not recognized as an internal command"
→ Python is not installed or not added to PATH. Reinstall Python and check ✅ "Add Python to PATH".

### ❌ "ModuleNotFoundError: No module named 'flask'"
→ You're not inside the virtual environment. Run:
```bash
# Windows
env\Scripts\activate

# Mac/Linux
source env/bin/activate
```
Then try again.

### ❌ "Model not loaded" / API Error 503
→ The model hasn't been trained yet. Run:
```bash
python model/train.py
```
Then restart the server.

### ❌ Port 5000 is already in use
→ Another process is using port 5000. Either:
- Close any other terminal windows running the server
- Restart your computer
- Or change the port in `run.py` (line 32: change `port=5000` to `port=5001`)

### ❌ PowerShell execution policy error
→ Run this command first:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### ❌ AI advisory shows "fallback" instead of AI responses
→ Make sure you have a valid HuggingFace API key in `.env`. This is optional — the app still works with rule-based suggestions.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Backend | Python, Flask, Flask-CORS |
| ML | scikit-learn, imbalanced-learn, pandas, numpy |
| AI Advisory | HuggingFace Inference API (Qwen model) |
| Data | Taiwan Economic Journal bankruptcy dataset |

---

## License

This project is built for academic/educational purposes.