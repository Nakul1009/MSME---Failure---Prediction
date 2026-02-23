import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Initialize FastAPI
app = FastAPI(title="MSME Failure Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model and Scaler
try:
    model = joblib.load("failure_model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("Model and Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback to local paths if needed
    try:
        model = joblib.load("backend/failure_model.pkl")
        scaler = joblib.load("backend/scaler.pkl")
    except:
        print("Final attempt to load model failed.")

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    chat_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    print("WARNING: GEMINI_API_KEY not found in environment variables.")

# Feature Names (in order expected by model)
FEATURES = [
    'cash_flow_rate', 'cash_flow_to_sales', 'cash_flow_to_liability',
    'current_ratio', 'quick_ratio', 'cash_current_liability',
    'debt_ratio', 'liability_to_equity', 'interest_coverage_ratio',
    'dfl', 'roa', 'operating_gross_margin',
    'gross_profit_to_sales', 'net_income_to_total_assets', 'revenue_growth_rate',
    'accounts_receivable_turnover', 'inventory_turnover_rate', 'average_collection_days'
]

class PredictionRequest(BaseModel):
    cash_flow_rate: float
    cash_flow_to_sales: float
    cash_flow_to_liability: float
    current_ratio: float
    quick_ratio: float
    cash_current_liability: float
    debt_ratio: float
    liability_to_equity: float
    interest_coverage_ratio: float
    dfl: float
    roa: float
    operating_gross_margin: float
    gross_profit_to_sales: float
    net_income_to_total_assets: float
    revenue_growth_rate: float
    accounts_receivable_turnover: float
    inventory_turnover_rate: float
    average_collection_days: float

class ChatRequest(BaseModel):
    prediction_result: dict
    user_data: dict
    message: str

@app.get("/")
async def root():
    return {"message": "MSME Failure Prediction API is running"}

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Prepare input data in correct order
        input_data = [getattr(request, f) for f in FEATURES]
        
        # Scale input
        input_scaled = scaler.transform([input_data])
        
        # Predict
        probability = model.predict_proba(input_scaled)[0][1]
        prediction = int(model.predict(input_scaled)[0])
        
        return {
            "prediction": prediction,
            "probability": float(probability),
            "status": "High Risk" if probability > 0.5 else "Low Risk"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")
    
    try:
        # Construct context for Gemini
        context = f"""
        You are a financial advisor for an MSME (Micro, Small, and Medium Enterprise).
        A business failure prediction model has analyzed their data.
        
        Prediction Result: {request.prediction_result['status']} (Probability of Failure: {request.prediction_result['probability']:.2%})
        
        Key Financial Ratios Provided:
        {request.user_data}
        
        Instructions:
        1. If the risk is high, provide concrete, actionable advice to overcome potential bankruptcy.
        2. Focus on the weakest financial metrics (e.g., if liquidity ratios like Current Ratio are low, explain how to improve cash flow).
        3. Be encouraging but professional and realistic.
        4. The user is asking: "{request.message}"
        """
        
        response = chat_model.generate_content(context)
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
