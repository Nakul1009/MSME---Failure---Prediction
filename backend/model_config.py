"""
MSME Failure Predictor - Model Configuration
Handles model loading, prediction, and feature management
"""

import pickle
import os
import logging
from typing import Dict, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================

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

FEATURE_CATEGORIES = {
    'Liquidity': ['Cash flow rate', 'Current Ratio', 'Quick Ratio', 'Cash/Current Liability'],
    'Solvency': ['Debt ratio %', 'Liability to Equity', 'Interest Coverage Ratio', 'DFL'],
    'Profitability': ['ROA', 'Operating Gross Margin', 'Gross Profit to Sales', 'Net Income to Total Assets'],
    'Efficiency': ['Accounts Receivable Turnover', 'Inventory Turnover Rate', 'Average Collection Days'],
    'Growth': ['Revenue Growth Rate', 'Cash Flow to Sales', 'Cash Flow to Liability'],
}

HEALTHY_RANGES = {
    'Cash flow rate': (0.1, 1.0, 'Higher is better'),
    'Current Ratio': (1.5, 3.0, 'Higher is better'),
    'Quick Ratio': (1.0, 2.0, 'Higher is better'),
    'Cash/Current Liability': (0.2, 1.0, 'Higher is better'),
    'Debt ratio %': (0, 60, 'Lower is better'),
    'Liability to Equity': (0, 2.0, 'Lower is better'),
    'Interest Coverage Ratio': (2.5, float('inf'), 'Higher is better'),
    'DFL': (0, 3.0, 'Lower is better'),
    'ROA': (0.05, 1.0, 'Higher is better'),
    'Operating Gross Margin': (0.15, 1.0, 'Higher is better'),
    'Gross Profit to Sales': (0.15, 1.0, 'Higher is better'),
    'Net Income to Total Assets': (0.05, 1.0, 'Higher is better'),
    'Revenue Growth Rate': (0.0, float('inf'), 'Higher is better'),
    'Accounts Receivable Turnover': (1, float('inf'), 'Higher is better'),
    'Inventory Turnover Rate': (1, float('inf'), 'Higher is better'),
    'Average Collection Days': (0, 60, 'Lower is better'),
}

# ============================================================================
# MODEL PATHS  (relative to repo root, not to backend/)
# ============================================================================

# Resolve to repo root regardless of where the script is run from
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_REPO_ROOT, 'model', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'stacking_gbm.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'robust_scaler.pkl')
METADATA_PATH = os.path.join(MODEL_DIR, 'model_metadata.pkl')

# ============================================================================
# MODEL LOADER CLASS
# ============================================================================

class ModelLoader:
    """Loads and manages the pickled model"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.model = None
        self.scaler = None
        self.metadata = None
        self.feature_names = list(FEATURE_MAP.keys())
        self.is_loaded = False

        self._load_model()
        self._initialized = True

    def _load_model(self):
        """Load model and scaler from pickle files"""
        try:
            if not os.path.exists(MODEL_PATH):
                logger.warning(f"Model not found at {MODEL_PATH}")
                logger.info("Please run model/train.py to train and save the model")
                self.is_loaded = False
                return

            # Load model
            with open(MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"✓ Model loaded from {MODEL_PATH}")

            # Load scaler
            if os.path.exists(SCALER_PATH):
                with open(SCALER_PATH, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"✓ Scaler loaded from {SCALER_PATH}")

            # Load metadata
            if os.path.exists(METADATA_PATH):
                with open(METADATA_PATH, 'rb') as f:
                    self.metadata = pickle.load(f)
                logger.info(f"✓ Metadata loaded from {METADATA_PATH}")

            self.is_loaded = True

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.is_loaded = False

    def predict(self, features: Dict[str, float]) -> Tuple[int, float, float]:
        """
        Make prediction on new data

        Args:
            features: Dictionary with feature names and values

        Returns:
            Tuple of (prediction, bankruptcy_risk_score, safe_score)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Please train and save the model first.")

        # Convert to DataFrame
        X = pd.DataFrame([features])
        X = X[self.feature_names]

        # Scale features
        if self.scaler:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=self.feature_names
            )
        else:
            X_scaled = X

        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]

        return prediction, probabilities[1], probabilities[0]

    def get_feature_importance(self) -> Dict:
        """Get feature importance from the model"""
        if not self.is_loaded or not hasattr(self.model, 'feature_importances_'):
            return {}

        importance = self.model.feature_importances_
        return {
            feature: float(imp)
            for feature, imp in zip(self.feature_names, importance)
        }

    def get_metadata(self) -> Dict:
        """Get model metadata"""
        return self.metadata or {}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_features(features: Dict[str, float]) -> Tuple[bool, list]:
    """
    Validate features

    Returns:
        Tuple of (is_valid, errors)
    """
    errors = []
    required_features = list(FEATURE_MAP.keys())

    # Check missing features
    missing = [f for f in required_features if f not in features]
    if missing:
        errors.append(f"Missing features: {', '.join(missing)}")

    # Check types
    for feature, value in features.items():
        if feature not in required_features:
            continue

        if not isinstance(value, (int, float)):
            errors.append(f"{feature}: Must be numeric")
            continue

        if np.isnan(value) or np.isinf(value):
            errors.append(f"{feature}: Cannot be NaN or infinite")

    return len(errors) == 0, errors


def get_feature_health_score(feature: str, value: float) -> float:
    """
    Get health score (0-100) for a feature

    Args:
        feature: Feature name
        value: Feature value

    Returns:
        Health score 0-100
    """
    if feature not in HEALTHY_RANGES:
        return 50.0

    min_val, max_val, direction = HEALTHY_RANGES[feature]

    if max_val == float('inf'):
        # Unbounded features
        if direction == 'Higher is better':
            if value >= min_val:
                return 100.0
            else:
                return (value / min_val) * 100
        else:
            if value <= min_val:
                return 100.0
            else:
                return 100.0 / (value / min_val)
    else:
        # Bounded features
        if direction == 'Higher is better':
            if value < min_val:
                return (value / min_val) * 50
            elif value > max_val:
                return max(0, 100 - ((value - max_val) / max_val * 50))
            else:
                return 100.0
        else:
            if value > max_val:
                return (max_val / value) * 50
            elif value < min_val:
                return 100.0
            else:
                return 100 - ((value - min_val) / (max_val - min_val) * 100)


def get_feature_interpretation(feature: str, value: float) -> str:
    """Get human-readable interpretation of a feature"""
    interpretations = {
        'Cash flow rate': f"Cash generation is {value:.2f}. {'Strong' if value > 0.5 else 'Weak'}.",
        'Current Ratio': f"Current assets are {value:.2f}x current liabilities. {'Good liquidity' if value >= 1.5 else 'Liquidity concern'}.",
        'Quick Ratio': f"Immediate liquidity ratio is {value:.2f}. {'Healthy' if value >= 1.0 else 'May struggle with immediate obligations'}.",
        'Debt ratio %': f"Debt is {value:.1f}% of assets. {'Moderate' if value < 60 else 'High'} leverage.",
        'Interest Coverage Ratio': f"EBIT covers interest {value:.2f}x. {'Comfortable' if value > 2.5 else 'Tight'} debt servicing.",
        'ROA': f"Assets generate {value:.2%} returns. {'Good efficiency' if value > 0.05 else 'Poor'} efficiency.",
        'Revenue Growth Rate': f"Revenue growing at {value:.2%}. {'Positive growth' if value > 0 else 'Declining'}.",
    }

    if feature in interpretations:
        return interpretations[feature]
    return f"Value: {value:.4f}"
