"""
MSME Failure Predictor - Model Trainer
Trains the Stacking GBM model and saves it to model/models/
"""

import sys
import os

# Resolve repo root so this script can be run from anywhere
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import logging

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, StackingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report
)
from imblearn.over_sampling import SMOTE

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(_REPO_ROOT, 'model', 'model_training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
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

DATA_PATH = os.path.join(_REPO_ROOT, 'data', 'data.csv')
MODEL_DIR = os.path.join(_REPO_ROOT, 'model', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'stacking_gbm.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'robust_scaler.pkl')
METADATA_PATH = os.path.join(MODEL_DIR, 'model_metadata.pkl')

# ============================================================================
# MODEL TRAINER CLASS
# ============================================================================

class MSMEModelTrainer:
    """Trains and saves the Stacking GBM model"""

    def __init__(self, data_path=DATA_PATH):
        self.data_path = data_path
        self.model = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.metrics = {}

    def load_data(self):
        """Load and prepare data"""
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)

        logger.info(f"Dataset shape: {df.shape}")

        feature_cols = [FEATURE_MAP[f] for f in FEATURE_MAP.keys()]

        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns: {missing_cols}")
            raise ValueError(f"Missing columns in dataset: {missing_cols}")

        X = df[feature_cols].copy()
        X.columns = list(FEATURE_MAP.keys())
        y = df['Bankrupt?'].copy()

        logger.info(f"\nClass Distribution:")
        logger.info(f"  Safe (0): {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
        logger.info(f"  Bankrupt (1): {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")
        logger.info(f"  Imbalance Ratio: 1:{int((y==0).sum()/(y==1).sum())}")

        return X, y

    def split_and_scale(self, X, y):
        """Split data and apply scaling"""
        logger.info("\nSplitting data (80/20 train/test)")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")

        logger.info("\nApplying RobustScaler normalization")
        self.scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns
        )

        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test

    def apply_smote(self):
        """Apply SMOTE to training data"""
        logger.info("\nApplying SMOTE for class imbalance")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(self.X_train, self.y_train)

        logger.info(f"After SMOTE:")
        logger.info(f"  Class 0: {(y_train_balanced==0).sum()}")
        logger.info(f"  Class 1: {(y_train_balanced==1).sum()}")

        self.X_train = pd.DataFrame(X_train_balanced, columns=self.X_train.columns)
        self.y_train = pd.Series(y_train_balanced)

    def train_model(self):
        """Train Stacking GBM model"""
        logger.info("\nBuilding Stacking GBM model...")

        base_learners = [
            ('rf', RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            )),
            ('et', ExtraTreesClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )),
            ('dt', DecisionTreeClassifier(
                max_depth=6,
                class_weight='balanced',
                random_state=42
            )),
        ]

        meta_learner = GradientBoostingClassifier(
            n_estimators=50,
            random_state=42
        )

        self.model = StackingClassifier(
            estimators=base_learners,
            final_estimator=meta_learner,
            cv=5,
            passthrough=True,
            n_jobs=-1
        )

        logger.info("Training model (this may take several minutes)...")
        self.model.fit(self.X_train, self.y_train)
        logger.info("✓ Model training complete")

    def evaluate_model(self):
        """Evaluate model on test set"""
        logger.info("\nEvaluating model on test set...")

        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]

        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        auc_roc = roc_auc_score(self.y_test, y_pred_proba)

        self.metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': auc_roc
        }

        logger.info(f"\nTest Set Metrics:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  F1 Score:  {f1:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  ROC-AUC:   {auc_roc:.4f}")
        logger.info(f"\nClassification Report:")
        logger.info("\n" + classification_report(self.y_test, y_pred))

    def save_model(self):
        """Save model and scaler to model/models/"""
        logger.info(f"\nSaving model artifacts to {MODEL_DIR}...")

        os.makedirs(MODEL_DIR, exist_ok=True)

        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"✓ Model saved to {MODEL_PATH}")

        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"✓ Scaler saved to {SCALER_PATH}")

        metadata = {
            'feature_map': FEATURE_MAP,
            'feature_names': list(FEATURE_MAP.keys()),
            'metrics': self.metrics,
            'training_date': datetime.now().isoformat(),
            'training_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'model_type': 'StackingClassifier',
            'base_learners': ['RandomForest', 'GradientBoosting', 'ExtraTrees', 'DecisionTree'],
            'meta_learner': 'GradientBoosting'
        }

        with open(METADATA_PATH, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"✓ Metadata saved to {METADATA_PATH}")

    def train_and_save(self):
        """Complete training pipeline"""
        logger.info("="*80)
        logger.info("MSME FAILURE PREDICTOR - MODEL TRAINING")
        logger.info("="*80)

        X, y = self.load_data()
        self.split_and_scale(X, y)
        self.apply_smote()
        self.train_model()
        self.evaluate_model()
        self.save_model()

        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE ✓")
        logger.info("="*80)
        logger.info(f"\nModel files saved in '{MODEL_DIR}':")
        logger.info(f"  - {MODEL_PATH}")
        logger.info(f"  - {SCALER_PATH}")
        logger.info(f"  - {METADATA_PATH}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data file not found: {DATA_PATH}")
        logger.info("Please ensure 'data.csv' is in the 'data/' directory")
        exit(1)

    trainer = MSMEModelTrainer()
    trainer.train_and_save()
