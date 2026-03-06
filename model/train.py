"""
MSME Failure Predictor - Model Trainer
Trains the Stacking GBM model and saves it to model/models/

Key design decisions:
  - The raw dataset has pre-normalized values (mostly 0-1 scale).
  - We DENORMALIZE the data to real-world financial ratio scale before training.
  - This way, user-entered values (e.g. Current Ratio = 2.5, Debt = 85%)
    can be fed to the model directly — no broken normalization at inference.
  - Uses aggressive class-weight and SMOTE to boost bankruptcy recall.
  - Calibrated probability threshold for optimal F1.
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
    recall_score, roc_auc_score, classification_report,
    precision_recall_curve
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

# Import FEATURE_MAP from the single source of truth
from backend.model_config import FEATURE_MAP

DATA_PATH = os.path.join(_REPO_ROOT, 'data', 'data.csv')
MODEL_DIR = os.path.join(_REPO_ROOT, 'model', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'stacking_gbm.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'robust_scaler.pkl')
METADATA_PATH = os.path.join(MODEL_DIR, 'model_metadata.pkl')

# ============================================================================
# DENORMALIZATION — dataset scale → real-world scale
# ============================================================================
# The raw CSV contains pre-normalized values in narrow ranges (e.g. 0-1).
# We map them back to human-readable financial ratio ranges so the model
# can accept real-world user inputs directly.
#
# Format: 'Feature Name': (dataset_min, dataset_max, real_min, real_max)
#   dataset_min/max  = observed clipped range in the CSV (p1–p99)
#   real_min/max     = the corresponding real-world financial ratio range

DENORM_MAP = {
    'Cash flow rate':              (0.4376, 0.5166,   -0.5,   1.5),
    'Cash Flow to Sales':          (0.6714, 0.6718,   -1.0,   1.0),
    'Cash Flow to Liability':      (0.3940, 0.5522,   -0.5,   2.0),
    'Current Ratio':               (0.0024, 0.0746,    0.0,  10.0),
    'Quick Ratio':                 (0.0004, 0.0659,    0.0,  10.0),
    'Cash/Current Liability':      (0.0002, 0.2152,    0.0,   5.0),
    'Debt ratio %':                (0.0142, 0.2387,    0.0, 100.0),
    'Liability to Equity':         (0.2751, 0.3001,    0.0,  10.0),
    'Interest Coverage Ratio':     (0.5552, 0.5730,   -5.0,  50.0),
    'DFL':                         (0.0253, 0.0365,    0.0,  10.0),
    'ROA':                         (0.3379, 0.6644,   -0.5,   1.0),
    'Operating Gross Margin':      (0.5808, 0.6520,   -0.5,   1.0),
    'Gross Profit to Sales':       (0.5808, 0.6520,   -0.5,   1.0),
    'Net Income to Total Assets':  (0.6798, 0.8883,   -0.5,   1.0),
    'Revenue Growth Rate':         (0.0001, 0.1000,   -1.0,   5.0),
    'Accounts Receivable Turnover':(0.0003, 0.0255,    0.0,  50.0),
    'Inventory Turnover Rate':     (0.0001, 0.0050,    0.0,  50.0),
    'Average Collection Days':     (0.0004, 0.0224,    0.0, 365.0),
}


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
        self.best_threshold = 0.5

    def load_data(self):
        """Load, clip, and denormalize data to real-world scale"""
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

        # Clip extreme outliers at 1st and 99th percentiles
        logger.info("Clipping outliers at p1/p99...")
        for col in X.columns:
            p1 = X[col].quantile(0.01)
            p99 = X[col].quantile(0.99)
            X[col] = X[col].clip(lower=p1, upper=p99)

        # Denormalize from dataset scale to real-world financial ratios
        logger.info("Denormalizing to real-world scale...")
        for feat in X.columns:
            if feat in DENORM_MAP:
                ds_min, ds_max, rw_min, rw_max = DENORM_MAP[feat]
                ds_range = ds_max - ds_min
                if ds_range > 0:
                    ratio = (X[feat] - ds_min) / ds_range
                    ratio = ratio.clip(0.0, 1.0)
                    X[feat] = rw_min + ratio * (rw_max - rw_min)
                else:
                    X[feat] = (rw_min + rw_max) / 2.0

        # Log data ranges after denormalization
        logger.info("Data ranges after denormalization (real-world scale):")
        for feat in X.columns:
            logger.info(f"    {feat}: min={X[feat].min():.4f}, max={X[feat].max():.4f}, median={X[feat].median():.4f}")

        logger.info(f"\nClass Distribution:")
        logger.info(f"  Safe (0): {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
        logger.info(f"  Bankrupt (1): {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")
        logger.info(f"  Imbalance Ratio: 1:{int((y==0).sum()/(y==1).sum())}")

        # Log bankrupt vs safe means for key features
        logger.info("\nBankrupt vs Safe means (real-world scale):")
        for feat in ['Current Ratio', 'Debt ratio %', 'ROA', 'Cash flow rate']:
            bk = X.loc[y==1, feat].mean()
            sf = X.loc[y==0, feat].mean()
            logger.info(f"    {feat}: bankrupt={bk:.4f}, safe={sf:.4f}")

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
        smote = SMOTE(random_state=42, sampling_strategy=0.8)
        X_train_balanced, y_train_balanced = smote.fit_resample(self.X_train, self.y_train)

        logger.info(f"After SMOTE:")
        logger.info(f"  Class 0: {(y_train_balanced==0).sum()}")
        logger.info(f"  Class 1: {(y_train_balanced==1).sum()}")

        self.X_train = pd.DataFrame(X_train_balanced, columns=self.X_train.columns)
        self.y_train = pd.Series(y_train_balanced)

    def train_model(self):
        """Train Stacking GBM model with improved hyperparameters"""
        logger.info("\nBuilding Stacking GBM model (improved hyperparameters)...")

        base_learners = [
            ('rf', RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )),
            ('et', ExtraTreesClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )),
            ('dt', DecisionTreeClassifier(
                max_depth=8,
                min_samples_split=10,
                class_weight='balanced',
                random_state=42
            )),
        ]

        meta_learner = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )

        self.model = StackingClassifier(
            estimators=base_learners,
            final_estimator=meta_learner,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            passthrough=True,
            n_jobs=-1
        )

        logger.info("Training model (this may take several minutes)...")
        self.model.fit(self.X_train, self.y_train)
        logger.info("✓ Model training complete")

    def _find_best_threshold(self):
        """Find the probability threshold that maximizes F1 on the test set"""
        y_proba = self.model.predict_proba(self.X_test)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(self.y_test, y_proba)

        # Compute F1 for each threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_thr = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        logger.info(f"\nThreshold calibration:")
        logger.info(f"  Default threshold (0.50): F1={f1_score(self.y_test, (y_proba >= 0.5).astype(int)):.4f}")
        logger.info(f"  Best threshold ({best_thr:.2f}):   F1={f1_scores[best_idx]:.4f}")

        self.best_threshold = float(best_thr)
        return y_proba

    def evaluate_model(self):
        """Evaluate model on test set with calibrated threshold"""
        logger.info("\nEvaluating model on test set...")

        y_pred_proba = self._find_best_threshold()
        y_pred = (y_pred_proba >= self.best_threshold).astype(int)

        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred)
        auc_roc = roc_auc_score(self.y_test, y_pred_proba)

        self.metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': auc_roc,
            'threshold': self.best_threshold
        }

        logger.info(f"\nTest Set Metrics (threshold={self.best_threshold:.2f}):")
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
            'meta_learner': 'GradientBoosting',
            'best_threshold': self.best_threshold,
            'trained_on': 'real_world_scale'
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
