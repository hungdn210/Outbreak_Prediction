import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import fbeta_score
from joblib import parallel_backend
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.multioutput import MultiOutputClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, fbeta_score, 
    matthews_corrcoef, roc_auc_score, balanced_accuracy_score
)
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import itertools
from sklearn.linear_model import LogisticRegression

# --- CONSTANT --- 
L = 6   # Look back window
K = 6   # Predict the next K months 
DATA_LAG = 6 # The amount of data during DATA_LAG months before month T are stored in each data point T
NUM_OF_K_FOLD = 6
DATA_START_DATE = pd.to_datetime('2005-01-01')
DATA_END_DATE = pd.to_datetime('2024-12-31')
TEST_START_DATE = pd.to_datetime('2022-01-01')
NUMBER_OF_BEST_MODELS_FOR_ENSEMBLE = 5
DEBUG_LENGTH = 3
VAL_SIZE = 0.15

INPUT_DIR_LIST = [
  'dataset_by_country/France/France_level_4/France_level_4_final.csv'
] 
OUTPUT_DIR = 'results_log_models_5.csv'

MODELS_LIST = {
    "k-Nearest Neighbors": MultiOutputClassifier(
        make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=7, weights='distance')
        )
    ),
    "SVC (RBF Kernel)": MultiOutputClassifier(
        make_pipeline(
            StandardScaler(),
            SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
        )
    ),
    "XGBoost": MultiOutputClassifier(
        XGBClassifier(
            scale_pos_weight=20,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1,
            reg_lambda=2,
            eval_metric="logloss",
            random_state=42
        )
    ),
    "Logistic Regression": MultiOutputClassifier(
        make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=10000,
                solver='saga' 
            )
        )
    ),
    # 🔹 New models below
    "Gradient Boosting": MultiOutputClassifier(
        GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42
        )
    ),
    "AdaBoost": MultiOutputClassifier(
        AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.8,
            random_state=42
        )
    ),
    "Hist Gradient Boosting": MultiOutputClassifier(
        HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=6,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=42
        )
    ),
    "CatBoost": MultiOutputClassifier(
        CatBoostClassifier(
            iterations=200,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            eval_metric="F1",  
            auto_class_weights="Balanced", 
            verbose=0,
            random_state=42
        )
    )
}

# --- END CONSTANT ---