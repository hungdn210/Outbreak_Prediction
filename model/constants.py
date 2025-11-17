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
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, HistGradientBoostingClassifier
)
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.multioutput import MultiOutputClassifier


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
  'dataset_by_country/France/France_level_2/France_level_2_final.csv',
  'dataset_by_country/Italy/Italy_level_3/Italy_level_3_final.csv',
  'dataset_by_country/Greece/Greece_level_3/Greece_level_3_final.csv'
] 
OUTPUT_DIR = 'results_log_models.csv'

MODELS_LIST = {
    # --- KNN (best: n=5, distance, p=1)
    "k-Nearest Neighbors": MultiOutputClassifier(
        make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5, weights='distance', p=1)
        )
    ),

    # --- SVC RBF (best: C=0.5, gamma='scale')
    "SVC (RBF Kernel)": MultiOutputClassifier(
        make_pipeline(
            StandardScaler(),
            SVC(kernel='rbf', C=0.5, gamma='scale',
                probability=True, class_weight='balanced', random_state=42)
        )
    ),

    # --- Logistic Regression (best: L1, C=0.1, saga)
    "Logistic Regression": MultiOutputClassifier(
        make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l1", C=0.1, solver='saga',
                class_weight="balanced", max_iter=20000
            )
        )
    ),

    # --- Linear SVM (Calibrated) (best: C=0.25)
    "Linear SVM (Calibrated)": MultiOutputClassifier(
        CalibratedClassifierCV(
            Pipeline([
                ("scaler", StandardScaler()),
                ("svc", LinearSVC(C=0.25, class_weight="balanced",
                                  random_state=42, max_iter=40000))
            ]),
            method="sigmoid", cv=3
        )
    ),

    # --- RidgeClassifier (Calibrated) (best: alpha=0.1)
    "RidgeClassifier (Calibrated)": MultiOutputClassifier(
        CalibratedClassifierCV(
            Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", RidgeClassifier(alpha=0.1))
            ]),
            method="sigmoid", cv=3
        )
    ),

    # --- Random Forest (best grid)
    "Random Forest": MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=1,
            max_features="sqrt", n_jobs=-1, class_weight="balanced",
            random_state=42
        )
    ),

    # --- Balanced Random Forest (best grid)
    "Balanced Random Forest": MultiOutputClassifier(
        BalancedRandomForestClassifier(
            n_estimators=400, max_depth=None, replacement=False,
            n_jobs=-1, random_state=42
        )
    ),

    # --- ExtraTrees (best grid)
    "ExtraTrees": MultiOutputClassifier(
        ExtraTreesClassifier(
            n_estimators=600, max_depth=None, min_samples_split=2,
            n_jobs=-1, random_state=42
        )
    ),

    # --- Gradient Boosting (best: 150, 0.03, depth=3, subsample=0.8)
    "Gradient Boosting": MultiOutputClassifier(
        GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.03, max_depth=3,
            subsample=0.8, random_state=42
        )
    ),

    # --- AdaBoost (best: 100, 0.5)
    "AdaBoost": MultiOutputClassifier(
        AdaBoostClassifier(
            n_estimators=100, learning_rate=0.5, random_state=42
        )
    ),

    # --- HistGradientBoosting (best: max_depth=None, max_iter=200, l2=1.0)
    "Hist Gradient Boosting": MultiOutputClassifier(
        HistGradientBoostingClassifier(
            max_depth=None, max_iter=200, l2_regularization=1.0,
            early_stopping=True, random_state=42
        )
    ),

    # --- XGBoost (best: 200, 0.03, depth=4, subsample=0.7, colsample=0.7, reg_alpha=0, reg_lambda=1, spw=5)
    "XGBoost": MultiOutputClassifier(
        XGBClassifier(
            n_estimators=200, learning_rate=0.03, max_depth=4,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=0, reg_lambda=1,
            scale_pos_weight=5,
            objective="binary:logistic", eval_metric="logloss",
            n_jobs=-1, random_state=42
        )
    ),

    # --- LightGBM (best: 300, 0.03, leaves=31, min_child_samples=5, subsample=0.8, colsample=0.8, reg_lambda=0.0)
    "LightGBM": MultiOutputClassifier(
        LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=31,
            min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=0.0, objective="binary", class_weight="balanced",
            random_state=42
        )
    ),

    # --- CatBoost (unchanged here; if you keep it, set a writable train_dir on Windows)
    "CatBoost": MultiOutputClassifier(
        CatBoostClassifier(
            iterations=200, learning_rate=0.05, depth=6,
            loss_function="Logloss", auto_class_weights="Balanced",
            verbose=0, random_state=42,
            train_dir=".\\catboost_info"  # <- avoids the Windows write error
        )
    ),
}
# --- END CONSTANT ---