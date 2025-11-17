# tune_models.py
import pandas as pd
import itertools, json
import numpy as np
from copy import deepcopy
from sklearn.base import clone
from joblib import parallel_backend
from sklearn.metrics import fbeta_score

import constants, data_processing, evaluation_logging
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, HistGradientBoostingClassifier
)
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.multioutput import MultiOutputClassifier

# ---------------------------
# 1) Model builders (consistent with your proba + MultiOutput needs)
# ---------------------------
def build_knn(p): 
    return MultiOutputClassifier(make_pipeline(StandardScaler(), KNeighborsClassifier(**p)))

def build_svc_rbf(p):
    return MultiOutputClassifier(make_pipeline(StandardScaler(), SVC(probability=True, class_weight='balanced', random_state=42, **p)))

def build_logreg(p):
    base = LogisticRegression(class_weight='balanced', max_iter=20000, solver='saga', **p)
    return MultiOutputClassifier(make_pipeline(StandardScaler(), base))

def build_linear_svm_calib(p):
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", LinearSVC(class_weight="balanced", random_state=42, max_iter=40000, **p))
    ])
    # Calibrate to get predict_proba
    return MultiOutputClassifier(CalibratedClassifierCV(base, method="sigmoid", cv=3))

def build_ridge_calib(p):
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeClassifier(**p))
    ])
    return MultiOutputClassifier(CalibratedClassifierCV(base, method="sigmoid", cv=3))

def build_rf(p):
    return MultiOutputClassifier(RandomForestClassifier(n_jobs=-1, random_state=42, class_weight="balanced", **p))

def build_brf(p):
    return MultiOutputClassifier(BalancedRandomForestClassifier(n_jobs=-1, random_state=42, **p))

def build_et(p):
    return MultiOutputClassifier(ExtraTreesClassifier(n_jobs=-1, random_state=42, **p))

def build_gbdt(p):
    return MultiOutputClassifier(GradientBoostingClassifier(random_state=42, **p))

def build_adaboost(p):
    return MultiOutputClassifier(AdaBoostClassifier(random_state=42, **p))

def build_hgbdt(p):
    return MultiOutputClassifier(HistGradientBoostingClassifier(random_state=42, **p))

def build_xgb(p):
    base = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", n_jobs=-1, random_state=42, **p
    )
    return MultiOutputClassifier(base)

def build_lgbm(p):
    base = LGBMClassifier(objective="binary", class_weight="balanced", random_state=42, **p)
    return MultiOutputClassifier(base)

def build_cat(p):
    base = CatBoostClassifier(
        loss_function="Logloss", auto_class_weights="Balanced", verbose=0, random_state=42, **p
    )
    return MultiOutputClassifier(base)

# ---------------------------
# 2) Search space (tight but meaningful)
#    Keep grids modest; you can widen later if needed.
# ---------------------------
SEARCH = {
    "k-Nearest Neighbors": {
        "builder": build_knn,
        "grid": {
            "n_neighbors": [5, 7, 11],
            "weights": ["distance"],
            "p": [1, 2],                 # Manhattan vs Euclidean
        },
    },
    "SVC (RBF Kernel)": {
        "builder": build_svc_rbf,
        "grid": {
            "C": [0.5, 1, 2],
            "gamma": ["scale", 0.1, 0.01],
        },
    },
    "Logistic Regression": {
        "builder": build_logreg,
        "grid": {
            "penalty": ["l1", "l2"],
            "C": [0.1, 1, 10],
        },
    },
    "Linear SVM (Calibrated)": {
        "builder": build_linear_svm_calib,
        "grid": {
            "C": [0.25, 1, 4],
        },
    },
    "RidgeClassifier (Calibrated)": {
        "builder": build_ridge_calib,
        "grid": {
            "alpha": [0.1, 1.0, 10.0],
        },
    },
    "Random Forest": {
        "builder": build_rf,
        "grid": {
            "n_estimators": [400, 800],
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", "log2"],
        },
    },
    "Balanced Random Forest": {
        "builder": build_brf,
        "grid": {
            "n_estimators": [400, 800],
            "max_depth": [None, 10, 20],
            "replacement": [False, True],
        },
    },
    "ExtraTrees": {
        "builder": build_et,
        "grid": {
            "n_estimators": [600, 1000],
            "max_depth": [None, 12, 24],
            "min_samples_split": [2, 5],
        },
    },
    "Gradient Boosting": {
        "builder": build_gbdt,
        "grid": {
            "n_estimators": [150, 300],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [3, 5],
            "subsample": [0.8, 1.0],
        },
    },
    "AdaBoost": {
        "builder": build_adaboost,
        "grid": {
            "n_estimators": [100, 300],
            "learning_rate": [0.5, 0.8, 1.0],
        },
    },
    "Hist Gradient Boosting": {
        "builder": build_hgbdt,
        "grid": {
            "max_depth": [None, 6, 10],
            "max_iter": [200, 400],
            "l2_regularization": [0.0, 1.0],
        },
    },
    "XGBoost": {
        "builder": build_xgb,
        "grid": {
            "n_estimators": [200, 400],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [4, 6],
            "subsample": [0.7, 0.9],
            "colsample_bytree": [0.7, 0.9],
            "reg_alpha": [0, 1],
            "reg_lambda": [1, 2],
            "scale_pos_weight": [5, 10, 20],  # tune class imbalance
        },
    },
    "LightGBM": {
        "builder": build_lgbm,
        "grid": {
            "n_estimators": [300, 600],
            "learning_rate": [0.03, 0.05],
            "num_leaves": [31, 63, 127],
            "min_child_samples": [5, 20],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_lambda": [0.0, 0.5, 1.0],
        },
    },
    "CatBoost": {
        "builder": build_cat,
        "grid": {
            "iterations": [300, 600],
            "depth": [4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.1],
        },
    },
}

# ---------------------------
# 3) Helpers: rolling last-block validation with embargo + threshold learning
# ---------------------------
def time_aware_train_val(
    X_tr, Y_tr, start_times_tr,
    val_fraction=constants.VAL_SIZE,
    embargo_months=constants.L
):
    """
    Time-based last-block validation that doesn't use DatetimeIndex.quantile.
    Compatible with old pandas on Win/Py3.7.
    """
    import pandas as pd
    st = pd.to_datetime(start_times_tr)              # robust to object dtype
    n  = len(st)
    val_fraction = float(max(0.01, min(0.5, val_fraction)))
    n_val = max(1, int(round(val_fraction * n)))

    # sort by time once
    order = np.argsort(st.values)                    # indices that sort ascending
    # indices of validation block = last n_val by time
    val_idx_sorted = order[-n_val:]
    # earliest timestamp inside validation block (i.e., block start)
    cutoff_val_start_ts = pd.Timestamp(st.values[val_idx_sorted[0]])

    # embargo: training must end before (cutoff - L months)
    last_train_time_ts = cutoff_val_start_ts - pd.DateOffset(months=embargo_months)

    # boolean masks against the original order
    val_mask   = st >= cutoff_val_start_ts
    train_mask = st <  last_train_time_ts

    # fallbacks for tiny data
    if train_mask.sum() == 0:
        train_mask = st < cutoff_val_start_ts
    if train_mask.sum() == 0 or val_mask.sum() == 0:
        split = int(max(1, round(0.85 * n)))
        return X_tr[:split], Y_tr[:split], X_tr[split:], Y_tr[split:]

    return X_tr[train_mask], Y_tr[train_mask], X_tr[val_mask], Y_tr[val_mask]

def scan_threshold_best_f2(y_true, proba_positives, beta=2.0):
    best_t, best = 0.5, -1
    for t in np.linspace(0.05, 0.95, 91):
        pred = (proba_positives >= t).astype(int)
        score = fbeta_score(y_true, pred, beta=beta, zero_division=0)
        if score > best:
            best, best_t = score, t
    return best_t, best

def fit_eval(model, X_tr, Y_tr, X_val, Y_val):
    # Skip if any horizon has single-class in training
    if not all(np.unique(Y_tr[:, k]).size >= 2 for k in range(Y_tr.shape[1])):
        return None, -np.inf, None

    with parallel_backend('threading', n_jobs=-1):
        model.fit(X_tr, Y_tr)
    Yp = model.predict_proba(X_val)  # list of K arrays (N,2)

    # threshold per horizon to maximize F2 on validation
    f2s = []
    ts = []
    for k in range(Y_val.shape[1]):
        yk = Y_val[:, k]
        pk = Yp[k][:, 1]
        if np.unique(yk).size < 2:
            # degenerate horizon: use 0.5; F2 computed safely
            t = 0.5
        else:
            t, _ = scan_threshold_best_f2(yk, pk, beta=2.0)
        ts.append(t)
        predk = (pk >= t).astype(int)
        from sklearn.metrics import fbeta_score
        f2s.append(fbeta_score(yk, predk, beta=2.0, zero_division=0))
    return Yp, float(np.mean(f2s)), ts

# ---------------------------
# 4) Main loop: for each dataset, for each model, sweep grid → pick best params
# ---------------------------
def run_param_search():
    for csv_file in constants.INPUT_DIR_LIST:
        # 1) Load and window
        df = data_processing.Get_data(csv_file, constants.DATA_START_DATE, constants.DATA_END_DATE, debug=False)
        X, Y, start_times = data_processing.Generate_data_by_location(df, L=constants.L, K=constants.K, debug=False)
        start_times = pd.to_datetime(start_times)
        X_train, Y_train, X_test, Y_test = data_processing.Split_train_test_data(
            X, Y, start_times, constants.TEST_START_DATE, debug=False
        )

        # For the inner split, we need times corresponding to TRAIN windows only
        train_mask = (start_times < constants.TEST_START_DATE - pd.DateOffset(months=constants.L))
        start_train_times = start_times[train_mask]

        # 2) Search per model
        for model_name, spec in SEARCH.items():
            builder, grid = spec["builder"], spec["grid"]

            best_mean_f2 = -np.inf
            best_params = None
            best_thresholds = None

            # time-aware train/val split once per dataset (keeps comparable)
            X_tr, Y_tr, X_val, Y_val = time_aware_train_val(
                X_train, Y_train, start_train_times,
                val_fraction=constants.VAL_SIZE, embargo_months=constants.L
            )

            # guard: if val set empty (tiny dataset), fall back to simple last-block split
            if len(X_val) == 0:
                split = int(max(1, 0.85 * len(X_train)))
                X_tr, Y_tr, X_val, Y_val = X_train[:split], Y_train[:split], X_train[split:], Y_train[split:]

            # iterate grid
            keys = list(grid.keys())
            for values in itertools.product(*[grid[k] for k in keys]):
                params = dict(zip(keys, values))
                model = builder(params)
                _, mean_f2, thresholds = fit_eval(model, X_tr, Y_tr, X_val, Y_val)
                if mean_f2 > best_mean_f2:
                    best_mean_f2 = mean_f2
                    best_params = deepcopy(params)
                    best_thresholds = deepcopy(thresholds)

            # 3) Refit on FULL TRAIN with best params, evaluate on TEST, log
            model = builder(best_params or {})
            with parallel_backend('threading', n_jobs=-1):
                model.fit(X_train, Y_train)
            Y_proba_test = model.predict_proba(X_test)

            # If we didn’t get thresholds (e.g., degenerate val), compute on train last-block again
            if best_thresholds is None:
                # recompute thresholds on a quick last-block from train
                _, _, thresholds = fit_eval(model, X_tr, Y_tr, X_val, Y_val)
                best_thresholds = thresholds or [0.5]*Y_test.shape[1]

            # Produce Y_pred on TEST using learned thresholds
            Y_pred_test = np.zeros_like(Y_test)
            for k in range(Y_test.shape[1]):
                pk = Y_proba_test[k][:, 1]
                tk = best_thresholds[k] if best_thresholds else 0.5
                Y_pred_test[:, k] = (pk >= tk).astype(int)

            # Metrics + CSV logging (your existing logger)
            metrics = evaluation_logging.Get_metric_results(Y_proba_test, Y_pred_test, Y_test, debug=False)
            evaluation_logging.Log_results_to_csv(
                model_name=f"{model_name} (tuned:{json.dumps(best_params)})",
                metrics=metrics,
                learned_thresholds=best_thresholds,
                K=constants.K,
                input_dir=csv_file,
                output_dir=constants.OUTPUT_DIR,
                Y_test=Y_test,
                Y_pred=Y_pred_test,
                debug=False
            )
            print(f"[{csv_file}] {model_name} → mean F2(val)={best_mean_f2:.3f} params={best_params}")

if __name__ == "__main__":
    import pandas as pd
    run_param_search()
