import os
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    balanced_accuracy_score
)

import constants

def Get_metric_results(Y_proba, Y_pred, Y_test, debug=False):
    metrics = {'f2': [], 'precision': [], 'recall': [], 'mcc': [], 'auc': [], 'bacc': []}
    beta2 = 4.0  # for F2

    for k in range(Y_test.shape[1]):
        y_true = Y_test[:, k]
        y_pred_k = Y_pred[:, k]
        # Y_proba is a list of (N,2) arrays in your codepaths
        y_proba_k = Y_proba[k][:, 1]

        # build 2x2 confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_k, labels=[0, 1]).ravel()

        # Manual precision/recall/F2
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f2   = (1 + beta2) * prec * rec / (beta2 * prec + rec) if (beta2 * prec + rec) > 0 else 0.0

        # MCC (handle 0 denominator)
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else 0.0

        # AUC & Balanced Accuracy only if both classes seen
        if np.unique(y_true).size == 2:
            try:
                auc = roc_auc_score(y_true, y_proba_k)
            except Exception:
                auc = np.nan
            bacc = balanced_accuracy_score(y_true, y_pred_k)
        else:
            auc = np.nan
            bacc = 0.5  # baseline when one class only

        metrics['f2'].append(f2)
        metrics['precision'].append(prec)
        metrics['recall'].append(rec)
        metrics['mcc'].append(mcc)
        metrics['auc'].append(auc)
        metrics['bacc'].append(bacc)

        if debug:
            print(f"\nConfusion Matrix for Month +{k+1} (labels=[0,1]):")
            print(np.array([[tn, fp], [fn, tp]]))
            print(f"Month +{k+1}: F2={f2:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, MCC={mcc:.3f}, AUC={auc}, BAcc={bacc:.3f}")

    return metrics

def Log_results_to_csv(model_name, metrics, learned_thresholds, K, input_dir, output_dir, Y_test, Y_pred, debug=False):
  current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  data_id = os.path.splitext(os.path.basename(input_dir))[0]
  
  # Define columns
  metric_columns = [
      'timestamp', 'data_id', 'model', 'month', 'thresholds', 'start_date', 'end_date', 'data_split_date',
      'F2', 'Precision', 'Recall', 'MCC', 'AUC', 'BalancedAccuracy',
      'TP', 'TN', 'FP', 'FN'
  ]

  metric_rows = []
  for k in range(K):
    tp = tn = fp = fn = 0

    if Y_test is not None and Y_pred is not None:
      tn, fp, fn, tp = confusion_matrix(Y_test[:, k], Y_pred[:, k], labels=[0, 1]).ravel()

    metric_rows.append([
      current_time, data_id, model_name, f"Month+{k+1}",
      f"{learned_thresholds[k]:.3f}" if learned_thresholds else "N/A",
      constants.DATA_START_DATE, constants.DATA_END_DATE, constants.TEST_START_DATE,
      f"{metrics['f2'][k]:.3f}",
      f"{metrics['precision'][k]:.3f}",
      f"{metrics['recall'][k]:.3f}",
      f"{metrics['mcc'][k]:.3f}",
      f"{metrics['auc'][k]:.3f}",
      f"{metrics['bacc'][k]:.3f}",
      tp, tn, fp, fn
    ])
  
  # Write or append to CSV
  if os.path.isfile(output_dir):
    existing_df = pd.read_csv(output_dir)
    new_df = pd.DataFrame(metric_rows, columns=metric_columns)
    full_df = pd.concat([existing_df, new_df], ignore_index=True)
    full_df = full_df.sort_values(by=['data_id', 'timestamp'])
    full_df.to_csv(output_dir, index=False)
  else:
    pd.DataFrame(metric_rows, columns=metric_columns).to_csv(output_dir, index=False)
  

