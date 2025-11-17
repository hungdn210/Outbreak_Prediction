import pandas as pd
import numpy as np 
from sklearn.base import clone
from joblib import parallel_backend
from sklearn.metrics import (
    accuracy_score, classification_report, fbeta_score, 
    matthews_corrcoef, roc_auc_score, balanced_accuracy_score
)
from sklearn.base import clone

def _has_all_classes(Y):
  return all(np.unique(Y[:, m]).size >= 2 for m in range(Y.shape[1]))

EPS = 1e-6
def _logit(p):
  p = np.clip(p, EPS, 1-EPS)
  return np.log(p) - np.log(1-p)

def _sigmoid(z):
  return 1.0 / (1.0 + np.exp(-z))

def _scan_threshold(Y_proba, Y_true, beta=2):
  best_threshold, best_score = 0.5, -1
  for threshold in np.linspace(0.05, 0.95, 91):
    cur_score = fbeta_score(Y_true, (Y_proba >= threshold).astype(int), beta=beta, zero_division=0)
    if cur_score > best_score:
      best_threshold, best_score = threshold, cur_score
  return best_threshold, best_score

def _combine_logit_weighted(proba_list_for_one_horizon, weights):
  ws = np.array(weights, dtype=float)
  ws = ws / (ws.sum() + EPS)

  logits_list = []

  for proba in proba_list_for_one_horizon:
    cur_proba = np.clip(proba[:, 1], EPS, 1-EPS)
    logits_list.append(_logit(cur_proba))
  
  Z = np.stack(logits_list, axis = 1)
  z = Z @ ws
  final_proba = _sigmoid(z)
  return np.c_[1 - final_proba, final_proba]

def _get_weight_and_thresholds_search(models_proba, Y_true, beta=2, grid_list=(0.0, 0.25, 0.5, 0.75, 1.0), iters=2, debug=False):
  # Create default values for each weight for each model
  num_models = len(models_proba)
  w = np.ones(num_models) / num_models  # all weights for all models are the same now (sum will be 1)
  
  # Create a default best_thresholds and best_score with weight of each model is the same 
  combined_results = _combine_logit_weighted(models_proba, w)[:, 1]
  best_thresholds, _ = _scan_threshold(Y_proba=combined_results, Y_true=Y_true, beta=beta)
  best_score = fbeta_score(Y_true, (combined_results >= best_thresholds).astype(int), beta=beta, zero_division=0)

  for _ in range(iters):
    improved = False
    for m in range(num_models):
      cur_best_results = (w[m], best_thresholds, best_score)
      for cur_grid in grid_list:
        w_try = w.copy()
        w_try[m] = cur_grid
        w_try = w_try / (w_try.sum() + EPS)
        proba_try = _combine_logit_weighted(models_proba, w_try)[:, 1]
        thresholds_try, _ = _scan_threshold(proba_try, Y_true, beta=beta)
        f2_try = fbeta_score(Y_true, (proba_try >= thresholds_try).astype(int), beta=beta, zero_division=0)
        if f2_try > cur_best_results[2]: # it is current best_score
          cur_best_results = (cur_grid, thresholds_try, f2_try)
      if cur_best_results[2] > best_score:
        w[m], best_thresholds, best_score = cur_best_results[0], cur_best_results[1], cur_best_results[2]
        improved = True
    if not improved:
      break
  
  return w / (w.sum() + EPS), best_thresholds

  

def Get_new_train_test_data(X_train, Y_train, new_test_size):
  new_X_train, new_Y_train, new_X_test, new_Y_test = [], [], [], []
  number_of_train_data = int(len(X_train) * (1 - new_test_size))

  for i in range(len(X_train)):
    if i < number_of_train_data:
      new_X_train.append(X_train[i])
      new_Y_train.append(Y_train[i])
    else:
      new_X_test.append(X_train[i])
      new_Y_test.append(Y_train[i])
  
  return np.array(new_X_train), np.array(new_Y_train), np.array(new_X_test), np.array(new_Y_test)

def Get_learned_thresholds(cur_model, X_train, Y_train, X_test, Y_test, debug=False):
  model = clone(cur_model)
  with parallel_backend('threading', n_jobs=-1):
    model.fit(X_train, Y_train)
  
  Y_proba = model.predict_proba(X_test)
  learned_thresholds = []
  for i in range(Y_test.shape[1]):
    #print(i)
    Y_true_i_list = Y_test[:, i]
    Y_proba_i_list = Y_proba[i][:, 1]
    
    if debug and i == 0:
      print("Y_true length for each month: ", len(Y_true_i_list))
      print("Y_prob length for each month: ", len(Y_proba_i_list))
    best_threshold = 0.5
    best_score = -1
    for cur_threshold in np.linspace(0.05, 0.95, 91):
      Y_pred = (Y_proba_i_list >= cur_threshold).astype(int)
      cur_score = fbeta_score(Y_true_i_list, Y_pred, beta=2, zero_division=0)
      if cur_score >= best_score:
        best_score = cur_score
        best_threshold = cur_threshold
    if debug: 
      print(f"Month +{i+1}: Learned optimal threshold = {best_threshold:.2f}")

    learned_thresholds.append(best_threshold)
    
  return learned_thresholds

def Run_single_model(model_name, model, X_train, Y_train, X_test, Y_test, val_size, thresholds, debug=True):
  if debug:
    print(f"=== Start running single model {model_name} ===")
  if thresholds == 'auto':
    temp_X_train, temp_Y_train, temp_X_test, temp_Y_test  =  Get_new_train_test_data(X_train=X_train, Y_train=Y_train, new_test_size=val_size)
    learned_thresholds = Get_learned_thresholds(cur_model=model, X_train=temp_X_train, Y_train=temp_Y_train, X_test=temp_X_test, Y_test=temp_Y_test, debug=debug)
    if debug:
      print(f"Learned thresholds for model {model_name}: ", learned_thresholds)
  elif isinstance(thresholds, float):
    learned_thresholds = [thresholds] * Y_test.shape[1]
  else:
    raise ValueError("Thresholds must be either 'auto' or float type!")
  
  if not _has_all_classes(Y_train):
    if debug: print(f"[mean] Skipping {model_name}: single-class horizon(s) in TRAIN slice.")
    return

  # Running the model
  with parallel_backend('threading', n_jobs=-1):
    model.fit(X_train, Y_train)
  
  Y_proba = model.predict_proba(X_test)

  Y_pred_list = []
  for k in range(Y_test.shape[1]):
    Y_proba_k = Y_proba[k][:, 1]
    Y_pred = (Y_proba_k >= learned_thresholds[k]).astype(int)
    Y_pred_list.append(Y_pred)
    if debug: 
      Y_true_k = Y_test[:, k]
      print(f"Length of Y_proba and Y_true for month {k + 1} (should be the same): ", len(Y_proba_k), len(Y_true_k))
  
  Y_pred_list = np.array(Y_pred_list).T 
  if debug: 
    print("Final Y_pred_list: ", Y_pred_list)

  return Y_proba, Y_pred_list, learned_thresholds

def Run_Ensemble_mean_weight(models_list, main_X_train, main_Y_train, main_X_test, main_Y_test, val_size, thresholds, num_best_models,enforce_fixed_threshold, equal_weights, debug=False):
  # Choose number of top models for ensemble
  model_scores = {}
  model_proba_list = {}
  cur_X_train, cur_Y_train, cur_X_val, cur_Y_val = Get_new_train_test_data(X_train=main_X_train, Y_train=main_Y_train, new_test_size=val_size)

  for model_name, model in models_list.items():
    # Get learned thresholds for each model
    if not _has_all_classes(cur_Y_train):
      if debug:
        print(f"[rank] Skipping {model_name}: single-class horizon(s) in TRAIN split.")
      continue
    # Get thresholds for each model
    if thresholds == 'auto':
      cur_thresholds = Get_learned_thresholds(cur_model=model, X_train=cur_X_train, Y_train=cur_Y_train, X_test=cur_X_val, Y_test=cur_Y_val, debug=debug)
      if debug:
        print(f"Learned thresholds for model {model_name}: ", cur_thresholds)
    elif isinstance(thresholds, float):
      cur_thresholds = [thresholds] * cur_Y_val.shape[1]
    else:
      raise ValueError("Thresholds must be either 'auto' or float type!")

    with parallel_backend('threading', n_jobs=-1):
      model.fit(cur_X_train, cur_Y_train)
    
    cur_Y_proba = model.predict_proba(cur_X_val)
    model_proba_list[model_name] = cur_Y_proba

    cur_f2_score_list = []
    for k in range(cur_Y_val.shape[1]):
      cur_Y_true_k = cur_Y_val[:, k]
      cur_Y_proba_k = cur_Y_proba[k][:, 1]
      cur_Y_pred_k = (cur_Y_proba_k >= cur_thresholds[k]).astype(int)
      f2_score_k = fbeta_score(cur_Y_true_k, cur_Y_pred_k, beta=2, zero_division=0)
      cur_f2_score_list.append(f2_score_k)
    
    model_scores[model_name] = float(np.mean(cur_f2_score_list))

    if debug:
      print(f"{model_name} mean F2 = {model_scores[model_name]:.3f}")
    
  # Rank models by score (highest first) and take the top-k
  ranked_models = sorted(model_scores, key=model_scores.get, reverse=True)
  best_model_names = ranked_models[:num_best_models]

  # Run all top models in validation
  proba_val_list = [model_proba_list[name] for name in best_model_names]

  if debug:
    print("\nTop models:", best_model_names)
    for name in best_model_names:
      pv = model_proba_list[name]                 # list length K
      shapes = [arr.shape for arr in pv]          # each (N_val, 2)
      print(f"  - {name}: per-horizon proba shapes = {shapes}")
  
  # Get per-horizon weight and threshold learning on validation data
  learned_weights = []
  learned_thresholds = []
  for k in range(cur_Y_val.shape[1]):
    proba_models_val_k = [cur_model_proba[k] for cur_model_proba in proba_val_list]
    Y_true_k = cur_Y_val[:, k]
    weights_k, thresholds_k = _get_weight_and_thresholds_search(models_proba=proba_models_val_k, Y_true=Y_true_k, debug=debug)

    if equal_weights:
      weights_k = np.ones(len(proba_models_val_k), dtype=float)
      weights_k /= weights_k.sum()
    if enforce_fixed_threshold and isinstance(thresholds, float):
      thresholds_k = float(thresholds)
    
    learned_weights.append(weights_k)
    learned_thresholds.append(thresholds_k)
    if debug and k == 0:
      print("Learned weight and its thresholds: ", weights_k, thresholds_k)

  # Refit each model on FULL training, get TEST probabilities
  proba_test_all = []
  for model_name, model in models_list.items():
    if model_name in best_model_names:
      cur_model = clone(model)
      with parallel_backend('threading', n_jobs=-1):
        cur_model.fit(main_X_train, main_Y_train)
      
      cur_Y_proba = cur_model.predict_proba(main_X_test)
      proba_test_all.append(cur_Y_proba)

  # Combine on TEST with learned weights per horizon, apply thresholds
  Y_pred_list = []
  Y_proba_ensemble_test = []
  for k in range(main_Y_test.shape[1]):
    proba_models_test_k = [model_proba[k] for model_proba in proba_test_all]
    combined_k = _combine_logit_weighted(proba_models_test_k, learned_weights[k])
    Y_proba_ensemble_test.append(combined_k)
    pred_k = (combined_k[:, 1] >= learned_thresholds[k]).astype(int)
    Y_pred_list.append(pred_k)
  Y_pred = np.stack(Y_pred_list, axis=1) # (N_test, K)

  return Y_pred, Y_proba_ensemble_test, learned_thresholds






    


  