import pandas as pd 
import os
import constants_2 as constants
import data_processing
import models
import evaluation_logging

def Run():
  for csv_file in constants.INPUT_DIR_LIST:
    # Get the dataset from csv file
    df = data_processing.Get_data(input_csv=csv_file, start_date=constants.DATA_START_DATE, end_date=constants.DATA_END_DATE, debug=False)

    # Summarize dataset distribution (optional)
    #Summarize_outbreak_distribution(df=df, test_start_date=TEST_START_DATE, debug=False)

    # Gather data by its location and generate the new dataset
    X, Y, start_window_time_list = data_processing.Generate_data_by_location(df=df, L=constants.L, K=constants.K, debug=False)

    # Split train and test data from the dataset
    X_train, Y_train, X_test, Y_test = data_processing.Split_train_test_data(X=X, Y=Y, start_window_time_list=start_window_time_list, test_start_date=constants.TEST_START_DATE, debug=False)

    for name, model in constants.MODELS_LIST.items():
     Y_proba, Y_pred, learned_thresholds = models.Run_single_model(model_name=name, model=model, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, val_size=constants.VAL_SIZE, thresholds='auto', debug=False)
     metrics = evaluation_logging.Get_metric_results(Y_proba=Y_proba, Y_pred=Y_pred, Y_test=Y_test, debug=True)
     evaluation_logging.Log_results_to_csv(model_name=name, metrics=metrics, learned_thresholds=learned_thresholds, K=constants.K, input_dir=csv_file, output_dir=constants.OUTPUT_DIR, Y_test=Y_test, Y_pred=Y_pred, debug=True)

    for i in range(3, constants.NUMBER_OF_BEST_MODELS_FOR_ENSEMBLE + 4):
      ensemble_Y_pred, ensemble_Y_proba, ensemble_learned_thresholds = models.Run_Ensemble_mean_weight(models_list=constants.MODELS_LIST, main_X_train=X_train, main_Y_train=Y_train, main_X_test=X_test, main_Y_test=Y_test, val_size=constants.VAL_SIZE, thresholds='auto', num_best_models=i,enforce_fixed_threshold=False, equal_weights=False, debug=False)
      ensemble_metrics = evaluation_logging.Get_metric_results(Y_proba=ensemble_Y_proba, Y_pred=ensemble_Y_pred, Y_test=Y_test, debug=False)
      evaluation_logging.Log_results_to_csv(model_name=f'Ensemble (mean weight) top {i} models_TH_auto_not_equal_weight', metrics=ensemble_metrics, learned_thresholds=ensemble_learned_thresholds, K=constants.K, input_dir=csv_file, output_dir=constants.OUTPUT_DIR, Y_test=Y_test, Y_pred=ensemble_Y_pred, debug=True)

    for i in range(3, constants.NUMBER_OF_BEST_MODELS_FOR_ENSEMBLE + 4):
      ensemble_Y_pred, ensemble_Y_proba, ensemble_learned_thresholds = models.Run_Ensemble_mean_weight(models_list=constants.MODELS_LIST, main_X_train=X_train, main_Y_train=Y_train, main_X_test=X_test, main_Y_test=Y_test, val_size=constants.VAL_SIZE, thresholds=0.5, num_best_models=i,enforce_fixed_threshold=True, equal_weights=True, debug=False)
      ensemble_metrics = evaluation_logging.Get_metric_results(Y_proba=ensemble_Y_proba, Y_pred=ensemble_Y_pred, Y_test=Y_test, debug=False)
      evaluation_logging.Log_results_to_csv(model_name=f'Ensemble (mean weight) top {i} model_TH_0.5_equal_weights', metrics=ensemble_metrics, learned_thresholds=ensemble_learned_thresholds, K=constants.K, input_dir=csv_file, output_dir=constants.OUTPUT_DIR, Y_test=Y_test, Y_pred=ensemble_Y_pred, debug=True)

    for i in range(3, constants.NUMBER_OF_BEST_MODELS_FOR_ENSEMBLE + 4):
      ensemble_Y_pred, ensemble_Y_proba, ensemble_learned_thresholds = models.Run_Ensemble_mean_weight(models_list=constants.MODELS_LIST, main_X_train=X_train, main_Y_train=Y_train, main_X_test=X_test, main_Y_test=Y_test, val_size=constants.VAL_SIZE, thresholds='auto', num_best_models=i,enforce_fixed_threshold=False, equal_weights=True, debug=False)
      ensemble_metrics = evaluation_logging.Get_metric_results(Y_proba=ensemble_Y_proba, Y_pred=ensemble_Y_pred, Y_test=Y_test, debug=False)
      evaluation_logging.Log_results_to_csv(model_name=f'Ensemble (mean weight) top {i} models_TH_auto_equal_weight', metrics=ensemble_metrics, learned_thresholds=ensemble_learned_thresholds, K=constants.K, input_dir=csv_file, output_dir=constants.OUTPUT_DIR, Y_test=Y_test, Y_pred=ensemble_Y_pred, debug=True)

    for i in range(3, constants.NUMBER_OF_BEST_MODELS_FOR_ENSEMBLE + 4):
      ensemble_Y_pred, ensemble_Y_proba, ensemble_learned_thresholds = models.Run_Ensemble_mean_weight(models_list=constants.MODELS_LIST, main_X_train=X_train, main_Y_train=Y_train, main_X_test=X_test, main_Y_test=Y_test, val_size=constants.VAL_SIZE, thresholds=0.5, num_best_models=i,enforce_fixed_threshold=True, equal_weights=False, debug=False)
      ensemble_metrics = evaluation_logging.Get_metric_results(Y_proba=ensemble_Y_proba, Y_pred=ensemble_Y_pred, Y_test=Y_test, debug=False)
      evaluation_logging.Log_results_to_csv(model_name=f'Ensemble (mean weight) top {i} models_TH_0.5_not_equal_weight', metrics=ensemble_metrics, learned_thresholds=ensemble_learned_thresholds, K=constants.K, input_dir=csv_file, output_dir=constants.OUTPUT_DIR, Y_test=Y_test, Y_pred=ensemble_Y_pred, debug=True)

# The code will start running from here
if __name__ == '__main__':
  Run()
