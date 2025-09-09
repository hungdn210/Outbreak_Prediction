import pandas as pd 
import os
import constants
import numpy as np

def Get_data(input_csv, start_date, end_date, debug = False):
  # get the data from csv file
  df = pd.read_csv(input_csv, parse_dates=['time'])

  if debug:
    print("Data length: ", len(df))
    print("Data preview:\n", df.head(constants.DEBUG_LENGTH), "\n", df.tail(constants.DEBUG_LENGTH), "\n", df.info())

  # sort the data and remove rows that contain NaN values
  df = df.sort_values('time').dropna().reset_index(drop=True)

  # remove all data that before the start date and after the end date
  df = df[(df['time'] >= start_date) & (df['time'] <= end_date)]

  if debug: 
    print("Data length after removing NaN rows: ", len(df))
    print("Data preview after removing NaN rows:\n", df.head(constants.DEBUG_LENGTH), "\n", df.tail(constants.DEBUG_LENGTH), "\n", df.info())
  
  return df

def Summarize_outbreak_distribution(df, test_start_date, debug=False):
  train_df = df[df['time'] < test_start_date]
  test_df = df[df['time'] >= test_start_date]

  print("\n=== Outbreak Class Distribution Summary ===\n")
  
  if debug: 
    print('Training data preview: \n', train_df.head(constants.DEBUG_LENGTH), '\n', train_df.tail(constants.DEBUG_LENGTH))
    print('\nTesting data preview: \n', test_df.head(constants.DEBUG_LENGTH), '\n', test_df.tail(constants.DEBUG_LENGTH))

  print(f"Train data from {train_df['time'].min()} to {train_df['time'].max()}")
  print("Train class distribution: ", train_df['outbreak'].value_counts())
  print("Percentage of class 0: ", round((len(train_df)-train_df['outbreak'].value_counts())/len(train_df)*100, 2), '%')

  print(f"Test data from {test_df['time'].min()} to {test_df['time'].max()}")
  print("Test class distribution: ", test_df['outbreak'].value_counts())
  print("Percentage of class 0: ", round((len(test_df)-test_df['outbreak'].value_counts())/len(test_df)*100, 2), '%')
  
  total_train_test_length = len(train_df) + len(test_df)
  print("\nTrain class size: ", round(len(train_df)/total_train_test_length * 100, 2), '%')
  print("Test class size: ", round(len(test_df)/total_train_test_length * 100, 2), '%')

def Generate_data_by_location(df, L, K, debug=False):
  X_window_list, Y_window_list, start_window_time_list = [], [], []
  for (lat, lon), group in df.groupby(['lat', 'lon']):
    group = group.sort_values('time').reset_index(drop=True)
    
    if debug:
      print("Group preview:\n", group.head(constants.DEBUG_LENGTH), '\n', group.tail(constants.DEBUG_LENGTH))
    
    end_window_time = len(group) - L - K + 1
    for i in range(end_window_time):
      x_window = group.iloc[i:i+L].drop(columns=['time', 'outbreak', 'area_name', 'lat', 'lon']).values.flatten()
      x_window = np.concatenate(([lat, lon], x_window))
      y_window = group['outbreak'].iloc[i+L:i+L+K].values
      X_window_list.append(x_window)
      Y_window_list.append(y_window)
      start_window_time_list.append(group['time'].iloc[i])

      if debug and (i == 0 or i == end_window_time - 1): 
        print("Current X window range:\n", group['time'].iloc[i:i+L])
        print("Current Y window range:\n", group['time'].iloc[i+L:i+L+K])
        print("Current window time list: \n", start_window_time_list)
        print("x_window: ", x_window)
        print("y_window: ", y_window)
    
  return np.array(X_window_list), np.array(Y_window_list), np.array(start_window_time_list)

def Split_train_test_data(X, Y, start_window_time_list, test_start_date, debug=False):
  X_train, Y_train, X_test, Y_test = [], [], [], []
  
  train_mask = start_window_time_list < test_start_date - pd.DateOffset(months=constants.L)
  test_mask = start_window_time_list >= test_start_date

  X_train, Y_train = X[train_mask], Y[train_mask]
  X_test, Y_test = X[test_mask], Y[test_mask]
  
  if debug:
    print("X_train length: ", len(X_train))
    print("Y_train length: ", len(Y_train))
    print("X_test length: ", len(X_test))
    print("Y_test length: ", len(Y_test))
    print("Training range: ", start_window_time_list[train_mask].min(), '->', start_window_time_list[train_mask].max())
    print("Testing range: ", start_window_time_list[test_mask].min(), '->', start_window_time_list[test_mask].max())
  
  return X_train, Y_train, X_test, Y_test
  