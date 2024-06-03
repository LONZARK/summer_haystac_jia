import numpy as np
import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
import pickle
import geopandas as gpd
from shapely import wkt
import time
import sys
from tqdm import tqdm
import optuna.integration.lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
# import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, roc_curve, auc, average_precision_score
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay



def travel_distance_anomaly(df, normalize_method):
    """
    Calculate the top 3 travel distance anomalies for an agent's daily distances.
    This function computes the Euclidean distances between consecutive geographic points recorded for an agent,
    aggregates these distances by day, calculates the mean daily distance, and identifies the days
    with the top three highest deviations from this mean.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing the agent's data with columns 'geometry', 'time_start'.
    - normalize_method (str): A string to select the normalization method; currently supports 'wo_std'.

    Returns:
    - pd.DataFrame: A DataFrame containing the dates and distance anomalies of the top three anomalies.
    """


    df['geometry'] = df['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    # Calculate Euclidean Distances
    gdf.sort_values(by=['time_start'], inplace=True)
    gdf['distance'] = gdf['geometry'].shift(-1).distance(gdf['geometry'])
    gdf['distance'] = gdf['distance'].fillna(0)

    # Aggregate Distances by Day
    gdf['date'] = pd.to_datetime(gdf['time_start']).dt.date
    daily_distances = gdf.groupby('date')['distance'].sum().reset_index()

    # Calculate Mean and Anomalies
    mean_distance = daily_distances['distance'].mean()
    if normalize_method == 'wo_std':
        daily_distances['anomaly'] = daily_distances['distance'] - mean_distance
    else:
        std_distance = daily_distances['distance'].std()
        daily_distances['anomaly'] = (daily_distances['distance'] - mean_distance) / std_distance

    top_travel_distance_anomaly = daily_distances.nlargest(3, 'anomaly')
    lowest_travel_distance_anomaly = daily_distances.nsmallest(3, 'anomaly')

    travel_distance_anomaly = pd.concat([top_travel_distance_anomaly, lowest_travel_distance_anomaly])
    
    return travel_distance_anomaly


def travel_time_anomaly(df, normalize_method):
    """
    Calculate the top 3 travel time anomalies for an agent's daily travel times.
    This function computes the total travel time for each day, calculates the mean daily travel time, and identifies
    the days with the top three highest deviations from this mean travel time.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing the agent's data with columns 'time_start' and 'time_stop'.

    Returns:
    - pd.DataFrame: A DataFrame containing the dates and travel time anomalies of the top three anomalies.
    """
    
    # Ensure time_start and time_stop are datetime
    df['arrive_time'] = pd.to_datetime(df['time_start'])
    df['depart_time'] = pd.to_datetime(df['time_stop'])

    # Calculate travel time in seconds
    df['travel_time'] = (df['arrive_time'].shift(-1) - df['depart_time']).dt.total_seconds()
    df = df[df['travel_time'] >= 0].copy() # Create a copy of df where travel time is non-negative

    # Aggregate travel times by day
    df['date'] = df['arrive_time'].dt.date
    daily_travel_times = df.groupby('date')['travel_time'].sum().reset_index()

    # Calculate mean and anomalies
    mean_travel_time = daily_travel_times['travel_time'].mean()
    if normalize_method == 'wo_std':
        daily_travel_times['anomaly'] = daily_travel_times['travel_time'] - mean_travel_time
    else:
        std_travel_time = daily_travel_times['travel_time'].std()
        daily_travel_times['anomaly'] = (daily_travel_times['travel_time'] - mean_travel_time) / std_travel_time

    top_travel_time_anomaly = daily_travel_times.nlargest(3, 'anomaly')
    lowest_travel_time_anomaly = daily_travel_times.nsmallest(3, 'anomaly')

    travel_time_anomaly = pd.concat([top_travel_time_anomaly, lowest_travel_time_anomaly])

    return travel_time_anomaly


def unique_locations_anomaly(df, normalize_method):
    """
    Calculate the daily unique locations visited by an agent and identify the deviation from the mean unique locations per day.
    This function counts the unique stop points visited each day, computes the mean of these counts, and finds the daily deviations.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing the agent's data with columns 'time_start' and 'unique_stop_point'.

    Returns:
    - pd.DataFrame: A DataFrame containing the dates, number of unique locations visited, and the deviations from the mean count of unique locations.
    """

    # Ensure time_start is datetime
    df['time_start'] = pd.to_datetime(df['time_start'])

    # Extract the date from time_start
    df['date'] = df['time_start'].dt.date

    # Group by date and count unique stop points
    daily_unique_locations = df.groupby('date')['unique_stop_point'].nunique().reset_index()
    daily_unique_locations.rename(columns={'unique_stop_point': 'unique_locations'}, inplace=True)

    # Calculate the mean of unique locations visited
    mean_unique_locations = daily_unique_locations['unique_locations'].mean()

    # Calculate deviations from the mean
    if normalize_method == 'wo_std':
        daily_unique_locations['anomaly'] = daily_unique_locations['unique_locations'] - mean_unique_locations
    else:
        std_unique_locations = daily_unique_locations['unique_locations'].std()
        daily_unique_locations['anomaly'] = (daily_unique_locations['unique_locations'] - mean_unique_locations) / std_unique_locations

    top_daily_unique_locations = daily_unique_locations.nlargest(3, 'anomaly')
    lowest_daily_unique_locations = daily_unique_locations.nsmallest(3, 'anomaly')

    unique_locations_anomaly = pd.concat([top_daily_unique_locations, lowest_daily_unique_locations])
    
    return unique_locations_anomaly


def duration_anomalies(df, normalize_method):
    """
    Calculate the top 3 highest anomalies in durations spent daily compared to the mean duration for an agent.
    This function computes the total duration spent each day, calculates the mean duration, and finds the days
    with the top three highest deviations from this mean duration.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing the agent's data with columns 'time_start' and 'time_stop'.

    Returns:
    - pd.DataFrame: A DataFrame containing the dates and the duration anomalies of the top three anomalies.
    """

    # Ensure time_start and time_stop are datetime
    df['time_start'] = pd.to_datetime(df['time_start'])
    df['time_stop'] = pd.to_datetime(df['time_stop'])

    # Calculate duration in seconds
    df['duration'] = (df['time_stop'] - df['time_start']).dt.total_seconds()

    # Aggregate durations by day
    df['date'] = df['time_start'].dt.date
    daily_durations = df.groupby('date')['duration'].sum().reset_index()

    # Calculate mean and anomalies
    mean_duration = daily_durations['duration'].mean()
    if normalize_method == 'wo_std':
        daily_durations['anomaly'] = daily_durations['duration'] - mean_duration
    else:
        std_duration = daily_durations['duration'].std()
        daily_durations['anomaly'] = (daily_durations['duration'] - mean_duration) / std_duration

    # Identify top 3 days with the highest duration anomalies
    top_duration_anomalies = daily_durations.nlargest(3, 'anomaly')
    lowest_daily_durations = daily_durations.nsmallest(3, 'anomaly')

    daily_durations_anomaly = pd.concat([top_duration_anomalies, lowest_daily_durations])
    
    return daily_durations_anomaly


def process_multiple_parquets(folder_path, file_list):
    """
    Process multiple parquet files to find travel distance anomalies for each agent.

    Parameters:
    - file_list (list): A list of strings, where each string is the file path to a parquet file.

    Returns:
    - dict: A dictionary where each key is the filename (assuming it's unique per agent) 
            and the value is the DataFrame of top three anomalies.
    """
    feature_dict = {}
    for item in tqdm(file_list):
        file_path = os.path.join(folder_path, item)
        df = pd.read_parquet(file_path)
        agent_id = os.path.splitext(os.path.basename(file_path))[0]  # Assuming file name is the agent_id

        # top3_travel_distance_anomaly = list(travel_distance_anomaly(df, 'wo_std')['anomaly'])
        # top3_travel_time_anomaly = list(travel_time_anomaly(df, 'wo_std')['anomaly'])
        # top3_unique_locations_anomaly = list(unique_locations_anomaly(df, 'wo_std')['anomaly'])
        # top3_duration_anomalies = list(duration_anomalies(df, 'wo_std')['anomaly'])

        top3_last3_travel_distance_anomaly = list(travel_distance_anomaly(df, 'wo_std')['anomaly'])
        top3_last3_travel_time_anomaly = list(travel_time_anomaly(df, 'wo_std')['anomaly'])
        top3_last3_unique_locations_anomaly = list(unique_locations_anomaly(df, 'wo_std')['anomaly'])
        top3_last3_duration_anomalies = list(duration_anomalies(df, 'wo_std')['anomaly'])

        # top3_travel_distance_anomaly = list(travel_distance_anomaly(df, 'w_std')['anomaly'])
        # top3_travel_time_anomaly = list(travel_time_anomaly(df, 'w_std')['anomaly'])
        # top3_unique_locations_anomaly = list(unique_locations_anomaly(df, 'w_std')['anomaly'])
        # top3_duration_anomalies = list(duration_anomalies(df, 'w_std')['anomaly'])

        # features = top3_travel_distance_anomaly + top3_travel_time_anomaly + top3_unique_locations_anomaly + top3_duration_anomalies
        features = top3_last3_travel_distance_anomaly + top3_last3_travel_time_anomaly + top3_last3_unique_locations_anomaly + top3_last3_duration_anomalies
        feature_dict[agent_id] = features
    return feature_dict


def get_group_feature_list(fis_folder, num_subsample_group=None):
    group_feature_list = []
    group_agent_list = []
    group_list = os.listdir(fis_folder)

    if num_subsample_group is not None:
        num_group = num_subsample_group
    else:
        num_group = len(group_list)
    print('num_group', num_group)
    for i in range(num_group):
        temp_group_folder = group_list[i]
        temp_parquet_list = os.listdir(os.path.join(fis_folder, temp_group_folder))
        temp_feature_dict = process_multiple_parquets(os.path.join(fis_folder, temp_group_folder), temp_parquet_list)
        temp_agent_list = [np.int64(temp_file.replace('.parquet', '')) for temp_file in temp_parquet_list]
        
        group_feature_list.append(temp_feature_dict)
        group_agent_list.append(temp_agent_list)
        print('group_feature_list', len(group_feature_list))
        print('group_agent_list', len(group_agent_list))

    return group_feature_list, group_agent_list


def preprocess_val_features(train_dataset_folder, fis_folder):

    group_feature_list , group_agent_list= get_group_feature_list(fis_folder, num_subsample_group=None)
    agent_anomaly_days_dict = np.load(os.path.join(train_dataset_folder, "preprocess/agent_anomaly_days_dict.npy"), allow_pickle=True).item()
    abnormal_agent_id_list = list(agent_anomaly_days_dict.keys())
    print('abnormal_agent_id_list', abnormal_agent_id_list)
    print('abnormal_agent_id_list', type(abnormal_agent_id_list[0]))

    X = []
    y = []

    print('group_feature_list', len(group_feature_list))
    for group in group_feature_list:
        for agent_id, features in group.items():
            X.append(features) 
            if np.int32(agent_id) in abnormal_agent_id_list:
                y.append(1)  
            else:
                y.append(0)  

    # with open(os.path.join(train_dataset_folder,'preprocess/', 'val_wo_std_X_y.pkl'), 'wb') as f:
    #     pickle.dump((X, y), f)
    with open(os.path.join(train_dataset_folder,'preprocess/', 'val_wo_std_X_y_top3_low3.pkl'), 'wb') as f:
        pickle.dump((X, y), f)
    # with open(os.path.join(train_dataset_folder,'preprocess/', 'val_w_std_X_y.pkl'), 'wb') as f:
    #     pickle.dump((X, y), f)

def find_parquet_files(directory):
    """
    Gathers a list of .parquet files in the specified directory,
    excluding 'StopPoints.parquet'.
    """
    parquet_files = []
    # Walk through all files and folders within the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".parquet") and file != "StopPoints.parquet":
                parquet_files.append(file)
    return parquet_files

def preprocess_training_set_features(train_dataset_folder):
    train_df_list = find_parquet_files(train_dataset_folder)
    train_feature_dict = process_multiple_parquets(train_dataset_folder, train_df_list)
    train_agent_list = [np.int64(temp_file.replace('.parquet', '')) for temp_file in train_df_list]

    X = []
    y = []
    for agent_id, features in train_feature_dict.items():
        if len(features) == 24 and not np.isnan(features).any(): ####!!!
            X.append(features) 
            y.append(0)  
    print(len(list(train_feature_dict.keys())))

    # with open(os.path.join(train_dataset_folder,'preprocess/', 'train_wo_std_X_y.pkl'), 'wb') as f:
    #     pickle.dump((X, y), f)

    with open(os.path.join(train_dataset_folder,'preprocess/', 'train_wo_std_X_y_top3_low3.pkl'), 'wb') as f:
        pickle.dump((X, y), f)
    
    # with open(os.path.join(train_dataset_folder,'preprocess/', 'train_w_std_X_y.pkl'), 'wb') as f:
    #     pickle.dump((X, y), f)


def lightgbm_feature_importance(X, y, test_size=0.3, random_state=None, save_path=None):
    if save_path is None:
        save_path = 'lightgbm_feature_importance'
    os.makedirs(save_path, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y )

    num_train_pos = np.sum(y_train==1)
    num_train_neg = np.sum(y_train==0)
    num_test_pos = np.sum(y_test==1)
    num_test_neg = np.sum(y_test==0)
    print(f'Training set number of positive samples: {num_train_pos}, negative samples: {num_train_neg} (ratio: {num_train_pos/(num_train_pos+num_train_neg)})')
    print(f'Testing set number of positive samples: {num_test_pos}, negative samples: {num_test_neg} (ratio: {num_test_pos/(num_test_pos+num_test_neg)})')

    # model = lgb.LGBMClassifier(
    #     num_leaves=31,
    #     max_depth=-1,
    #     learning_rate=0.1,
    #     n_estimators=100,
    #     is_unbalance=True,
    #     #scale_pos_weight=(num_train_pos+num_train_neg)/num_train_pos,
    #     random_state=random_state,
    #     # default='split'. If ‘split’, result contains numbers of times the feature is used in a model. If ‘gain’, result contains total gains of splits which use the feature.
    #     importance_type='split', 
    #     verbosity=-1,
    # )

    # model.fit(X_train, y_train)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'sample_pos_weight': num_train_neg/num_train_pos,
    }

    # tuned from '/home/jxl220096/data/hay/haystac_trial1/fix_Baseline_TA1_Trial_Train_Submission/preprocess/full_train_raw_X_y.pkl'
    # params = {
    #     'objective': 'binary',
    #     'metric': 'binary_logloss',
    #     'verbosity': -1,
    #     'boosting_type': 'gbdt',
    #     'is_unbalanced': True,
    #     'random_state': None,
    #     'feature_pre_filter': False,
    #     'lambda_l1': 1.0404930253233646e-06,
    #     'lambda_l2': 0.002242148126928162,
    #     'num_leaves': 3,
    #     'feature_fraction': 0.9799999999999999,
    #     'bagging_fraction': 0.7531541955265642,
    #     'bagging_freq': 1,
    #     'min_child_samples': 20,
    #     'num_iterations': 1000
    # }

    dataset_train = lgb.Dataset(X_train, label=y_train)
    dataset_test = lgb.Dataset(X_test, label=y_test)

    model = lgb.train(
        params,
        dataset_train,
        valid_sets=[dataset_train, dataset_test],
        callbacks=[early_stopping(100), log_evaluation(100)],
    )

    best_params = model.params
    print('Best parameters:')
    print(best_params)

    y_score = model.predict(X_train)
    print('===  Training  ===')
    train_metrics, train_fig = plot_auroc_aupr(y_train, y_score)

    y_score = model.predict(X_test)
    print('=== Testing ===')
    test_metrics, test_fig = plot_auroc_aupr(y_test, y_score)

    importance = model.feature_importance('split')

    print('===            ===')
    print(f'Feature importance: {importance}')

    importance_gain = model.feature_importance('gain')

    all_metrics = {
        'train': train_metrics,
        'test': test_metrics,
        'feature_importance': importance.tolist(),
        'feature_importance_gain': importance_gain.tolist(),
    }
    if save_path is not None:
        model.save_model(os.path.join(save_path, 'model.lgb'))
        with open(os.path.join(save_path, 'params.json'), 'w') as f:
            json.dump(best_params, f, indent=4)

        with open(os.path.join(save_path, 'metrics.json'), 'w') as f:
            json.dump(all_metrics, f, indent=4)

        train_fig.savefig(os.path.join(save_path, 'train_roc_pr.png'))
        test_fig.savefig(os.path.join(save_path, 'test_roc_pr.png'))

    return importance, model, all_metrics


def plot_auroc_aupr(y_true, y_score, verbose=True):
    y_pred = np.rint(y_score)
    acc = accuracy_score(y_true, y_pred)

    if verbose:
        print(f'Accuracy: {acc}')

    cm = confusion_matrix(y_true, y_pred)
    if verbose:
        print('Confusion Matrix:')
        print(cm)

    # ROC, PR
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    # aupr = average_precision_score(y_true, y_score)
    try:
        aupr = auc(recall, precision)
    except ValueError as e:
        traceback.print_exc()
        aupr = 0.0

    if verbose:
        print(f'AUPR: {aupr}')

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    try:
        auroc = auc(fpr, tpr)
    except ValueError as e:
        traceback.print_exc()
        auroc = 0.0
    
    if verbose:
        print(f'AUROC: {auroc}')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)
    roc_display.plot(ax=ax1, label='roc')
    pr_display.plot(ax=ax2, label='prc')

    metrics = {
        'accuracy': acc,
        'confusion_matrix': cm.tolist(),
        'aupr': aupr,
        'auroc': auroc,
    }
    return metrics, fig


if __name__ == '__main__':

    start = time.time()
    args = sys.argv
    if len(args) == 4:
        train_dataset_folder = args[1]
        fis_folder = args[2]
        dataset = args[3]
    else:
        raise Exception("no dataset")

    # preprocess_val_features(train_dataset_folder, fis_folder)
    preprocess_training_set_features(train_dataset_folder)

    # with open(train_dataset_folder+'/preprocess/val_wo_std_X_y.pkl', 'rb') as f:
    #     X1, y1 = pickle.load(f)
    # with open(train_dataset_folder+'/preprocess/train_wo_std_X_y.pkl', 'rb') as f:
    #     X2, y2 = pickle.load(f)

    with open(train_dataset_folder+'/preprocess/val_wo_std_X_y_top3_low3.pkl', 'rb') as f:
        X1, y1 = pickle.load(f)
    with open(train_dataset_folder+'/preprocess/train_wo_std_X_y_top3_low3.pkl', 'rb') as f:
        X2, y2 = pickle.load(f)

    # with open(train_dataset_folder+'/preprocess/val_w_std_X_y_top3_low3.pkl', 'rb') as f:
    #     X1, y1 = pickle.load(f)
    # with open(train_dataset_folder+'/preprocess/train_w_std_X_y_top3_low3.pkl', 'rb') as f:
    #     X2, y2 = pickle.load(f)

    print(len(X1[0]))
    print(len(X2))

    print(f"total runtime: {time.time() - start}s")

# python gen_lightgbm_feature.py /home/jxl220096/data/hay/new_format/trial2/losangeles/train_stops /home/jxl220096/data/hay/new_format/trial2/losangeles/test_stops_valsplit losangeles
# python gen_lightgbm_feature.py /home/jxl220096/data/hay/new_format/trial2/sanfrancisco/train_stops /home/jxl220096/data/hay/new_format/trial2/sanfrancisco/test_stops_valsplit sanfrancisco_wo_std
# python gen_lightgbm_feature.py /home/jxl220096/data/hay/new_format/trial2/singapore/train_stops /home/jxl220096/data/hay/new_format/trial2/singapore/test_stops_valsplit singapore_wo_std
# python gen_lightgbm_feature.py /home/jxl220096/data/hay/new_format/trial2/knoxvile/train_stops /home/jxl220096/data/hay/new_format/trial2/knoxvile/test_stops_valsplit knoxvile_wo_std