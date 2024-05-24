import copy
import os
# Setting CUDA_VISIBLE_DEVICES within the Python script
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
try:
    import faiss
except ImportError:
    faiss = None  # or handle it in another way
# read the following page about how to install faiss-gpu:
# https://github.com/Victorwz/LongMem/blob/93a4bb8b106261c2734d25d3d0e0e85e451bbad0/README.md
    
import numpy as np
# import matplotlib as plt
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
import datetime
import math
from os import listdir
from os.path import isfile, join
import os
import statistics
import pickle
# import dill as pickle
import multiprocessing
from scipy.stats import gaussian_kde
import time
import argparse
import sys
from sklearn.preprocessing import RobustScaler
from distutils.dir_util import copy_tree
from sklearn.mixture import GaussianMixture
from scipy.stats import wasserstein_distance
from sklearn.cluster import AgglomerativeClustering
import json
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
# from spatiotemp_feat_ext import train_test_spatiotemporal_density_feat_ext, edge_spatiotemporal_density_feat_ext
from tqdm import tqdm
import os
import psutil
import psutil
import FastLOF, FastKNN
from sklearn.preprocessing import StandardScaler
import random
import shutil
# from insert_utils.trajectory_insertion_feng import *
# from insert_utils.preprocessor_update import Data_preprocessor
# from hay_tasks.code_update.pipeline_feb27_feng.insert_utils.utilsmilp import *
import matplotlib.pyplot as plt
import networkx as nx

from haystac_kw.ta1.agent_behavior_sim.map import OfflineMap
from haystac_kw.data.schemas.has import InternalHAS
from haystac_kw.data.types.itinerary import InternalStart, InternalMove, Stay, InternalMovements, InternalItinerary
from haystac_kw.data.types.time import TimeWindow
import gc
import dill
from sklearn.preprocessing import MinMaxScaler
from pyod.models.cblof import CBLOF
from sklearn.neighbors import NearestNeighbors

from fengutils import *
import networkx as nx
import scipy as sp
from sklearn.metrics import precision_recall_curve, roc_curve, auc



def subsample_train_files(train_parquet_folder, abnormal_agent_id_list, subsample_size):
    """
    Subsample file names from the train dataset folder ensuring that all file names
    from the test dataset folder are included in the subsample with a fixed seed.
    the subsample file names will be the same for different runs.

    Parameters:
    - train_dataset_folder:     Path to the folder containing the training dataset files.
    - abnormal_agent_id_list:   a list of abnormal agent ids
    - subsample_size:           Desired size of the subsample. Must be at least as large as
                                the number of files in the test dataset folder.

    Returns:
    - subsampled_filenames: A list of subsampled filenames from the training dataset.
    """

    # Get all filenames in the test dataset folder
    # test_filenames = set(os.listdir(test_dataset_folder)) #- set(['1.parquet', '2.parquet'])

    test_filenames = set(["{}.parquet".format(agent_id) for agent_id in abnormal_agent_id_list])
    # Ensure the subsample size is at least as large as the number of test files
    num_test_files = len(test_filenames)
    if subsample_size < num_test_files:
        raise ValueError("Subsample size must be at least as large as the number of test files.")

    # Get all filenames in the train dataset folder
    train_files_list = [filename for filename in os.listdir(train_parquet_folder) if filename.replace('.parquet', '').isdigit()]
    train_filenames = set(train_files_list) #- set(['1.parquet', '2.parquet'])
    # print(len(set(os.listdir(train_dataset_folder))), len(train_filenames))
    # Initialize the subsample with the test filenames

    subsampled_filenames = list(test_filenames)

    # Calculate remaining number of filenames needed
    remaining_size = subsample_size - num_test_files

    # Ensure the sets are sorted for consistency
    remaining_filenames = sorted(list(train_filenames - test_filenames))

    # Add random filenames from the train dataset excluding the test filenames
    random.seed(42)
    # print(type(train_filenames), type(test_filenames))
    additional_filenames = random.sample(remaining_filenames, remaining_size)
    subsampled_filenames.extend(additional_filenames)

    return subsampled_filenames


class Consumer(multiprocessing.Process):

    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        # proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means we should exit
                # print
                # '%s: Exiting' % proc_name
                break
            # print
            # '%s: %s' % (proc_name, next_task)
            answer = next_task()
            self.result_queue.put(answer)
        return

class FeatureExtractionTask(object):
    def __init__(self, parquet_fpath_list, train_dataset_folder, test_dataset_folder, loc_coord_dict, chunk_agent_2_date_2_his_coord_set_dict = None, subsample_label = "", train_phase_boolean = True, have_tel = True, abnormal_agent_id_list = None, anomaly_agent_index_dict = None, params = None, params_folder_name = ""):
        self.parquet_fpath_list     = parquet_fpath_list
        self.train_dataset_folder   = train_dataset_folder
        self.test_dataset_folder    = test_dataset_folder
        self.loc_coord_dict         = loc_coord_dict
        self.chunk_agent_2_date_2_his_coord_set_dict = chunk_agent_2_date_2_his_coord_set_dict
        self.subsample_label        = subsample_label
        self.train_phase_boolean    = train_phase_boolean
        self.have_tel               = have_tel
        self.abnormal_agent_id_list = abnormal_agent_id_list
        self.anomaly_agent_index_dict = anomaly_agent_index_dict 
        self.params                 = params
        self.params_folder_name     = params_folder_name

    def __call__(self):

        chunk_agents_id                         = []
        # chunk_agent_2_KDEs_train_data_dict      = dict()  # {hos agent ID: KDEs_train_data}
        chunk_agent_2_his_coord_set_dict        = dict()  # {hos agent ID: faiss_indices_train_data}
        chunk_X                                 = None
        chunk_id_agent_X_array                  = None
        chunk_agent_2_date_2_his_coord_set_dict = dict()
        chunk_trip_start_end_datetimes_df       = None
        agent_2_date_2_his_coord_set_dict       = None
        chunk_trip_df_idx                        = None
        chunk_raw_X                             = None 
        # chunk_roadseg_stopp_duration_all_dict         = dict()  # {location ID: list of durations} (event level) [id of each event, duration between this event and its subsequent event]
        # chunk_travel_time_dict                        = dict()  # {stopp ID: {stopp ID: list of travel times}}
        # chunk_travel_time_agent_dict                  = dict()  # {agent ID: travel_time_dict}
        # chunk_transition_agent_count_dict             = dict()  # {agent ID: transition_count_dict}, transition_count_dict: {stopp ID: {stopp ID: count of transition observations}}
        # chunk_duration_agent_dict                     = dict()  # {agent ID: {stopp ID: list of durations}}
        # chunk_transition_agent_count_dict             = dict()
        # chunk_duration_agent_dict                     = dict()
        # chunk_trip_road_segments_list                 = []      # [list of locations (LocationUUID) for each trip]
        # chunk_duration_dict                           = dict()  # {stopp ID: list of durations}
        # chunk_agent_vertex_dict                       = dict()  # {agent ID: list of stopp IDs visited in the train and NAT}
        # chunk_transition_count_dict                   = dict()  # {stopp ID: {stopp ID: count of transition observations}}

        # for idx, parquet_fpath in enumerate(self.parquet_fpath_list):
        #     agent_id = int(parquet_fpath.split('/')[-1].split('.')[0]) # agent ID should be integer format by default
        #     chunk_agents_id.append(agent_id)

        print("!!!!!!!!!!! {} files:  process started".format(len(self.parquet_fpath_list)))
        # if self.train_phase_boolean == False: # if the current phase is the test phase, load the trained KDE estimators
        #     # agent_2_KDEs_dict                   = load_from_pickle(self.train_dataset_folder + "preprocess/" + "agent_2_KDEs_dict.pkl")
        #     agent_2_date_2_his_coord_set_dict   = load_from_pickle(self.train_dataset_folder + "preprocess/" + "{}{}train_agent_2_date_2_his_coord_set_dict.pkl".format(self.params_folder_name, self.subsample_label))
            
        agent_2_date_2_stopp_coords_dict = dict()

        for idx, parquet_fname in enumerate(self.parquet_fpath_list):
            try: #
                if idx % 500 == 0: 
                    print("idx, parquet_fname", idx, parquet_fname)

                agent_id = np.int64(parquet_fname.split('/')[-1].split('.')[0]) # agent ID should be integer format by default

                if self.chunk_agent_2_date_2_his_coord_set_dict is not None: 
                    date_2_his_coord_set_dict = self.chunk_agent_2_date_2_his_coord_set_dict[agent_id] 
                else: 
                    date_2_his_coord_set_dict = None
                # agent_id, trip_datetime_df, event_loc_type_dict, event_loc_adj_list, stopp_duration_dict, \
                #     roadseg_stopp_duration_returnlist, stopp2stopp_traveltime_dict, trip_locations, X, \
                #         KDEs_train_data, stopps_NN_faiss_train_data, date_2_his_coord_set_dict \
                agent_id, trip_datetime_df, X, agent_id_X_array, his_coord_set, date_2_his_coord_set_dict, trip_start_end_datetimes_df, trip_df_idx, _, temp_date_2_stopp_coords_dict, raw_X \
                     = trip_feature_extraction(parquet_fname, self.train_dataset_folder, self.test_dataset_folder, self.loc_coord_dict, \
                                               date_2_his_coord_set_dict, self.subsample_label, self.train_phase_boolean, abnormal_agent_id_list = self.abnormal_agent_id_list,\
                                                  anomaly_agent_index_dict = self.anomaly_agent_index_dict, params = self.params, params_folder_name = self.params_folder_name)
                agent_2_date_2_stopp_coords_dict[agent_id] = temp_date_2_stopp_coords_dict
                chunk_agents_id.append(agent_id)

                if X.ndim == 0:
                    print("!!!!!!!!!!!!!!!!!!!!!!!{} X has zero dimensions".format(parquet_fname))
                    continue 

                if chunk_X is None:
                    chunk_X                 = X
                    chunk_raw_X             = raw_X
                    chunk_id_agent_X_array  = agent_id_X_array
                    chunk_trip_df_idx        = trip_df_idx
                    chunk_trip_start_end_datetimes_df = trip_start_end_datetimes_df
                else:
                    chunk_X                 = np.concatenate((chunk_X, X), axis = 0)
                    chunk_raw_X             = np.concatenate((chunk_raw_X, raw_X), axis = 0)
                    chunk_id_agent_X_array  = np.concatenate((chunk_id_agent_X_array, agent_id_X_array), axis = 0)
                    chunk_trip_df_idx.extend(trip_df_idx)
                    chunk_trip_start_end_datetimes_df = pd.concat([chunk_trip_start_end_datetimes_df, trip_start_end_datetimes_df], axis = 0)
                    # np.concatenate((chunk_trip_start_end_datetimes_df, trip_start_end_datetimes_df), axis = 0)

                if self.train_phase_boolean == True:
                    # chunk_agent_2_KDEs_train_data_dict[agent_id]     = KDEs_train_data # [n by 2 array, ]
                    chunk_agent_2_his_coord_set_dict[agent_id]       = his_coord_set
                    chunk_agent_2_date_2_his_coord_set_dict[agent_id]= date_2_his_coord_set_dict

                # roadseg_stopp_duration_returnlist: (event level) [id of each event, duration between this event and its subsequent event]
                # for item in roadseg_stopp_duration_returnlist:
                #     if item[0] not in chunk_roadseg_stopp_duration_all_dict.keys():
                #         chunk_roadseg_stopp_duration_all_dict[item[0]] = [item[1]]
                #     else:
                #         chunk_roadseg_stopp_duration_all_dict[item[0]].append(item[1])

                # # (trip level) [list of locations (LocationUUID) for each trip]
                # chunk_trip_road_segments_list = chunk_trip_road_segments_list + trip_locations

                # i_transition_count_dict = dict()
                # # stopp2stopp_traveltime_dict: {stopp: {stopp: list of travel times}}
                # for stopp1, stopp2_traveltime_list in stopp2stopp_traveltime_dict.items():
                #     for stopp2, traveltime_list in stopp2_traveltime_list.items():
                #         if stopp1 not in i_transition_count_dict:
                #             i_transition_count_dict[stopp1] = dict()
                #         if stopp2 not in i_transition_count_dict[stopp1]:
                #             i_transition_count_dict[stopp1][stopp2]  = len(traveltime_list)
                #         else:
                #             i_transition_count_dict[stopp1][stopp2] += len(traveltime_list)

                #         if stopp1 not in chunk_travel_time_dict:
                #             chunk_travel_time_dict[stopp1]         = dict()
                #             chunk_transition_count_dict[stopp1]    = dict()

                #         if stopp2 not in chunk_travel_time_dict[stopp1]:
                #             chunk_travel_time_dict[stopp1][stopp2]      = traveltime_list
                #             chunk_transition_count_dict[stopp1][stopp2] = len(traveltime_list)
                #         else:
                #             chunk_travel_time_dict[stopp1][stopp2].extend(traveltime_list)
                #             chunk_transition_count_dict[stopp1][stopp2] += len(traveltime_list)

                # chunk_duration_dict = merge_dict_combine_list(chunk_duration_dict, stopp_duration_dict)

                # if agent_id in self.agents_hos:
                #     chunk_transition_agent_count_dict[agent_id] = i_transition_count_dict
                #     chunk_travel_time_agent_dict[agent_id]      = stopp2stopp_traveltime_dict
                #     chunk_duration_agent_dict[agent_id]         = stopp_duration_dict
                #     chunk_agent_vertex_dict[agent_id]           = list(stopp_duration_dict.keys())
                #     chunk_hos_agent_KDEs_train_data_dict[agent_id]           = KDEs_train_data
                #     chunk_hos_agent_faiss_indices_train_data_dict[agent_id]  = stopps_NN_faiss_train_data
            except Exception as error:
                print(parquet_fname)
                print(error, parquet_fname)
        print("!!!!!!!!!!! {} files:  process ended".format(len(self.parquet_fpath_list)))
        return chunk_agents_id, chunk_X, chunk_id_agent_X_array, chunk_agent_2_his_coord_set_dict, chunk_agent_2_date_2_his_coord_set_dict, chunk_trip_start_end_datetimes_df, chunk_trip_df_idx, agent_2_date_2_stopp_coords_dict, chunk_raw_X
        # agent_id, trip_datetime_df, X, KDEs_train_data, his_coord_set, date_2_his_coord_set_dict
    def __str__(self):
        return self.parquet_fname_list


def merge_train_test_stopp_parquet(train_dataset_dir, test_dataset_dir):
    """
    This function aims to merge the stop points in both the training and test sets.
    /data/jxl220096/hay/haystac_trial2/train/sanfrancisco_joined_train/StopPoints.parquet
    /data/jxl220096/hay/haystac_trial2/nat/sanfrancisco_joined_test/StopPoints.parquet
    The merged stop points are stored as a pandas dataframe in the files:
        os.path.join(train_dataset_dir,"merged_stopp.parquet")
        os.path.join(test_dataset_dir,"merged_stopp.parquet")
    the pandas dataframe has the following columns:
        Data columns (total 3 columns):
        #   Column        Non-Null Count  Dtype
        ---  ------        --------------  -----
        0   LocationUUID  15631 non-null  object
        1   Latitude      15631 non-null  float64
        2   Longitude     15631 non-null  float64
        dtypes: float64(2), object(1)
    The following are the top for rows:
                                LocationUUID   Latitude   Longitude
        0  4e2112db-daa3-45a2-a881-22b6d1477927  37.713923 -122.441357
        1  ad35bd18-be10-4e56-b456-1d615998ad00  37.750026 -122.388772
        2  af2e3d58-d4ba-4f0b-8c81-441a8c26f4d7  37.746500 -122.421117
        3  b35ca40f-ffd0-44f4-ad21-ef5e6b785ce8  37.713152 -122.420505
        4  cdfd3645-b540-4882-a773-5504bacb880e  37.711352 -122.460170
    INPUT
    train_dataset_dir: The folder address where StopPoints.parquet files related to training are stored.
    test_dataset_dir: The folder address where StopPoints.parquet files related to testing are stored.
    """
    train_stopp_file = ""
    for root, ds, fs in os.walk(train_dataset_dir):
        break_flag = False
        for f in sorted(fs):
            if f.endswith('StopPoints.parquet'):
                train_stopp_file = os.path.join(root, f)
                break_flag = True
                break
        if break_flag:
            break
    # StopPoints.parquet file name related to training
    test_stopp_file = ""
    for root, ds, fs in os.walk(test_dataset_dir):
        break_flag = False
        for f in sorted(fs):
            if f.endswith('StopPoints.parquet'):
                test_stopp_file = os.path.join(root, f)
                break_flag = True
                break
        if break_flag:
            break
    hos_stopp_file = ""
    for root, ds, fs in os.walk(os.path.join(test_dataset_dir, 'HOS')):
        break_flag = False
        for f in sorted(fs):
            if f.endswith('StopPoints.parquet'):
                hos_stopp_file = os.path.join(root, f)
                break_flag = True
                break
        if break_flag:
            break

    print(train_stopp_file)
    print(test_stopp_file)
    print(hos_stopp_file)
    # StopPoints.parquet file name related to testing
    stopp_files_list = [train_stopp_file, test_stopp_file, hos_stopp_file]
    stopp_df_list = []
    for temp_stopp_file in stopp_files_list:
        if temp_stopp_file != "":
            temp_stopp_df = pd.read_parquet(temp_stopp_file)
            stopp_df_list.append(temp_stopp_df)
    merged_stopp = pd.concat(stopp_df_list, axis=0)
    merged_stopp.drop_duplicates(subset='LocationUUID', keep='first', inplace=True)
    merged_stopp.to_parquet(os.path.join(train_dataset_dir,"merged_stopp.parquet"))
    merged_stopp.to_parquet(os.path.join(test_dataset_dir,"merged_stopp.parquet"))





def multi_anomlay_detectors_proc(X, detector_save_dir, outliers_fraction = 0.00128, params = None):
    """
    INPUT
    X: n by p matrix, where n is the number of data records and p is the number of features
    detector_save_dir: the folder where the anomaly detectors are stored.

    columns in X: travel_time (0), Euclidan distance (1), distance of start stopp to historical stopps (2),
    distance of end stopp to historical stopps (3), depature time density (4), depature spatiotemporal density (5), 
    start stopp duration (6), end stopp duration (7), date travel time density (8)

    OUTPUT
    result: n by 2, column 1 are the anomaly scores based on FastLOF for the n data records in X
                    column 2 are the anomaly scores based on FastKNN for the n data records in X
    """
    if not os.path.exists(detector_save_dir):
        os.makedirs(detector_save_dir)
    if params is not None: 
        classifiers = {
            'FastLOF':    FastLOF.FastLOF(contamination=outliers_fraction, novelty=True, n_neighbors = params.K_LOF),
            'FastKNN':    FastKNN.FastKNN(contamination=outliers_fraction, n_neighbors = params.K_KNN),
            # 'FastSelKNN': FastKNN.FastKNN(contamination=outliers_fraction, n_neighbors = params.K_KNN),
            # 'CBLOF':      CBLOF(contamination=outliers_fraction)
        }
    else:
        classifiers = {
            'FastLOF':    FastLOF.FastLOF(contamination=outliers_fraction, novelty=True),
            'FastKNN':    FastKNN.FastKNN(contamination=outliers_fraction),
            # 'FastSelKNN': FastKNN.FastKNN(contamination=outliers_fraction),
            # 'CBLOF':      CBLOF(contamination=outliers_fraction)
        }
        
    result = []
    for clf_name, clf in classifiers.items():
        model_path = os.path.join(detector_save_dir, f'{clf_name}.pkl')
        if clf_name == 'FastLOF':
            clf.fit(X)
            test_scores = clf.decision_function(X)
            save_fast_lof(clf, detector_save_dir + 'FastLOF.pkl')
        elif clf_name == 'FastKNN':
            clf.fit(X)
            test_scores = clf.decision_function(X)
            save_fast_knn(clf, detector_save_dir + 'FastKNN.pkl')
        elif clf_name == 'FastSelKNN':
            X_sel = X[:,[4, 5, 8]]
            clf.fit(X_sel)
            test_scores = clf.decision_function(X_sel)
            save_fast_knn(clf, detector_save_dir + 'FastSelKNN.pkl')
        elif clf_name == 'CBLOF':
            X_sel = X[:,[4, 5, 8]]
            clf.fit(X_sel)
            test_scores = clf.decision_function(X_sel)
            save_to_pickle(detector_save_dir + 'CBLOF.pkl', clf)
        result.append(list(test_scores)) # Transpose to get the desired n by 2 shape
    result = np.array(result).T
    return result, classifiers




def read_traveldis(df, have_tel = True):
    """
    INPUT:
    df:         event list of a specific agent in the format of pandas data frame.
                columns: index, agent_id, timestamp, EventType, LocationType, LocationUUID, datetime, geometry, latitude, longitude
    have_tel:   True/False. True means that the trips have teleporting trips

    OUTPUT:
    The following are all related to one specific agent

    trip_features:      list (trip level) [start hour of day (float),end hour of day (float), travel_time in hours (float),
                                            travel_distance,start_lat,start_lon,end_lat,end_lon,day_week,
                                            start_dur (hours (float)),end_dur (hours (float))]
    trip_points:        list (trip level) [end_lat, end_lon]
    trip_datetime:      pandas data frame (trip level) [start_datetime, end_datetime, start_arr_datetime, end_dep_datetime]
                        start_datetime: the depature time of the start SP of each trip
                        end_datetime: the arrival time of the end SP of each trip
                        start_arr_datetime: the arrival time of the start SP
                        end_dep_datetime: the depature time of the end SP
    trip_locations:     list (trip level) [list of locations (LocationUUID) for each trip]
    trip_df_idx:        List (trip level) [start and end event indices of each trip]


    event_loc_type_dict: Dictionary (event level) {LocationUUID: LocationType}
    event_loc_adj_list:  List (event level) [LocationUUIDs of two adjacent events]
    roadseg_stopp_duration_returnlist: List (event level) [id of each event, duration between this event and its subsequent event]
    file_stopp2stopp_list (teleporting):       List of teleporting trips [start SP1 (LocationUUID) of a trip, end SP2 (LocationUUID) of the trip, travel time between the two SPs (arrival time of SP 2 - depature time of SP 1)]
                                        We require that the start and end SPs of each trip should not have  the same LocationUUID
    """
    df['arrival_day'] = df['arrival_datetime'].apply(lambda x: x.weekday())
    df['depart_day'] = df['depart_datetime'].apply(lambda x: x.weekday())
    df['Longitude'] = df['geometry'].apply(lambda x: float(x.replace('POINT (', '').replace(')','').split(' ')[0]))
    df['Latitude'] = df['geometry'].apply(lambda x: float(x.replace('POINT (', '').replace(')', '').split(' ')[1]))
    df = df.sort_values(by = ['time_start'], ascending = [True]).reset_index()
    trip_features = []
    trip_points = []
    count = 0
    trip_datetime_df = pd.DataFrame(columns=['start_datetime','end_datetime','start_arr_datetime','end_dep_datetime'])
    trip_locations = []
    trip_df_idx = []
    for i in range(df.shape[0]-1):
        start_index = i
        start_row = df.iloc[start_index]
        end_index = i + 1
        end_row = df.iloc[end_index]
        count = count + 1
        start_hour_of_day = float(start_row['depart_datetime'].hour) + float(start_row['depart_datetime'].minute) / 60
        end_hour_of_day = float(end_row['arrival_datetime'].hour) + float(end_row['arrival_datetime'].minute) / 60
        travel_time = diff_hour(start_row['depart_timestamp'], end_row['arrival_timestamp'])
        start_lat = start_row['Latitude']
        start_lon = start_row['Longitude']
        end_lat = end_row['Latitude']
        end_lon = end_row['Longitude']
        points=[[start_lat,start_lon], [end_lat,end_lon]]
        temp_list = [start_row['global_stop_points'], end_row['global_stop_points']]
        travel_distance = math.sqrt((start_lon - end_lon) ** 2 + (start_lat - end_lat) ** 2)
        trip_locations.append(temp_list)
        trip_points.append(points)

        day_week = float(start_row['depart_day'])
        start_dur = diff_hour(start_row['arrival_timestamp'], start_row['depart_timestamp'])
        end_dur = diff_hour(end_row['arrival_timestamp'], end_row['depart_timestamp'])
        trip_features.append([start_hour_of_day,end_hour_of_day,travel_time,travel_distance,start_lat,start_lon,end_lat,end_lon,day_week,start_dur,end_dur])

        # trip_datetime: pandas data frame [start_datetime, end_datetime, start_arr_datetime, end_dep_datetime]
        # start_datetime: the depature time of the start SP of a trip
        # end_datetime: the arrival time of the end SP of the trip
        # start_arr_datetime: the arrival time of the start SP
        # end_dep_datetime: the depature time of the end SP
        trip_datetime_df.loc[len(trip_datetime_df.index)] = [start_row['depart_datetime'],end_row['arrival_datetime'],start_row['arrival_datetime'],end_row['depart_datetime']]

        # trip_df_idx: [start and end event indices of each trip]
        trip_df_idx.append([start_index, end_index])
    return trip_features, trip_points, count, df.iloc[0]['agent_id'], trip_datetime_df, trip_locations, trip_df_idx


def debug_check(stopp_duration_dict, stopp2stopp_traveltime_dict):
    stopp_dict = dict()
    for i_stopp, j_stopp_traveltimes in stopp2stopp_traveltime_dict.items():
        stopp_dict[i_stopp] = 1
        for j_stopp, traveltimes in j_stopp_traveltimes.items():
            stopp_dict[j_stopp] = 1

    n_mismatch = 0
    for i_stopp in stopp_dict:
        if i_stopp not in stopp_duration_dict:
            # print("!!!!!!!!!!!!!!!!! stopp2stopp_traveltime_dict not in stopp_duration_dict", i_stopp)
            n_mismatch += 1

    for i_stopp in stopp_duration_dict:
        if i_stopp not in stopp_dict:
            # print("!!!!!!!!!!!!!!!!! stopp_duration_dict not in stopp2stopp_traveltime_dict", i_stopp)
            n_mismatch += 1

    print("!!!!!!!!!!!!!!!!! stopp2stopp_traveltime_dict not in stopp_duration_dict: ", n_mismatch)




def read_traveldis_new(df, have_tel = True, min_duration_seconds = 60 * 2):
    """
    INPUT:
    df:         event list of a specific agent in the format of pandas data frame.
                columns: index, agent_id, timestamp, EventType, LocationType, LocationUUID, datetime, geometry, latitude, longitude
    have_tel:   True/False. True means that the trips have teleporting trips

    OUTPUT:
    The following are all related to one specific agent

    trip_features:      list (trip level) [start hour of day (float),end hour of day (float), travel_time in hours (float),
                                            travel_distance,start_lat,start_lon,end_lat,end_lon,day_week,
                                            start_dur (hours (float)),end_dur (hours (float))]
    trip_points:        list (trip level) [end_lat, end_lon]
    trip_datetime:      pandas data frame (trip level) [start_datetime, end_datetime, start_arr_datetime, end_dep_datetime]
                        start_datetime: the depature time of the start SP of each trip
                        end_datetime: the arrival time of the end SP of each trip
                        start_arr_datetime: the arrival time of the start SP
                        end_dep_datetime: the depature time of the end SP
    trip_locations:     list (trip level) [list of locations (LocationUUID) for each trip]
    trip_df_idx:        List (trip level) [start and end event indices of each trip]


    event_loc_type_dict: Dictionary (event level) {LocationUUID: LocationType}
    event_loc_adj_list:  List (event level) [LocationUUIDs of two adjacent events]
    roadseg_stopp_duration_returnlist: List (event level) [id of each event, duration between this event and its subsequent event]
    file_stopp2stopp_list (teleporting):       List of teleporting trips [start SP1 (LocationUUID) of a trip, end SP2 (LocationUUID) of the trip, travel time between the two SPs (arrival time of SP 2 - depature time of SP 1)]
                                        We require that the start and end SPs of each trip should not have  the same LocationUUID
    """
    df["day"] = df['datetime'].apply(lambda x: x.weekday())
    df = df.sort_values(by = ['datetime','EventType'], ascending = [True,False]).reset_index()
    # df = df.drop(df[(df['LocationType']=='ROAD_EDGE')&(df['geometry'].isnull())].index)
    trip_features = []
    trip_points = []
    count = 0
    trip_datetime_df = pd.DataFrame(columns=['start_datetime','end_datetime','start_arr_datetime','end_dep_datetime'])
    if not have_tel:
        del_stoppoint_index = []
        for i in range(len(df.index) - 3):
            if (df.iloc[i]['LocationType'] == 'STOP_POINT' and df.iloc[i + 1]['LocationType'] == 'STOP_POINT' and \
                    df.iloc[i + 2]['LocationType'] == 'STOP_POINT' and df.iloc[i + 3]['LocationType'] == 'STOP_POINT' and
                    df.iloc[i + 3]['LocationUUID'] == df.iloc[i + 2]['LocationUUID'] == df.iloc[i + 1]['LocationUUID'] == df.iloc[i]['LocationUUID']):
                del_stoppoint_index.append(i + 1)
                del_stoppoint_index.append(i + 2)
            # if df.iloc[i+1]['LocationType'] == df.iloc[i + 2]['LocationType'] == 'STOP_POINT' and df.iloc[i+1]['EventType'] == 'DEPART' and df.iloc[i+2]['EventType'] == 'ARRIVE' and df.iloc[i+1]['LocationUUID'] != df.iloc[i+2]['LocationUUID']:
            #     raise Exception("Jump from stopp to stopp")
        df = df.drop(del_stoppoint_index)
    trip_locations = []
    trip_df_idx = []
    for i in range(df.shape[0]-1):
        start_index = i
        start_row = df.iloc[i]
        # if start_row['LocationType'] == 'STOP_POINT' and start_row['EventType'] == 'DEPART' and df.iloc[i+1]['LocationType'] != 'STOP_POINT': #trip start
        if (start_row['LocationType'] == 'STOP_POINT' and start_row['EventType'] == 'DEPART'
                and (have_tel or df.iloc[i+1]['LocationType'] != 'STOP_POINT')):
            end_index = -1
            j = i+1
            while j < df.shape[0]-1:
                if df.iloc[j]['LocationType'] == 'STOP_POINT' and df.iloc[j]['EventType'] == 'ARRIVE': 
                    if df.iloc[j+1]['LocationType'] == 'STOP_POINT' and df.iloc[j+1]['EventType'] == 'DEPART' \
                        and df.iloc[j+1]['timestamp'] - df.iloc[j]['timestamp'] > min_duration_seconds:
                        end_index = j
                        break 
                j += 1
            # for j in range(i+1,df.shape[0]): #find trip end
            #     # if df.iloc[j]['LocationType'] == 'STOP_POINT' and df.iloc[j]['EventType'] == 'ARRIVE' and j != i+1:
            #     if df.iloc[j]['LocationType'] == 'STOP_POINT' and df.iloc[j]['EventType'] == 'ARRIVE' and (have_tel or j != i+1):
            #         end_index = j
            #         break
            # start_index: the index of an event that is a departure event at a stop point
            # end_index: the following index of an event after start index that is an arrival event at a stop point
            # if end_index != -1 and end_index != i + 1: #there is trip end
            if end_index != -1 and (have_tel or end_index != i + 1):  # there is trip end
                count = count + 1
                end_row = df.iloc[end_index]
                start_hour_of_day = float(start_row['datetime'].hour) + float(start_row['datetime'].minute) / 60
                # print(start_row['datetime'], start_row['datetime'].hour, start_hour_of_day)
                end_hour_of_day = float(end_row['datetime'].hour) + float(end_row['datetime'].minute) / 60
                travel_time = diff_hour(start_row['timestamp'], end_row['timestamp'])
                travel_distance = 0.0

                points=[]
                points.append([start_row['Latitude'],start_row['Longitude']])
                temp_list = [start_row['LocationUUID']]
                for k in range(start_index + 1, end_index-1, 2): # add roads to travel_distance
                    temp_list.append(df.iloc[k]['LocationUUID'])

                    points = points + str_pointlist(df.iloc[k]['geometry']) # save points in trajectory
                    if isinstance(df.iloc[k]['geometry'], str):
                        linestr_from_data = str_linestr(df.iloc[k]['geometry'])
                    else:
                        linestr_from_data = df.iloc[k]['geometry']
                    line = linestr_from_data
                    length = line.length
                    travel_distance = travel_distance + length
                temp_list.append(end_row['LocationUUID'])
                trip_locations.append(temp_list)

                points.append([end_row['Latitude'],end_row['Longitude']])
                trip_points.append(points)

                start_lat = start_row['Latitude']
                start_lon = start_row['Longitude']
                end_lat = end_row['Latitude']
                end_lon = end_row['Longitude']
                day_week = float(start_row['day'])
                start_dur = diff_hour(df.iloc[start_index-1]['timestamp'],start_row['timestamp'])
                end_dur = diff_hour(end_row['timestamp'],df.iloc[end_index+1]['timestamp'])
                trip_features.append([start_hour_of_day,end_hour_of_day,travel_time,travel_distance,start_lat,start_lon,end_lat,end_lon,day_week,start_dur,end_dur])

                # trip_datetime: pandas data frame [start_datetime, end_datetime, start_arr_datetime, end_dep_datetime]
                # start_datetime: the depature time of the start SP of a trip
                # end_datetime: the arrival time of the end SP of the trip
                # start_arr_datetime: the arrival time of the start SP
                # end_dep_datetime: the depature time of the end SP
                trip_datetime_df.loc[len(trip_datetime_df.index)] = [start_row['datetime'],end_row['datetime'],df.iloc[start_index-1]['datetime'],df.iloc[end_index+1]['datetime']]

                # trip_df_idx: [start and end event indices of each trip]
                trip_df_idx.append([start_index, end_index])

    # event_loc_type_dict = {} # {LocationUUID: LocationType}
    # event_loc_adj_list = [] # LocationUUIDs of two adjacent events]
    # # file_stopp2stopp_list: [start SP1 (LocationUUID) of a trip, end SP2 (LocationUUID) of the trip, travel time between the two SPs (arrival time of SP 2 - depature time of SP 1)]
    # # We require that the start and end SPs of each trip should not have the same LocationUUID
    # file_stopp2stopp_list = []
    # for i in range(1, df.shape[0] - 1, 2):
    #     start_row = df.iloc[i]
    #     end_row = df.iloc[i + 1]
    #     if start_row['LocationUUID'] not in event_loc_type_dict.keys():
    #         event_loc_type_dict[start_row['LocationUUID']] = start_row['LocationType']
    #     if end_row['LocationUUID'] not in event_loc_type_dict.keys():
    #         event_loc_type_dict[end_row['LocationUUID']] = end_row['LocationType']
    #     event_loc_adj_list.append([start_row['LocationUUID'], end_row['LocationUUID']])
    #     if start_row['LocationType'] == 'STOP_POINT' and start_row['EventType'] == 'DEPART' and end_row[
    #         'LocationType'] == 'STOP_POINT' and end_row['EventType'] == 'ARRIVE' and start_row['LocationUUID'] != end_row['LocationUUID']:
    #         file_stopp2stopp_list.append([start_row['LocationUUID'], end_row['LocationUUID'], end_row['timestamp']-start_row['timestamp']])

    # roadseg_stopp_duration_returnlist: [id of each event, duration between this event and its subsequent event]


    # stopp_duration_dict               = dict()
    # stopp2stopp_traveltime_dict       = dict()
    # roadseg_stopp_duration_returnlist = []
    # for i in range(0, df.shape[0], 2):
    #     id       = df.iloc[i]['LocationUUID']
    #     duration = df.iloc[i+1]['timestamp'] - df.iloc[i]['timestamp']
    #     roadseg_stopp_duration_returnlist.append([id,duration])
    #     if df.iloc[i]['LocationType'] == 'STOP_POINT' and df.iloc[i]['EventType'] == 'ARRIVE':
    #         if id not in stopp_duration_dict:
    #             stopp_duration_dict[id] = [duration]
    #         else:
    #             stopp_duration_dict[id].append(duration)


    # del_stoppoint_index = []
    # for i in range(len(df.index) - 3):
    #     if df.iloc[i]['LocationType'] == 'STOP_POINT' and df.iloc[i + 1]['LocationType'] == 'STOP_POINT' and \
    #             df.iloc[i + 2]['LocationType'] == 'STOP_POINT' and df.iloc[i + 3]['LocationType'] == 'STOP_POINT' and df.iloc[i + 3]['LocationUUID'] == df.iloc[i + 2]['LocationUUID'] == df.iloc[i + 1]['LocationUUID'] == df.iloc[i]['LocationUUID']:
    #         del_stoppoint_index.append(i + 1)
    #         del_stoppoint_index.append(i + 2)
    # df = df.drop(del_stoppoint_index).reset_index(drop=True)
    # for i in range(0,df.shape[0],2):
    #     if df.iloc[i]['EventType'] != 'ARRIVE' or df.iloc[i+1]['EventType'] != 'DEPART' or df.iloc[i]['LocationUUID'] != df.iloc[i+1]['LocationUUID']:
    #         raise Exception("Dirty data")

    # stopp2stopp_traveltime_dict: {stopp: {stopp: list of travel times}}
    # df = df.drop(df[df['LocationType'] == 'ROAD_EDGE'].index).reset_index(drop=True)
    # for i in range(0, df.shape[0], 2):
    #     if i > 1:
    #         if df.iloc[i-1]['LocationUUID'] not in stopp2stopp_traveltime_dict.keys():
    #             stopp2stopp_traveltime_dict[df.iloc[i-1]['LocationUUID']] = \
    #                 {df.iloc[i]['LocationUUID']: [int(df.iloc[i]['timestamp']-df.iloc[i-1]['timestamp'])]}
    #         elif df.iloc[i]['LocationUUID'] not in stopp2stopp_traveltime_dict[df.iloc[i-1]['LocationUUID']].keys():
    #             stopp2stopp_traveltime_dict[df.iloc[i - 1]['LocationUUID']][df.iloc[i]['LocationUUID']] = \
    #                 [int(df.iloc[i]['timestamp']-df.iloc[i-1]['timestamp'])]
    #         else:
    #             temp_list = stopp2stopp_traveltime_dict[df.iloc[i - 1]['LocationUUID']][df.iloc[i]['LocationUUID']]
    #             temp_list.append(int(df.iloc[i]['timestamp']-df.iloc[i-1]['timestamp']))
    #             stopp2stopp_traveltime_dict[df.iloc[i - 1]['LocationUUID']][df.iloc[i]['LocationUUID']] = temp_list


    # start_end_stopps = [[item[0], item[-1]] for item in trip_locations]
    # for i_stopp, j_stopp in start_end_stopps:
    #     if i_stopp not in stopp2stopp_traveltime_dict or j_stopp not in stopp2stopp_traveltime_dict[i_stopp]:
    #         print("!!!!!!!!!!!!!!!! error: i_stopp not in stopp2stopp_traveltime_dict")
        # else:
        #     print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    # print("file_stopp2stopp_list, trip_locations", len(file_stopp2stopp_list), len(trip_locations))
    # stopp_dict = dict()
    # for idx, (stopp1, stopp2, travel_time) in enumerate(file_stopp2stopp_list):
    #     stopp_dict[stopp1] = 1
    #     stopp_dict[stopp2] = 1
        # if trip_locations[idx][0] != stopp1 or trip_locations[idx][-1] != stopp2:
        #     print("!!!!!!!!!!!!!!trip locations do not match file_stopp2stopp_list", stopp1, stopp2)
        # if stopp1 not in stopp_duration_dict:
        #     print("!!!!!!!!!!!!!!stopp1 not in stopp_duration_dict", stopp1)

    # stopp_duration_dict = {i_stopp: durations for i_stopp, durations in stopp_duration_dict.items() if i_stopp in stopp_dict}

    # for stopp1, durations in stopp_duration_dict.items():
        # if stopp1 not in temp_list:
        #     print("!!!!!!!!!!!!!!stopp1 not in temp_list", stopp1)

    # debug_check(stopp_duration_dict, stopp2stopp_traveltime_dict)

    return trip_features, trip_points, count, df.iloc[0]['agent_id'], trip_datetime_df, trip_locations, trip_df_idx

def debug_check(stopp_duration_dict, stopp2stopp_traveltime_dict):
    stopp_dict = dict()
    for i_stopp, j_stopp_traveltimes in stopp2stopp_traveltime_dict.items():
        stopp_dict[i_stopp] = 1
        for j_stopp, traveltimes in j_stopp_traveltimes.items():
            stopp_dict[j_stopp] = 1

    n_mismatch = 0
    for i_stopp in stopp_dict:
        if i_stopp not in stopp_duration_dict:
            # print("!!!!!!!!!!!!!!!!! stopp2stopp_traveltime_dict not in stopp_duration_dict", i_stopp)
            n_mismatch += 1

    for i_stopp in stopp_duration_dict:
        if i_stopp not in stopp_dict:
            # print("!!!!!!!!!!!!!!!!! stopp_duration_dict not in stopp2stopp_traveltime_dict", i_stopp)
            n_mismatch += 1

    print("!!!!!!!!!!!!!!!!! stopp2stopp_traveltime_dict not in stopp_duration_dict: ", n_mismatch)


def merge_dict_combine_list(dict1, dict2):
    """
    is designed to merge two dictionaries (dict1 and dict2), where the values are lists.
    For each key in either dictionary, it combines the lists from both dictionaries, creating a new list.
    If a key exists in only one dictionary, the function uses the list from that dictionary; if the key
    doesn't exist in a dictionary, it defaults to an empty list ([]). This ensures that all keys from both
    dictionaries are included in the resulting dictionary, and their corresponding lists are merged.
    """
    return {key: dict1.get(key, []) + dict2.get(key, []) for key in set(dict1) | set(dict2)}

    #{key: duration_dict.get(key, []) + agent_stopp_duration_dict.get(key, []) for key in set(duration_dict) | set(agent_stopp_duration_dict)}



import seaborn as sns
def lat_lon_time_KDE_plot(agent_id, kde_depature_ST, ST_scaler, train_data, test_data, train_dataset_folder, params_folder_name=""):

    sampled_data = kde_depature_ST.resample(5000).T
    # sampled_data = scaler.inverse_transform(ori_data)
    sampled_data = ST_scaler.inverse_transform(sampled_data)

    # Create grid points for evaluation
    # time_min, lat_min, lon_min = sampled_data.min(axis=0)
    # time_max, lat_max, lon_max = sampled_data.max(axis=0)
    # lat, lon = np.mgrid[lat_min:lat_max:100j, lon_min:lon_max:100j]
    # Evaluate KDE on a grid (for latitude and longitude in this example)
    # grid_coords = np.vstack([lat.ravel(), lon.ravel(), np.repeat(time_min, lat.size)])
    # density     = np.reshape(kde(ST_scaler.transform(grid_coords)), lat.shape)

    # Set up a 1x3 subplot grid
    fig, axes   = plt.subplots(2, 2, figsize=(24, 12))

    # Latitude vs. Longitude
    # print("sampled_data", sampled_data.shape, train_data.shape, test_data.shape)
    contour_1 = sns.kdeplot(x = sampled_data[:, 1], y = sampled_data[:, 2], cmap="viridis", shade=True, cbar=True, shade_lowest=False, ax=axes[0,0])
    if test_data is not None: 
        axes[0,0].scatter(test_data[:, 1],  test_data[:, 2],  color='red', s=10, alpha=0.5)
    axes[0,0].scatter(train_data[:, 1], train_data[:, 2], color='black', s=10, alpha=0.5)

    if test_data is not None: 
        axes[0,0].set_title('Latitude vs Longitude (Agent: {}, train: {} test: {})'.format(agent_id, train_data.shape[0], test_data.shape[0]))
    else: 
        axes[0,0].set_title('Latitude vs Longitude (Agent: {})'.format(agent_id))
    axes[0,0].set_xlabel('Latitude')
    axes[0,0].set_ylabel('Longitude')
    # fig.colorbar(contour_1, ax=axes[0,0])

    # Latitude vs. Arrival Time
    contour_2 = sns.kdeplot(x = sampled_data[:, 1], y = sampled_data[:, 0], cmap="viridis", shade=True, cbar=True, shade_lowest=False, ax=axes[1,0])
    if test_data is not None: 
        axes[1,0].scatter(test_data[:, 1],  test_data[:, 0],  color='red', s=10, alpha=0.5)
    axes[1,0].scatter(train_data[:, 1], train_data[:, 0], color='black', s=10, alpha=0.5)
    axes[1,0].set_title('Density Contour of Latitude vs Arrival Time')
    axes[1,0].set_xlabel('Latitude')
    axes[1,0].set_ylabel('Arrival Time')
    # fig.colorbar(contour_2, ax=axes[1,0])

    # Longitude vs. Arrival Time
    contour_3 = sns.kdeplot(x = sampled_data[:, 0], y = sampled_data[:, 2], cmap="viridis", shade=True, cbar=True, shade_lowest=False, ax=axes[0,1])
    if test_data is not None: 
        axes[0,1].scatter(test_data[:, 0],  test_data[:, 2],  color='red',   s=10, alpha=0.5)
    axes[0,1].scatter(train_data[:, 0], train_data[:, 2], color='black', s=10, alpha=0.5)
    axes[0,1].set_title('Density Contour of Longitude vs Arrival Time')
    axes[0,1].set_xlabel('Arrival Time')
    axes[0,1].set_ylabel('Longitude')
    # fig.colorbar(contour_3, ax=axes[0,1])

    # Adjust layout and display the plot
    plt.tight_layout()
    # plt.colorbar()
    plt.show()
    # print(root + '{}.png'.format(agent))
    create_folder_if_not_exists(train_dataset_folder + "preprocess/plots/{}KDE".format(params_folder_name))
    fig.savefig(train_dataset_folder + 'preprocess/plots/{}KDE/{}_spatiotempora_KDE.png'.format(params_folder_name, agent_id))
    print("!!!!!!!!", train_dataset_folder + 'preprocess/plots/{}KDE/{}_spatiotempora_KDE.png'.format(params_folder_name, agent_id))
    # fig.savefig(root + 'figures/{}-{}-true-anomaly.png'.format(dataset_name, agent))
    plt.close()




def read_traveldis_version2(file_name, stopp_filter_list):
    df = pd.read_parquet(file_name)
    df = df.drop_duplicates(keep='first')
    df[['LocationUUID']] = df[['LocationUUID']].astype(str)
    df['datetime'] = pd.to_datetime(df['timestamp'])
    df['timestamp'] = df['datetime'].apply(lambda x: x.timestamp())
    df = df.sort_values(by=['datetime', 'EventType'], ascending=[True, False]).reset_index(drop=True)
    df = df.drop(df[df['LocationType'] == 'ROAD_EDGE'].index).reset_index(drop=True)

    del_stoppoint_index = []
    for i in range(len(df.index) - 3):
        if df.iloc[i]['LocationType'] == 'STOP_POINT' and df.iloc[i + 1]['LocationType'] == 'STOP_POINT' and \
                df.iloc[i + 2]['LocationType'] == 'STOP_POINT' and df.iloc[i + 3]['LocationType'] == 'STOP_POINT' and df.iloc[i + 3]['LocationUUID'] == df.iloc[i + 2]['LocationUUID'] == df.iloc[i + 1]['LocationUUID'] == df.iloc[i]['LocationUUID']:
            del_stoppoint_index.append(i + 1)
            del_stoppoint_index.append(i + 2)
    df = df.drop(del_stoppoint_index).reset_index(drop=True)
    for i in range(0,df.shape[0],2):
        if df.iloc[i]['EventType'] != 'ARRIVE' or df.iloc[i+1]['EventType'] != 'DEPART' or df.iloc[i]['LocationUUID'] != df.iloc[i+1]['LocationUUID']:
            raise Exception("Dirty data")
    visit_stopp_list = []
    transition_dict = dict()
    duration_dict = dict()
    for i in range(0, df.shape[0], 2):

        if stopp_filter_list != None and (df.iloc[i]['LocationUUID'] not in stopp_filter_list or df.iloc[i-1]['LocationUUID'] not in stopp_filter_list):
            continue

        if df.iloc[i]['LocationUUID'] not in visit_stopp_list:
            visit_stopp_list.append(df.iloc[i]['LocationUUID'])
        if i > 1:
            if df.iloc[i-1]['LocationUUID'] not in transition_dict.keys():
                transition_dict[df.iloc[i-1]['LocationUUID']] = \
                    {df.iloc[i]['LocationUUID']: [int(df.iloc[i]['timestamp']-df.iloc[i-1]['timestamp'])]}
            elif df.iloc[i]['LocationUUID'] not in transition_dict[df.iloc[i-1]['LocationUUID']].keys():
                transition_dict[df.iloc[i - 1]['LocationUUID']][df.iloc[i]['LocationUUID']] = \
                    [int(df.iloc[i]['timestamp']-df.iloc[i-1]['timestamp'])]
            else:
                temp_list = transition_dict[df.iloc[i - 1]['LocationUUID']][df.iloc[i]['LocationUUID']]
                temp_list.append(int(df.iloc[i]['timestamp']-df.iloc[i-1]['timestamp']))
                transition_dict[df.iloc[i - 1]['LocationUUID']][df.iloc[i]['LocationUUID']] = temp_list
        if df.iloc[i]['LocationUUID'] not in duration_dict.keys():
            duration_dict[df.iloc[i]['LocationUUID']] = [int(df.iloc[i+1]['timestamp']-df.iloc[i]['timestamp'])]
        else:
            temp_list = duration_dict[df.iloc[i]['LocationUUID']]
            temp_list.append(int(df.iloc[i+1]['timestamp']-df.iloc[i]['timestamp']))
            duration_dict[df.iloc[i]['LocationUUID']] = temp_list
    return df.iloc[0]['agent_id'], visit_stopp_list, transition_dict, duration_dict


class GetMilpFilesTask(object):
    def __init__(self, file_name, stopp_filter_list):
        self.file_name         = file_name
        self.stopp_filter_list = stopp_filter_list

    def __call__(self):
        try:
            agent_id, visit_stopp_list, transition_dict, duration_dict = read_traveldis_version2(
                self.file_name, self.stopp_filter_list)
        except Exception as error:
            print(self.file_name)
            print(error)
            return self.file_name, None
        return self.file_name, (agent_id, visit_stopp_list, transition_dict, duration_dict)

    def __str__(self):
        return self.file_name


class GetMilpFilesTask(object):
    def __init__(self, file_name, stopp_filter_list):
        self.file_name         = file_name
        self.stopp_filter_list = stopp_filter_list

    def __call__(self):
        try:
            agent_id, visit_stopp_list, transition_dict, duration_dict = read_traveldis_version2(
                self.file_name, self.stopp_filter_list)
        except Exception as error:
            print(self.file_name)
            print(error)
            return self.file_name, None
        return self.file_name, (agent_id, visit_stopp_list, transition_dict, duration_dict)

    def __str__(self):
        return self.file_name




def get_hos_agents_info(hos_dir):
    hos_agents_dict = dict()
    agent_HOS_dict = dict()
    for root, dirs, files in os.walk(hos_dir):
        for file in files:
            if file.startswith('hos_') and file.endswith('.json'):
                with open(os.path.join(root, file), "r") as f:
                    temp_json_dict = json.load(f)
                temp_agent = temp_json_dict['events'][0]['agents'][0]
                temp_hos = file.replace('hos_', '').replace('.json', '')
                hos_agents_dict[temp_hos] = temp_agent
                agent_HOS_dict[temp_agent] = temp_json_dict
    return hos_agents_dict, agent_HOS_dict


def find_hos_agent_stopp_neighbors(test_dataset_folder, loc_coord_dict, k = 500):
    """
    This function finds k (1000 by default) nearest stopps from those in the training data for each hos stopp

    OUTPUT:
    hos_stopp_neighbors_dict: {hos stopp: a list of 300 nearest stopps}
    """

    coord_stopp_dict = dict()
    for stopp, coord_list in loc_coord_dict.items():
        if not bool_list_of_list(coord_list): # ignore road segments
            coord_stopp_dict[tuple(coord_list)] =  stopp

    coords = []
    for stopp, coord_list in loc_coord_dict.items():
        if not bool_list_of_list(coord_list): # ignore road segments
            coords.append(coord_list)
    coords_array = np.array(coords)

    # hos_stopp_dict: {hos: list of locations}
    hos_stopp_dict = get_hos_agent_stopp_dict(test_dataset_folder)

    hos_coords = []
    hos_stopps = []
    for hos, stopps in hos_stopp_dict.items():
        for stopp in stopps:
            hos_coords.append(loc_coord_dict[stopp])
            hos_stopps.append(stopp)
    hos_coords_array = np.array(hos_coords)

    coords_array     = coords_array.astype(np.float32)
    hos_coords_array = hos_coords_array.astype(np.float32)

    hos_stopp_neighbors_dict = dict()
    for hos, stopps in hos_stopp_dict.items():
        for stopp in stopps:
            hos_stopp_neighbors_dict[stopp] = []

    print("find_hos_agent_stopp_neighbors: faiss process start")
    faiss_index = build_faiss_index(coords_array)
    I, _        = find_nearest_neighbor_indices_faiss(hos_coords_array, coords_array, faiss_index, k)
    print("find_hos_agent_stopp_neighbors: faiss process end")
    for i in range(hos_coords_array.shape[0]):
        # stopp = coord_stopp_dict[tuple(hos_coords[i])]
        stopp = hos_stopps[i]
        for j in I[i,:]:
            stopp_nn = coord_stopp_dict[tuple(coords[j])]
            hos_stopp_neighbors_dict[stopp].append(stopp_nn)

    hos_stopp_neighbors_coord_dict = dict()
    for i in range(hos_coords_array.shape[0]):
        # stopp = coord_stopp_dict[tuple(hos_coords[i])]
        stopp = hos_stopps[i]
        for j in I[i,:]:
            if stopp in hos_stopp_neighbors_coord_dict:
                hos_stopp_neighbors_coord_dict[stopp].append(coords[j])
            else:
                hos_stopp_neighbors_coord_dict[stopp] = [coords[j]]

    del faiss_index

    return hos_stopp_neighbors_dict, hos_stopp_neighbors_coord_dict, hos_stopp_dict, coord_stopp_dict

# def find_hos_agent_stopp_neighbors(test_dataset_folder, loc_coord_dict, k = 500):
#     """
#     This function finds k (1000 by default) nearest stopps from those in the training data for each hos stopp

#     OUTPUT:
#     hos_stopp_neighbors_dict: {hos stopp: a list of 300 nearest stopps}
#     """

#     coord_stopp_dict = dict()
#     for stopp, coord_list in loc_coord_dict.items():
#         if not bool_list_of_list(coord_list): # ignore road segments
#             coord_stopp_dict[tuple(coord_list)] =  stopp

#     coords = []
#     for stopp, coord_list in loc_coord_dict.items():
#         if not bool_list_of_list(coord_list): # ignore road segments
#             coords.append(coord_list)
#     coords_array = np.array(coords)

#     # hos_stopp_dict: {hos: list of locations}
#     hos_stopp_dict = get_hos_agent_stopp_dict(test_dataset_folder)

#     hos_coords = []
#     for hos, stopps in hos_stopp_dict.items():
#         for stopp in stopps:
#             hos_coords.append(loc_coord_dict[stopp])
#     hos_coords_array = np.array(hos_coords)

#     coords_array     = coords_array.astype(np.float32)
#     hos_coords_array = hos_coords_array.astype(np.float32)

#     hos_stopp_neighbors_dict = dict()
#     for hos, stopps in hos_stopp_dict.items():
#         for stopp in stopps:
#             hos_stopp_neighbors_dict[stopp] = []

#     print("find_hos_agent_stopp_neighbors: faiss process start")
#     faiss_index = build_faiss_index(coords_array)
#     I, _        = find_nearest_neighbor_indices_faiss(hos_coords_array, coords_array, faiss_index, k)
#     print("find_hos_agent_stopp_neighbors: faiss process end")
#     for i in range(hos_coords_array.shape[0]):
#         stopp = coord_stopp_dict[tuple(hos_coords[i])]
#         for j in I[i,:]:
#             stopp_nn = coord_stopp_dict[tuple(coords[j])]
#             hos_stopp_neighbors_dict[stopp].append(stopp_nn)

#     hos_stopp_neighbors_coord_dict = dict()
#     for i in range(hos_coords_array.shape[0]):
#         stopp = coord_stopp_dict[tuple(hos_coords[i])]
#         for j in I[i,:]:
#             if stopp in hos_stopp_neighbors_coord_dict:
#                 hos_stopp_neighbors_coord_dict[stopp].append(coords[j])
#             else:
#                 hos_stopp_neighbors_coord_dict[stopp] = [coords[j]]

#     del faiss_index

#     return hos_stopp_neighbors_dict, hos_stopp_neighbors_coord_dict, hos_stopp_dict, coord_stopp_dict

def debug(train_dataset_folder, test_dataset_folder):
    filenames       = os.listdir(test_dataset_folder + "event_logs/") #- set(['1.parquet', '2.parquet'])
    filenames       = filenames[:5]

    for filename in filenames:
        train_fpath = train_dataset_folder + 'event_logs/' + filename
        print("\n----------------------------\n")
        print(train_fpath)
        df_train = pd.read_parquet(train_fpath)
        print(df_train.head(20))
        print("\n\n")
        test_fpath = test_dataset_folder  + 'event_logs/' + filename
        print(test_fpath)
        df_test = pd.read_parquet(test_fpath)
        print(df_test.head(20))





    # for filename in filenames:
    #     source_file = train_dataset_folder + 'event_logs/' + filename
    #     dest_file   = train_dataset_folder + 'temp/train_' + filename
    #     shutil.copy(source_file, dest_file)

    #     source_file = test_dataset_folder + 'new_event_logs/' + filename
    #     dest_file   = train_dataset_folder + 'temp/test_' + filename
    #     shutil.copy(source_file, dest_file)


    # print(filenames)
    # print(len(train_file_list), len(test_file_list))




def calc_npss(stage_2_pvalues, sliding_window_size = 8):
    """
    INPUT
    stage_2_pvalues:    NumPy array, shape: (n, ), where n is the size

    OUTPUT
    max_indices:        list of indices within the list stage_2_pvalues that relate to the p-values used to calculate the largest NPSS score.
    max_score:          largest NPSS score.
    """
    size = stage_2_pvalues.shape[0]
    if size <= 8:
        sliding_window_size = size

    max_score   = -float('inf')
    max_indices = None
    max_alpha   = None
    for alpha in np.arange(0.0001, 0.15 + 0.0001, 0.0001):
        for i in range(0, size - sliding_window_size + 1):
            temp_indicies, temp_subset_score = npss_subset_scan(stage_2_pvalues[i:i + sliding_window_size], alpha)
            temp_indices    = temp_indicies + i
            if temp_subset_score > max_score:
                max_score   = temp_subset_score
                max_indices = temp_indices
                max_alpha   = alpha
    return max_indices, max_score


def npss_subset_scan(tra_p, alpha):
    """
    tra_p: a list of p-values
    """
    sorted_tra_p        = np.sort(tra_p)
    sorted_tra_indices  = np.argsort(tra_p)
    max_score           = -float('inf')
    max_indicies        = None
    for i in range(sorted_tra_p.shape[0]):
        temp_set_p      = sorted_tra_p[:i + 1]
        temp_indices    = sorted_tra_indices[:i + 1]
        n               = temp_set_p.shape[0]
        n_alpha         = np.where(temp_set_p < alpha)[0].shape[0]
        temp_score      = n * sp.special.kl_div(n_alpha / (n * 1.0) + 0.000001, alpha)
        if temp_score > max_score:
            max_score   = temp_score
            max_indicies = temp_indices
    return max_indicies, max_score



def normalize_X_ST(X_ST, ST_Scaler = None):
    """
    Normalizes the spatiotemporal features in X_ST using MinMax scaling.

    Inputs:
    - X_ST (array-like): The spatiotemporal feature data to be normalized. This is expected to be a 2D array where
                         each row corresponds to a data point and each column corresponds to a feature.
    - ST_Scaler (MinMaxScaler, optional): An optional MinMaxScaler instance. If provided, this scaler will be used to
                                          transform the data. If not provided, a new MinMaxScaler will be instantiated,
                                          fitted on X_ST, and then used to transform X_ST.

    Outputs:
    - norm_X_ST (array-like): The normalized spatiotemporal feature data.
    - ST_Scaler (MinMaxScaler): The MinMaxScaler instance used to fit and transform the data. This can be useful for
                                inverse transformations or for applying the same scaling to other data.
    """    
    if ST_Scaler is None:
        ST_Scaler = MinMaxScaler()
        ST_Scaler.fit(X_ST)
        norm_X_ST = ST_Scaler.transform(X_ST)  # Fit and transform latitude and longitude
        # print("!!!!!!!!!!!!!!!!!********* ST_Scaler is None", ST_Scaler)
    else:
        norm_X_ST = ST_Scaler.transform(X_ST)  # Transform using the provided scaler

    return [norm_X_ST, ST_Scaler]


    # def normalize_X_ST(X_ST, lat_lon_scaler=None):
    #     """
    #     Normalizes the dataset containing latitude, longitude, and hour of day.

    #     Parameters:
    #     - X_ST: numpy array with shape (n_samples, 3), where columns are hour of day, latitude, and longitude.
    #     - lat_lon_scaler: An optional MinMaxScaler instance for latitude and longitude. If None, a new scaler is created and fitted.

    #     Returns:
    #     - A tuple containing the normalized dataset and the latitude/longitude scaler.
    #     """

    #     if lat_lon_scaler is None:
    #         lat_lon_scaler = MinMaxScaler()
    #         scaled_lat_lon = lat_lon_scaler.fit_transform(X_ST[:, 1:3])  # Fit and transform latitude and longitude
    #     else:
    #         scaled_lat_lon = lat_lon_scaler.transform(X_ST[:, 1:3])  # Transform using the provided scaler

    #     # Transform 'hour_of_day' into cyclical features
    #     hour_cos = np.cos(X_ST[:, 0] * (2. * np.pi / 24)).reshape(-1, 1)
    #     hour_sin = np.sin(X_ST[:, 0] * (2. * np.pi / 24)).reshape(-1, 1)

    #     # Concatenate scaled latitude, longitude, and transformed hour of day
    #     normalized_X_ST = np.concatenate((scaled_lat_lon, hour_cos, hour_sin), axis=1)

    #     return normalized_X_ST, lat_lon_scaler            


def find_nearest_neighbor_distances_sklearn(query_coords, historical_coords):
    """
    Finds the nearest neighbor distances for each query coordinate from a set of historical coordinates
    using scikit-learn's NearestNeighbors for efficient nearest neighbor search.

    INPUT:
    - query_coords: A (n, 2) array of query coordinates, where each row represents a coordinate pair (latitude, longitude).
    - historical_coords: A (m, 2) array of historical coordinates, where each row represents a coordinate pair.

    OUTPUT:
    - nn_distances: A list of nearest neighbor distances for each query coordinate.
    """
    # Initialize the NearestNeighbors model with the L2 (Euclidean) distance metric
    nn_model = NearestNeighbors(metric='euclidean')
    
    # Fit the model to the historical coordinates
    nn_model.fit(historical_coords)

    # Find the nearest neighbor for each query coordinate (k=1)
    distances, _ = nn_model.kneighbors(query_coords, n_neighbors=1)

    # Extract the distances as a flat list
    nn_distances = distances.flatten().tolist()

    return nn_distances

def trip_feature_extraction(parquet_fpath, train_dataset_folder, test_dataset_folder, loc_coord_dict, date_2_his_coord_set_dict = None, subsample_label = "", train_phase_boolean = True, abnormal_agent_id_list = None, anomaly_agent_index_dict = None, params = None, params_folder_name = ""):
    """
    The Python function trip_feature_extraction is designed to process trip data from parquet files, performing a comprehensive extraction of trip-related features 
    for machine learning or analytics purposes. It handles both training and testing datasets, manipulating data based on numerous input parameters and dictionaries 
    that define location coordinates, geographic data frames, and more.

    The function primarily extracts and computes various features such as travel times, Euclidean distances, departure time densities, and spatiotemporal densities 
    for each trip. It conditions its operations on whether the data is from a training or testing phase, and if necessary, combines these datasets. Additionally, the 
    function handles abnormal data points, possibly identified by an input list of abnormal agent IDs, and generates Kernel Density Estimates (KDEs) which are used for 
    further analysis of departure times and spatiotemporal information.


    INPUT
    parquet_fpath:                      The parquet file path of events of a specific agent
    train_dataset_folder:
    test_dataset_folder:
    loc_coord_dict:                     {LocationUUID: coordinates_list}, where coordinates_list is a list of start and end coordinates of this location (road segment or SP)
    gdf:
    stopp_df:
    agent_2_date_2_his_coord_set_dict:  {agent_id: {date: set of historical coordinates (tupes of (latitude, longitude))}}


    OUTPUT
    agent_id:                           the agent id associated with parquet_fpath
    trip_datetime_df:                   pandas data frame with columns: start_datetime, end_datetime, start_arr_datetime, end_dep_datetime
                                        start_datetime:     the depature time of the start SP of each trip
                                        end_datetime:       the arrival time of the end SP of each trip
                                        start_arr_datetime: the arrival time of the start SP
                                        end_dep_datetime:   the depature time of the end SP
                                        The number of rows should be the total number of trips extracted from the input (train or test) parquet files

    X:                                  n by 6 array, where n is the total number of trips in the input parquet file
                                           columns in X: travel_time (0), Euclidan distance (1), distance of start stopp to historical stopps (2),
                                                    distance of end stopp to historical stopps (3), depature time density (4), depature spatiotemporal density (5)

    agent_id_X_array:                   range (n,) array with values equal to agent ID or -1.
                                        This is used to retrieve the rows in X related to a specific agent ID and to its trips in the training parquet file.
                                        Note that, a agent may include both trips from the training and test parquet files.
                                        for rows in X related to the trips in the test parquet file, we set the values to -1.
                                        It is used to do npss scanning on the training trips, so we ignore test trips by using the value "-1"


    [kde_depature_train_data,
    kde_ST_train_data]:
        kde_depature_train_data:        An array (n,0) with start hours of day (float) in the trips
        kde_ST_train_data:              An array of (n+m, 3), columns: start hour of day (float), latitude, longitude
                                        m is the number of faked observations of start hours of day (near hours 0 and 24)
    his_coord_set:                      set of tuples of historical coordinates (latitude, longitude) for all the historical days
    date_2_his_coord_set_dict:          {date: set of tuples of historical coordinates (latitude, longitude)}
    bool_combine_train_test:            True: combine training and test parquet files for each agent during the training phase; False: only use training parquet files
                                        the defult is True for the training phase. It's boolean value is ignored of the current phase is the test phase.



    file_type_dict:     (event level) {LocationUUID: LocationType}
    file_adj_list:      (event level) [LocationUUIDs of two adjacent events]
    file_stopp2stopp_list:       (event level) [start SP1 (LocationUUID) of a trip, end SP2 (LocationUUID) of the trip, travel time between the two SPs (arrival time of SP 2 - depature time of SP 1)]
                                        We require that the start and end SPs of each trip should not have  the same LocationUUID
    stopp_duration_dict: {stopp ID: list of durations}
    roadseg_stopp_duration_returnlist: (event level) [id of each event, duration between this event and its subsequent event]
    trip_locations:     (trip level) [list of locations (LocationUUID) for each trip]

    """

    # if params is not None: 
    #     params_folder_name = "bw-{}-K-{}/".format(params.bw_ST_KDE, params.K_ad)
    # else:
    #     params_folder_name = ""
    agent_id = np.int64(parquet_fpath.split('/')[-1].split('.')[0]) # agent ID should be integer format by default
    have_tel = True
    # print(parquet_fpath)

    # bool_test_phase = False
    # if "new_event_logs" not in parquet_fpath: # all test parquet files should be stored within the folder: "new_event_logs"
    #     bool_test_phase = True # if it is False, it means it is the training phase
    # print("!!!!!!!!!!!!!!!!!!!!!! step 1")
    df                      = pd.read_parquet(parquet_fpath)
    if df['agent_id'].nunique() > 1:
        df                  = df.loc[df['agent_id'] == agent_id]
    df = df.drop_duplicates(keep='first')
    df = df.drop_duplicates(subset=["time_start"], keep='first')
    df['arrival_datetime'] = pd.to_datetime(df['time_start'])
    df['depart_datetime'] = pd.to_datetime(df['time_stop'])
    df['arrival_timestamp'] = df['arrival_datetime'].apply(lambda x: x.timestamp())
    df['depart_timestamp'] = df['depart_datetime'].apply(lambda x: x.timestamp())

    """
    trip_features:      list (trip level)
                        [start hour of day (float),end hour of day (float), travel_time in hours (float),
                         travel_distance,start_lat,start_lon,end_lat,end_lon,day_week,
                         start_dur (hours (float)),end_dur (hours (float))]
    trip_datetime_df:   pandas data frame with columns: start_datetime, end_datetime, start_arr_datetime, end_dep_datetime
                        start_datetime: the depature time of the start SP of each trip
                        end_datetime: the arrival time of the end SP of each trip
                        start_arr_datetime: the arrival time of the start SP
                        end_dep_datetime: the depature time of the end SP
    """
    trip_features, trip_points, trip_count, _, trip_datetime_df, trip_locations, trip_df_idx = read_traveldis(df, have_tel)
    n_trips_without_add = len(trip_features)

    if train_phase_boolean == True and params.bool_combine_train_test == True: # if it is the training phase and it needs to combine train and test trajectories
        add_parquet_fpath = test_dataset_folder + 'event_logs/{}.parquet'.format(agent_id)
    else:
        add_parquet_fpath = None



    if add_parquet_fpath is not None and os.path.exists(add_parquet_fpath):
        df_add                      = pd.read_parquet(add_parquet_fpath)
        if df_add['agent_id'].nunique() > 1:
            df_add  = df_add.loc[df_add['agent_id'] == agent_id]
        df_add                      = df_add.drop_duplicates(keep='first')
        df_add                      = df_add.drop_duplicates(subset=["time_start"],keep='first')
        df_add['arrival_datetime']  = pd.to_datetime(df_add['time_start'])
        df_add['depart_datetime']   = pd.to_datetime(df_add['time_stop'])
        df_add['arrival_timestamp'] = df_add['arrival_datetime'].apply(lambda x: x.timestamp())
        df_add['depart_timestamp']  = df_add['depart_datetime'].apply(lambda x: x.timestamp())


        add_trip_features, add_trip_points, add_trip_count, _, add_trip_datetime_df, \
            add_trip_locations, add_trip_df_idx = read_traveldis(df_add, have_tel)
        # print("!!!!!!!!!!!!!!! train and test trips combined: ", add_parquet_fpath)
        # print("trip_features", len(trip_features), len(test_trip_features))
        trip_features.extend(add_trip_features)
        # print("trip_points", len(trip_points), len(test_trip_points))
        trip_points.extend(add_trip_points)
        # print("trip_locations", len(trip_locations), len(test_trip_locations))
        trip_locations.extend(add_trip_locations)
        trip_df_idx.extend(add_trip_df_idx)
        trip_datetime_df = pd.concat([trip_datetime_df, add_trip_datetime_df], axis = 0)

    n_trips_after_add = len(trip_features)

    trip_datetime_df['weekday']       = trip_datetime_df['start_datetime'].apply(lambda x: x.weekday())
    trip_datetime_df['dayofweek']     = trip_datetime_df['weekday'].apply(lambda x: 0 if x<4.5 else 1)
    trip_datetime_df['start_time']    = trip_datetime_df['start_datetime'].apply(lambda x: x.time())
    trip_datetime_df['end_time']      = trip_datetime_df['end_datetime'].apply(lambda x: x.time())
    trip_datetime_df['start_date']    = trip_datetime_df['start_datetime'].apply(lambda x: x.date())
    trip_datetime_df['end_weekday']   = trip_datetime_df['end_datetime'].apply(lambda x: x.weekday())
    trip_datetime_df['end_dayofweek'] = trip_datetime_df['end_weekday'].apply(lambda x: 0 if x<4.5 else 1)
    trip_start_end_datetimes_df       = trip_datetime_df[['start_datetime', 'end_datetime']]

    # columns in raw_X: [start_hour_of_day (0), end_hour_of_day (1),travel_time (2), travel_distance (3), start_lat (4), start_lon (5),
    #                end_lat (6), end_lon (7), day_week (8), start_dur (9), end_dur (10)]
    # columns in X: travel_time (0), Euclidan distance (1), distance of start stopp to historical stopps (2),
    #               distance of end stopp to historical stopps (3), depature time density (4),
    #               depature spatiotemporal density (5)
    n_trips                          = len(trip_features) # list of trip-level feature vectors
    raw_X                            = np.array(trip_features).astype(float)
    X                                = np.zeros((n_trips, 9))
    # print(trip_features)
    # print("X[:, 0], raw_X[:, 2]", X, raw_X)
    X[:, 0]                          = raw_X[:, 2]  # travel_times of the trips (third column in raw_X)
    X[:, 1]                          = np.sqrt((raw_X[:, 4] - raw_X[:, 6]) ** 2 + (raw_X[:, 5] - raw_X[:, 7]) ** 2) # Euclidean disance between start and end stop points

    date_2_stopp_coords_dict  = dict()
    trip_datetime_coords_df = trip_datetime_df.copy()
    trip_datetime_coords_df['start_lat'] =  raw_X[:, 4] # start latitude
    trip_datetime_coords_df['start_lon'] =  raw_X[:, 5] # start longitude
    trips_date_group        = trip_datetime_coords_df.groupby(by='start_date')
    for i_date, i_group in trips_date_group:
        temp_coords_list = i_group[['start_lat', 'start_lon']].values
        date_2_stopp_coords_dict[i_date] = temp_coords_list  

    trip_start_date_2_travel_distance = dict()
    trip_start_date_2_indices = dict()
    # day_2_his_coord_set_dict: {date: set of coordinates (road segment centers or stop points) visited by the agent in the date}
    if train_phase_boolean == True: # if it is the training phase, date_2_his_coord_set_dict will be an empty set in the beginning
        date_2_his_coord_set_dict = dict()
        trips_date_group  = trip_datetime_df.groupby(by='start_date')
        for i_date, i_group in trips_date_group:
            date_trip_coord_set  = set()
            date_trip_indices    = list(i_group.index)
            trip_start_date_2_travel_distance[i_date] = 0
            trip_start_date_2_indices[i_date] = date_trip_indices
            for trip_idx in date_trip_indices: # if it is a list, we do not always use "_list" at the end if the variable name is plural
                trip_start_date_2_travel_distance[i_date] += trip_features[trip_idx][2] # travel time 
                temp_trip_coord_list = []
                # trip_locations: (trip level) [list of locations (LocationUUID) for each trip]
                # temp_trip_locations: list of locations (road segments or stop points) of the trip indexed by trip_idx
                temp_trip_locations = trip_locations[trip_idx] # row i in trip_datetime_df and the i-th item in trip_location relate to the same agent.  [to do: Need Kai to check].
                for i in range(len(temp_trip_locations)):
                    temp_coord = loc_coord_dict[temp_trip_locations[i]]
                    
                    if isinstance(temp_coord[0],list): # it means this location is a road segment. so we skip this. 
                        # there are multiple coordinates within temp_coord and we return the middle one
                        # continue
                        temp_coord = temp_coord[int(len(temp_coord)/2)]
                    # temp_coord_tuple = tuple(temp_coord)
                    temp_trip_coord_list.append(temp_coord) 
                temp_trip_coord_set = set([tuple(coord) for coord in temp_trip_coord_list]) # the elements within a set must be tuples but not lists
                date_trip_coord_set = date_trip_coord_set | temp_trip_coord_set # A trip may have multile locations.
            if i_date not in date_2_his_coord_set_dict:
                date_2_his_coord_set_dict[i_date] = date_trip_coord_set # set of unique locations (stopps, or middle points of road segments)
            else:
                date_2_his_coord_set_dict[i_date] = date_2_his_coord_set_dict[i_date] | date_trip_coord_set # this may occur if day_2_coord_set_dict was generated in the training phase

        his_coord_set = set()
        for i_day, i_coord_set in date_2_his_coord_set_dict.items():
            his_coord_set = his_coord_set.union(i_coord_set)

    else: # if it is the test phase and agent_2_date_2_his_coord_set_dict exists, preload it
        trips_date_group  = trip_datetime_df.groupby(by='start_date')
        for i_date, i_group in trips_date_group:
            date_trip_indices = list(i_group.index)
            trip_start_date_2_travel_distance[i_date] = 0
            trip_start_date_2_indices[i_date] = date_trip_indices
            for trip_idx in date_trip_indices: # if it is a list, we do not always use "_list" at the end if the variable name is plural
                trip_start_date_2_travel_distance[i_date] += trip_features[trip_idx][2] # travel time
 
        # date_2_his_coord_set_dict = agent_2_date_2_his_coord_set_dict[agent_id]
        his_coord_set = None

    # print("!!!!!!!!!!!!!!!!!!!!!! step 2")
    if n_trips_after_add == n_trips_without_add:
        agent_id_X_array = np.ones((n_trips_without_add, )) * agent_id
    else:
        agent_id_X_array = np.concatenate((np.ones((n_trips_without_add, )) * agent_id, np.ones((n_trips_after_add - n_trips_without_add, )) * -1 * agent_id), axis = 0)
    
    if params.raw_X_bool == True: # use raw features instead of the nine features extracted above
        X = raw_X
        kde_depature_ST_train_data = None
        return agent_id, trip_datetime_df, X, agent_id_X_array, his_coord_set, date_2_his_coord_set_dict, trip_start_end_datetimes_df, trip_df_idx, kde_depature_ST_train_data, date_2_stopp_coords_dict, raw_X

    date_2_start_stopp_coords_dict  = {}
    date_2_end_stopp_coords_dict    = {}
    # trip_locations: (trip level) [list of locations (LocationUUID) for each trip]
    # The keys (dates) are added based on the order of the date from the old to recent. The order in which the keys are added will be preserved when iterating over the dictionary using methods, such as items(), keys() and values()
    for trip_idx, locations in enumerate(trip_locations):
        trip_stopps                 = [locations[0],locations[-1]] # start and end stopps of the trip
        trip_start_stopp            = trip_stopps[0]
        trip_end_stopp              = trip_stopps[-1]
        trip_start_stopp_coord      = loc_coord_dict[trip_start_stopp] #
        trip_end_stopp_coord        = loc_coord_dict[trip_end_stopp]
        trip_date                   = trip_datetime_df.iloc[trip_idx]['start_date']
        if trip_date not in date_2_start_stopp_coords_dict.keys():
            date_2_start_stopp_coords_dict[trip_date]  = [trip_start_stopp_coord]
            date_2_end_stopp_coords_dict[trip_date]    = [trip_end_stopp_coord]
        else:
            date_2_start_stopp_coords_dict[trip_date].append(trip_start_stopp_coord)
            date_2_end_stopp_coords_dict[trip_date].append(trip_end_stopp_coord)


    # print("!!!!!!!!!!!!!!!!!!!!!! step 3")
    start_stopps_nn_distance_column = []
    end_stopps_nn_distance_column   = []
    for i_date, _ in date_2_start_stopp_coords_dict.items():
        i_his_coord_set = set()
        for j_date, j_coord_set in date_2_his_coord_set_dict.items():
            if j_date != i_date:
                i_his_coord_set = i_his_coord_set.union(j_coord_set)

        # print("!!!!!! trip_feature_extraction: faiss process start start")
        faiss_index                  = build_faiss_index(np.array(list(i_his_coord_set)))
        start_stopps_nn_dist_list, _ = find_nearest_neighbor_distances_faiss(np.array(date_2_start_stopp_coords_dict[i_date]), np.array(list(i_his_coord_set)), faiss_index)
        end_stopps_nn_dist_list, _   = find_nearest_neighbor_distances_faiss(np.array(date_2_end_stopp_coords_dict[i_date]),   np.array(list(i_his_coord_set)), faiss_index)
        del faiss_index
        # print("!!!!!! trip_feature_extraction: faiss process start end")
        
        # i_his_coord_set = np.array(list(i_his_coord_set))  # Convert the set of historical coordinates to a NumPy array
        # Find nearest neighbor distances for start stop points
        # if train_phase_boolean == False: 
        #     print("i_date, date_2_start_stopp_coords_dict, i_his_coord_set", len(date_2_start_stopp_coords_dict[i_date]), len(list(i_his_coord_set)))
        # start_stopps_nn_dist_list = find_nearest_neighbor_distances_sklearn(np.array(date_2_start_stopp_coords_dict[i_date]), np.array(list(i_his_coord_set)))
        # # Find nearest neighbor distances for end stop points
        # end_stopps_nn_dist_list = find_nearest_neighbor_distances_sklearn(np.array(date_2_end_stopp_coords_dict[i_date]), np.array(list(i_his_coord_set)))

        start_stopps_nn_distance_column.extend(start_stopps_nn_dist_list)
        end_stopps_nn_distance_column.extend(end_stopps_nn_dist_list)

    # print("!!!!!!!!!!!!!!!!!!!!!! step 4")
    X[:, 2] = start_stopps_nn_distance_column
    X[:, 3] = end_stopps_nn_distance_column
    kde_depature_train_data     = None
    kde_depature_hour           = None
    kde_depature_ST_train_data  = None
    kde_depature_ST             = None
    ST_Scaler                   = None
    date_travel_time_kde        = None

    if train_phase_boolean == False:
        # print("!!!!!!!!!!!************** ", train_dataset_folder + "preprocess/{}KDE/{}{}_KDEs.dill".format(params_folder_name, subsample_label, agent_id))
        kdes_file_path = train_dataset_folder + "preprocess/{}KDE/{}{}_KDEs.dill".format(params_folder_name, subsample_label, agent_id)
        [kde_depature_hour, kde_depature_ST, date_travel_time_kde, ST_Scaler, check_kdes_file_path] = load_from_dill(kdes_file_path)
        if check_kdes_file_path != kdes_file_path:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! check_kdes_file_path != kdes_file_path !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # kde_depature_hour       = agent_2_KDEs_dict[agent_id][0]
    else:
        if n_trips > 10:
            kde_depature_train_data = raw_X[:, 0] # start hour of day within [0, 24] (float)  range(n,)
            kde_depature_hour       = gaussian_kde(kde_depature_train_data.reshape(1, -1), bw_method=0.1) # reshape to (1,n)
    if kde_depature_hour is not None:
        depature_hour_densities = kde_depature_hour(raw_X[:, 0])
        # mean_density            = np.mean(depature_hour_densities)
        # depature_hour_densities[depature_hour_densities > mean_density] = mean_density
        X[:, 4]                 = depature_hour_densities
    else:
        X[:, 4]                 = np.ones((n_trips, )) * 1

    X_ST_test                   = raw_X[:, [0, 4, 5]]

    # print("!!!!!!!!!!!!!!!!!!!!!! ST_Scaler", ST_Scaler.data_max_, ST_Scaler.data_min_)
    if train_phase_boolean == True:
        # train phase
        if n_trips > 10:
            # In the following lines, we address the issue of spatiotemporal density for data points near 0 or 24.
            # The data points near 24 should also be near 0 and vice versa.
            # For the trips with arrival hour of day greater than 22, we dupliate them and change their arrival hours to -1 (24 - ...)

            if params is not None and params.ab_trip_removal_ST_KDE_bool == True and agent_id in abnormal_agent_id_list: 
                trip_df_idx_array     = np.array(trip_df_idx)
                y                     = np.zeros((n_trips,))
                # print("abnormal agent id:", ab_agent_id)
                [a, b]                = anomaly_agent_index_dict[agent_id]
                # Determine which trips overlap with the anomaly range
                starts_within_range   = trip_df_idx_array[n_trips_without_add: n_trips_after_add, 0] <= b
                ends_within_range     = trip_df_idx_array[n_trips_without_add: n_trips_after_add, 1] >= a
                overlapping_indices   = np.where(starts_within_range & ends_within_range)[0]
                # test_trip_indices = range(n_trips_without_add, n_trips_after_add)
                # Update y for the overlapping trips of the current ab_agent_id
                y[n_trips_without_add + overlapping_indices] = 1
                # [a, b]                      = anomaly_agent_index_dict[agent_id]
                # starts_within_range         = trip_df_idx_array[:, 0] > b
                # ends_within_range           = trip_df_idx_array[:, 1] < a
                # normal_event_indices        = np.where(starts_within_range | ends_within_range)[0]
                # print("normal_event_indices", normal_event_indices)
                # print("raw_X.shape, y.shape", raw_X.shape, y.shape)

                selected_rows               = raw_X[y == 0]
                X_ST_train                  = selected_rows[:, [0, 4, 5]]
                # print("X_ST_train.shape, # of abnormal trips", X_ST_train.shape, np.sum(y))
            else: 
                X_ST_train              = raw_X[:, [0, 4, 5]]

            # norm_X_ST_train, lat_log_scaler = normalize_X_ST(X_ST_train)

            indices                     = X_ST_train[:,0] > 22
            X_ST_add_1                  = X_ST_train[indices,:]
            X_ST_add_1[:,0]             = -1 * (24 - X_ST_add_1[:,0])
            # For the trips with arrival hour of day less than 2, we dupliate them and change their arrival hours to ... + 24
            indices                     = X_ST_train[:,0] < 2
            X_ST_add_2                  = X_ST_train[indices,:]
            X_ST_add_2[:,0]             = X_ST_add_2[:,0] + 24

            X_ST_train_aug              = copy.deepcopy(X_ST_train)
            if X_ST_add_1.shape[0] > 0:
                X_ST_train_aug = np.concatenate((X_ST_train_aug, X_ST_add_1), axis = 0)
            if X_ST_add_2.shape[0] > 0:
                X_ST_train_aug = np.concatenate((X_ST_train_aug,  X_ST_add_2), axis = 0)


            noise                       = np.random.normal(0, 0.0000001, size=X_ST_train_aug.shape)
            X_ST_train_aug              = X_ST_train_aug + noise
            [norm_X_ST_train_aug, ST_Scaler] = normalize_X_ST(X_ST_train_aug)
            if params is not None: 
                kde_depature_ST         = gaussian_kde(norm_X_ST_train_aug.T, bw_method=params.bw_ST_KDE)  # the input of gaussian_kde is a (d, n) array, where d is the number of features
            else: 
                kde_depature_ST         = gaussian_kde(norm_X_ST_train_aug.T)  # the input of gaussian_kde is a (d, n) array, where d is the number of features
                # kde_depature_ST.set_bandwidth(bw_method=kde_depature_ST.factor / 3.)
                
            kde_depature_ST_train_data  = X_ST_train # X_ST_train
        else:
            kde_depature_ST             = None
            ST_Scaler                   = None 
            kde_depature_ST_train_data  = X_ST_test
    else: 
        kde_depature_ST_train_data = X_ST_test

    # print("!!!!!!!!!!!!!!!!!!!!!! step 6")

    if kde_depature_ST is not None:
        norm_X_ST_test,_    = normalize_X_ST(X_ST_test, ST_Scaler)
        log_ST_densities    = np.log(kde_depature_ST(norm_X_ST_test.T) + 0.00001)
        # print("log_ST_densities", log_ST_densities)
        # mean_log_density        = np.mean(log_ST_densities)
        # log_ST_densities[log_ST_densities > mean_log_density] = mean_log_density
        X[:, 5]             = log_ST_densities
    else:
        X[:, 5]             = np.log(np.ones((n_trips, )) * 1) # Feng: debug: need to check if 0.5 is agood density value

    X[:,6]                  = raw_X[:,9]
    X[:,7]                  = raw_X[:,10]

    if train_phase_boolean == True:
        date_travel_time_list = [i_travel_time for i_date, i_travel_time in trip_start_date_2_travel_distance.items()]
        date_travel_time_kde  = gaussian_kde(np.array(date_travel_time_list).reshape(1, -1))
        # we save the KDEs only if it is the training phase
        create_folder_if_not_exists(train_dataset_folder + "preprocess/{}KDE".format(params_folder_name))
        kdes_file_path = train_dataset_folder + "preprocess/{}KDE/{}{}_KDEs.dill".format(params_folder_name, subsample_label, agent_id)
        save_to_dill(kdes_file_path, [kde_depature_hour, kde_depature_ST, date_travel_time_kde, ST_Scaler, kdes_file_path])
        # print("!!!!!!! ************", train_dataset_folder + "preprocess/{}KDE/{}{}_KDEs.dill".format(params_folder_name, subsample_label, agent_id))
        #  /home/jxl220096/data/hay/haystac_trial1/fix_Novateur_TA1_Trial_Train_Submission/preprocess/bw-0.001-K-10-25-rawX-0-STKDE-1/KDE/2k_9067932_KDEs.dill
        #  /home/jxl220096/data/hay/haystac_trial1/fix_Novateur_TA1_Trial_Train_Submission/preprocess/bw-0.001-K-10-25-rawX-0-STKDE-1/KDE/2k_11600183_KDEs.dill
    for i_date, i_travel_dist in trip_start_date_2_travel_distance.items():
        # indices = trip_datetime_df.index[trip_datetime_df['start_date'] == i_date].tolist()
        X[trip_start_date_2_indices[i_date],8] = date_travel_time_kde(i_travel_dist)


    if params.agg_bool == True:
        X = np.concatenate((raw_X, X[:,[4,5,8]]), axis = 1)

    return agent_id, trip_datetime_df, X, agent_id_X_array, his_coord_set, date_2_his_coord_set_dict, trip_start_end_datetimes_df, trip_df_idx, kde_depature_ST_train_data, date_2_stopp_coords_dict, raw_X
    # return agent_id, trip_datetime_df, event_loc_type_dict, event_loc_adj_list, stopp_duration_dict, roadseg_stopp_duration_returnlist, stopp2stop_traveltime_dict, trip_locations, X, [kde_depature_train_data, kde_ST_train_data], his_coord_set, date_2_his_coord_set_dict



    # if os.path.exists(kdes_file_path):
    #     # Delete the file
    #     os.remove(kdes_file_path)
    # print("!!!!!!!!!!!!!!!!!!!!!! ST_Scaler", ST_Scaler)
    # print("!!!!!!!!!!!!!!!!!!!!!! ST_Scaler", ST_Scaler.data_max_, ST_Scaler.data_min_)
    # [kde_depature_hour, kde_depature_ST, date_travel_time_kde, ST_Scaler, check_kdes_file_path] = load_from_dill(kdes_file_path)
    # save_to_pickle(train_dataset_folder + "preprocess/{}KDE_ST_scaler.pkl".format(params_folder_name), ST_Scaler)
    # ST_Scaler = load_from_pickle(train_dataset_folder + "preprocess/{}KDE_ST_scaler.pkl".format(params_folder_name))
    # print("!!!!!!!!!!!!!!!!!!!!!! ST_Scaler", ST_Scaler.data_max_, ST_Scaler.data_min_)
    # print("!!!!!!!!!X_ST_test[:,0]", X_ST_test[:,0])
    # norm_X_ST_test,_    = normalize_X_ST(X_ST_test, ST_Scaler)
    # print("!!!!!!!!!norm_X_ST_test[:,0]", norm_X_ST_test[:,0])
    # log_ST_densities    = np.log(kde_depature_ST(norm_X_ST_test.T) + 0.00001)
    # print("!!!!!!!!!log_ST_densities", log_ST_densities)        

    # if "new_event_logs" not in parquet_fpath: # meaning that this is a parquet file from the train folder:
    #     add_parquet_fname = test_dataset_folder + 'new_event_logs/{}.parquet'.format(agent_id)
    # else: # meaning that this is a parquet file from the train folder
    #     add_parquet_fname = train_dataset_folder + 'event_logs/{}.parquet'.format(agent_id)


    # kde_ST_train_data           = X_ST_train
    # lat_lon_time_KDE_plot(agent_id, kde_depature_ST, X_ST_train, X_ST_test, test_dataset_folder)

    # for idx, item in enumerate(date_2_end_stopp_coords_dict.items()):
    #     if i < 5:
    #         print("date_2_end_stopp_coords_dict: ", item)

    # for idx, item in enumerate(date_2_his_coord_set_dict.items()):
    #     if idx < 5:
    #         print("date_2_his_coord_set_dict: ", item)

    # return


def gen_anomaly_agent_index_dict(test_dataset_dir, bool_trial_2):
   anomaly_agent_index_dict = dict()
   if bool_trial_2:
       gts_folder = os.path.join(test_dataset_dir, "gts")
       for root, ds, fs in os.walk(gts_folder):
           for f in sorted(fs):
               if f.endswith('.parquet'):
                   temp_parquet_file = os.path.join(root, f)
                   temp_parquet_content = pd.read_parquet(temp_parquet_file)
                   temp_agent = np.int64(temp_parquet_content.iloc[0]['agent'])
                   sorted_abnormal_parquet = temp_parquet_content.sort_values(by=['timestamp'],ascending=[True]).reset_index(drop=True)
                   start_timestamp = sorted_abnormal_parquet.iloc[0]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                   end_timestamp = sorted_abnormal_parquet.iloc[-1]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                   # abnormal_dict[temp_abnormal_agent] = [start_timestamp, end_timestamp]
                   temp_trajectory_df = pd.read_parquet(os.path.join(test_dataset_dir, str(temp_agent) + '.parquet')).reset_index()
                   if temp_trajectory_df['agent_id'].nunique() > 1:
                       temp_trajectory_df = temp_trajectory_df.loc[temp_trajectory_df['agent_id'] == temp_agent]
                   temp_trajectory_df = temp_trajectory_df.drop_duplicates(keep='first')
                   temp_trajectory_df = temp_trajectory_df.drop_duplicates(subset=["time_start"], keep='first')
                   temp_trajectory_df = temp_trajectory_df.sort_values(by=['time_start'], ascending=[True]).reset_index()

                   indices = temp_trajectory_df[((start_timestamp <= temp_trajectory_df['time_start']) & (
                               temp_trajectory_df['time_start'] <= end_timestamp)) | ((start_timestamp <= temp_trajectory_df['time_stop']) & (
                               temp_trajectory_df['time_stop'] <= end_timestamp))].index.tolist()
                   if len(indices) == 0:
                       temp_before_df = temp_trajectory_df[start_timestamp >= temp_trajectory_df['time_stop']]
                       temp_after_df = temp_trajectory_df[end_timestamp <= temp_trajectory_df['time_start']]
                       if len(temp_before_df) == 0: # anomaly time range before trajectory
                           indices = [0, 0]
                       elif len(temp_after_df) == 0:# anomaly time range after trajectory
                           indices = [len(temp_after_df) - 1, len(temp_after_df) - 1]
                       else:
                           temp_start_index = \
                           temp_trajectory_df[start_timestamp >= temp_trajectory_df['time_stop']].index.tolist()[
                               -1]
                           temp_end_index = \
                           temp_trajectory_df[end_timestamp <= temp_trajectory_df['time_start']].index.tolist()[0]
                           indices = [temp_start_index, temp_end_index]
                   anomaly_agent_index_dict[temp_agent] = [indices[0], indices[-1]]
   else:
       for filename in os.listdir(test_dataset_dir + 'truth/gts'):
           if filename.startswith('agent='):
               temp_abnormal_agent = np.int64(filename.replace('agent=', ''))
               for parquet_filename in os.listdir(test_dataset_dir + 'truth/gts/' + filename):
                   if parquet_filename.endswith('.parquet'):
                       abnormal_parquet = pd.read_parquet(
                           test_dataset_dir + 'truth/gts/' + filename + '/' + parquet_filename)
               sorted_abnormal_parquet = abnormal_parquet.sort_values(by=['timestamp'], ascending=[True]).reset_index(
                   drop=True)
               start_timestamp = sorted_abnormal_parquet.iloc[0]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
               end_timestamp = sorted_abnormal_parquet.iloc[-1]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
               # abnormal_dict[temp_abnormal_agent] = [start_timestamp, end_timestamp]
               temp_trajectory_parquet = pd.read_parquet(os.path.join(test_dataset_dir, 'event_logs', str(temp_abnormal_agent) + '.parquet')).reset_index()
               indices = temp_trajectory_parquet[(start_timestamp <= temp_trajectory_parquet['timestamp']) & (temp_trajectory_parquet['timestamp'] <= end_timestamp)].index.tolist()
               if len(indices) == 0:
                   temp_start_index = temp_trajectory_parquet[start_timestamp >= temp_trajectory_parquet['timestamp']].index.tolist()[-1]
                   temp_end_index = temp_trajectory_parquet[end_timestamp <= temp_trajectory_parquet['timestamp']].index.tolist()[0]
                   indices = [temp_start_index, temp_end_index]
               anomaly_agent_index_dict[temp_abnormal_agent] = [indices[0], indices[-1]]
   return anomaly_agent_index_dict



def detection_train(train_dataset_folder, test_dataset_folder, parquet_fpath_list, abnormal_agent_id_list = None, anomaly_agent_index_dict = None, ad_label = "ensemble", subsample_label = "", trip_extraction_boolean = True, params = None, params_folder_name = ""):
    """
    In this version, the test NAT trajectories are not used in the training phase. We may add them to the training phase in the future.
    """

    # delete_files_in_folder(train_dataset_folder + "preprocess/{}KDE".format(params_folder_name))

    time1 = time.time()
    # merge_train_test_stopp_parquet(train_dataset_folder, test_dataset_folder)

    # hos_agents_dict = get_hos_agents(os.path.join(test_dataset_folder, 'HOS'))
    # agents_hos = [hos_agent for hos, hos_agent in hos_agents_dict.items()]
    # [205273, 59404, 165571, 154906, 129440, 32611, 36135, 148574, 182729, 200034, 25588, 151668, 12087, 166383, 73399, 168136, 180588, 71625, 7286, 82977, 178325, 149736, 42936, 131191, 123119, 196069, 110431, 100482, 99275, 68206]

    print("loading loc_coord_dict.pickle")
    if not os.path.exists(train_dataset_folder + 'preprocess/loc_coord_dict.pickle'):
        loc_coord_dict = new_create_location_coordinates(train_dataset_folder, test_dataset_folder)
    else:
        # loc_coord_dict = load_from_pickle(train_dataset_folder + 'preprocess/loc_coord_dict.pickle')
        loc_coord_dict = load_from_pickle(train_dataset_folder + 'preprocess/loc_coord_dict.pickle')

    # parquet_fpath = "/home/jxl220096/data/hay/haystac_trial1/fix_Novateur_TA1_Trial_Train_Submission/event_logs/13215469.parquet"
    # print(parquet_fpath)
    # agent_id, trip_datetime_df, X, agent_id_X_array, his_coord_set, date_2_his_coord_set_dict, trip_start_end_datetimes_df, \
    #     trip_df_idx, kde_depature_ST_train_data, date_2_stopp_coords_dict, raw_X = trip_feature_extraction(parquet_fpath, train_dataset_folder, test_dataset_folder, loc_coord_dict, gdf, stopp_df, train_phase_boolean = True, abnormal_agent_id_list = abnormal_agent_id_list, anomaly_agent_index_dict = anomaly_agent_index_dict, params = params, params_folder_name = params_folder_name)
    # print(raw_X)
    # return
    # for parquet_fpath in parquet_fpath_list[:5]:
    #     print(parquet_fpath)
    #     trip_feature_extraction(parquet_fpath, train_dataset_folder, test_dataset_folder, loc_coord_dict, gdf, stopp_df, None, subsample_label, train_phase_boolean = True, abnormal_agent_id_list = abnormal_agent_id_list, anomaly_agent_index_dict = anomaly_agent_index_dict, params = params, params_folder_name = params_folder_name)
    # return
    # print("file_list", len(parquet_fpath_list))

    """
    Step 1: Extract trips and their features from the trajectories stored in the parquet files in the file_list
    """
    
    # trip_extraction_boolean = False
    if trip_extraction_boolean:
        agent_id_list                       = []
        X                                   = None
        raw_X                               = None 
        # agent_2_KDEs_train_data_dict        = dict()
        agent_2_his_coord_set_dict          = dict()
        agent_2_date_2_his_coord_set_dict   = dict()
        id_agent_X                          = None
        trip_df_idx                         = None
        agent_2_date_2_stopp_coords_dict    = dict()
        # print("filtered file_list size based on feature extraction done previously: {}".format(len(file_list)))
        n_agents        = len(parquet_fpath_list)
        if(n_agents == 0):
            return
        tasks           = multiprocessing.Queue()
        results         = multiprocessing.Queue()
        num_consumers   = 32

        # if the number of threads is greater than the number of agents (n_agents). We reduce the numberof threads to the half of n_agents
        if num_consumers > n_agents:
            num_consumers = int(np.ceil(n_agents * 0.5))

        K               = int(np.ceil(n_agents / num_consumers))
        consumers       = [Consumer(tasks, results)
                            for _ in range(num_consumers)]
        for w in consumers:
            w.start()

        chunk_parquet_fpath_list = chunk_file_paths(parquet_fpath_list, K)
        num_jobs                 = len(chunk_parquet_fpath_list)
        count                    = 0
        debug_time_start         = time.time()
        for i_parquet_fpath_list in chunk_parquet_fpath_list:
            tasks.put(FeatureExtractionTask(i_parquet_fpath_list, train_dataset_folder, test_dataset_folder, loc_coord_dict, None, subsample_label, train_phase_boolean = True, have_tel = True, abnormal_agent_id_list = abnormal_agent_id_list, anomaly_agent_index_dict = anomaly_agent_index_dict, params = params, params_folder_name = params_folder_name))
            count += 1
            if count % 100 == 0:
                print(count, 'tasks generated')

        # Add a poison pill for each consumer
        for _ in range(num_consumers):
            tasks.put(None)
        while num_jobs:
            # chunk_agents_id, chunk_agent_vertex_dict, chunk_transition_count_dict, chunk_transition_agent_count_dict, chunk_duration_dict, chunk_duration_agent_dict, chunk_roadseg_stopp_duration_all_dict, chunk_travel_time_dict, chunk_travel_time_agent_dict,\
            #     chunk_trip_road_segments_list, chunk_X, chunk_hos_agent_KDEs_train_data_dict, chunk_hos_agent_faiss_indices_train_data_dict = results.get()
            chunk_agents_id, chunk_X, chunk_id_agent_X, chunk_agent_2_his_coord_set_dict, \
                chunk_agent_2_date_2_his_coord_set_dict, _, chunk_trip_df_idx, chunk_agent_2_date_2_stopp_coords_dict, chunk_raw_X = results.get()
            agent_id_list.extend(chunk_agents_id)
            if X is None:
                X           = chunk_X
                raw_X       = chunk_raw_X
                id_agent_X  = chunk_id_agent_X
                trip_df_idx = chunk_trip_df_idx
            else:
                X           = np.concatenate((X, chunk_X), axis = 0)
                raw_X       = np.concatenate((raw_X, chunk_raw_X), axis = 0)
                id_agent_X  = np.concatenate((id_agent_X, chunk_id_agent_X), axis = 0)
                trip_df_idx.extend(chunk_trip_df_idx)

            # agent_2_KDEs_train_data_dict      = {**agent_2_KDEs_train_data_dict,      **chunk_agent_2_KDEs_train_data_dict}
            agent_2_his_coord_set_dict        = {**agent_2_his_coord_set_dict,        **chunk_agent_2_his_coord_set_dict}
            agent_2_date_2_his_coord_set_dict = {**agent_2_date_2_his_coord_set_dict, **chunk_agent_2_date_2_his_coord_set_dict}
            agent_2_date_2_stopp_coords_dict  = {**agent_2_date_2_stopp_coords_dict,  **chunk_agent_2_date_2_stopp_coords_dict}

            num_jobs -= 1
            if num_jobs % 100 == 0:
                print(num_jobs, 'agents left')

        create_folder_if_not_exists(train_dataset_folder + "preprocess/{}".format(params_folder_name))
        save_to_pickle(train_dataset_folder + "preprocess/" + "{}{}train_agent_id_list.pkl".format(params_folder_name, subsample_label),                     agent_id_list)
        save_to_pickle(train_dataset_folder + "preprocess/" + "{}{}train_id_agent_X.pkl".format(params_folder_name, subsample_label),                        id_agent_X)
        save_to_pickle(train_dataset_folder + "preprocess/" + "{}{}train_X.pkl".format(params_folder_name, subsample_label),                                 X)
        save_to_pickle(train_dataset_folder + "preprocess/" + "{}{}train_raw_X.pkl".format(params_folder_name, subsample_label),                             raw_X)
        save_to_pickle(train_dataset_folder + "preprocess/" + "{}{}train_trip_df_idx.pkl".format(params_folder_name, subsample_label),                       trip_df_idx)
        # save_to_pickle(train_dataset_folder + "preprocess/" + "train_agent_2_KDEs_train_data_dict.pkl",      agent_2_KDEs_train_data_dict)
        save_to_pickle(train_dataset_folder + "preprocess/" + "{}{}train_agent_2_his_coord_set_dict.pkl".format(params_folder_name, subsample_label),        agent_2_his_coord_set_dict)
        save_to_pickle(train_dataset_folder + "preprocess/" + "{}{}train_agent_2_date_2_his_coord_set_dict.pkl".format(params_folder_name, subsample_label), agent_2_date_2_his_coord_set_dict)
        save_to_pickle(train_dataset_folder + "preprocess/" + "{}{}train_agent_2_date_2_stopp_coords_dict.pkl".format(params_folder_name, subsample_label),  agent_2_date_2_stopp_coords_dict)

        return []


    agent_id_list                           = load_from_pickle(train_dataset_folder + "preprocess/" + "{}{}train_agent_id_list.pkl".format(params_folder_name, subsample_label))
    id_agent_X                              = load_from_pickle(train_dataset_folder + "preprocess/" + "{}{}train_id_agent_X.pkl".format(params_folder_name, subsample_label))
    X                                       = load_from_pickle(train_dataset_folder + "preprocess/" + "{}{}train_X.pkl".format(params_folder_name, subsample_label))
    raw_X                                   = load_from_pickle(train_dataset_folder + "preprocess/" + "{}{}train_raw_X.pkl".format(params_folder_name, subsample_label))
    # agent_2_KDEs_train_data_dict            = load_from_pickle(train_dataset_folder + "preprocess/" + "train_agent_2_KDEs_train_data_dict.pkl")
    agent_2_his_coord_set_dict              = load_from_pickle(train_dataset_folder + "preprocess/" + "{}{}train_agent_2_his_coord_set_dict.pkl".format(params_folder_name, subsample_label))
    agent_2_date_2_his_coord_set_dict       = load_from_pickle(train_dataset_folder + "preprocess/" + "{}{}train_agent_2_date_2_his_coord_set_dict.pkl".format(params_folder_name, subsample_label))
    trip_df_idx                             = load_from_pickle(train_dataset_folder + "preprocess/" + "{}{}train_trip_df_idx.pkl".format(params_folder_name, subsample_label))
    # abnormal_agent_id_list                  = retrieve_abnormal_agent_id_list(test_dataset_folder)
    
    # X = raw_X
    X = agg_X_raw_X(X, raw_X, params)

    if id_agent_X.shape[0] != X.shape[0]:
        print("id_agent_X, X", id_agent_X.shape, X.shape)
        print("!!!!!!!!!!!!!!!!! id_agent_X is not aligned with X in dimensions")
        return

    print("id_agent_X, X", id_agent_X.shape[0], X.shape[0])
    
    debug_step_2 = False
    if not debug_step_2:

        
        X_normalized, scaler = normalize_data(X)
        np.save(train_dataset_folder + 'preprocess/{}{}train_X.npy'.format(params_folder_name, subsample_label), X)
        save_to_pickle(train_dataset_folder + 'preprocess/{}{}train_X_normalized.pkl'.format(params_folder_name, subsample_label), [X_normalized, scaler])
        save_to_pickle(train_dataset_folder + 'preprocess/{}{}train_X_scaler.pkl'.format(params_folder_name, subsample_label), scaler)

        """
        Step 2: train the LOF and KNN based anomaly detectors based on trips in the training trajectories
        """
        print("!!!!!!!!!! multi_anomlay_detectors_proc: GPU-based Faiss process start")
        trip_multi_ad_scores, classifiers       = multi_anomlay_detectors_proc(X_normalized, train_dataset_folder + 'preprocess/{}{}'.format(params_folder_name, subsample_label), params = params)
        print("!!!!!!!!!!!!!! trip_multi_ad_scores.shape", trip_multi_ad_scores.shape) # range(n,2)
        print("!!!!!!!!!! multi_anomlay_detectors_proc: GPU-based Faiss process end")

        train_multi_ad_scores_quantiles_list    = []
        stage_1_multi_ad_pvalues_list           = []
        print("trip_multi_ad_scores.shape", trip_multi_ad_scores.shape) # shape (n,2)
        for ad_scores in trip_multi_ad_scores.T: # trip_multi_ad_scores: (n,2). trip_multi_ad_scores.T: (2,n)
            print("ad_scores.shape", ad_scores.shape) # shape (n,)
            train_ad_scores_quantiles = np.quantile(ad_scores, np.arange(0.0, 1.00001, 0.00001)) # shape (m,)
            print("train_ad_scores_quantiles.shape", train_ad_scores_quantiles.shape) # shape (m,)
            print("train_ad_scores_quantiles", train_ad_scores_quantiles) # shape (m, 0)
            stage_1_ad_pvalues_array = calc_p_values(train_ad_scores_quantiles, ad_scores) # shape (n,)
            print("stage 1 p-values", stage_1_ad_pvalues_array) # shape (n,0)
            stage_1_multi_ad_pvalues_list.append(stage_1_ad_pvalues_array)
            train_multi_ad_scores_quantiles_list.append(train_ad_scores_quantiles)

        train_multi_ad_scores_quantiles_array     = np.array(train_multi_ad_scores_quantiles_list).T # .T ensures (m,2) shape
        print("train_multi_ad_scores_quantiles_array.shape", train_multi_ad_scores_quantiles_array.shape) # range(m,2)
        stage_1_multi_ad_pvalues_array            = np.array(stage_1_multi_ad_pvalues_list).T  # range(n,2)
        print("stage_1_multi_ad_pvalues_array.shape", stage_1_multi_ad_pvalues_array.shape) # range(n,2)
        stage_1_min_pvalues_array                 = np.min(stage_1_multi_ad_pvalues_array, axis=1) # range(n,)
        stage_1_avg_pvalues_array                 = np.mean(stage_1_multi_ad_pvalues_array, axis=1) # shape (n,)
        print("stage_1_min_pvalues_array.shape", stage_1_min_pvalues_array.shape) # range(n,)
        print("stage_1_min_pvalues_array: mean, min, max", np.mean(stage_1_min_pvalues_array), np.min(stage_1_min_pvalues_array), np.max(stage_1_min_pvalues_array))
        print("!!!!!!!!!!! stage_1_min_pvalues_array", sorted(stage_1_min_pvalues_array)[:10])
        # print("train_min_pvalues_array", train_min_pvalues_array)
        stage_1_min_pvalue_quantiles_array       = np.quantile(stage_1_min_pvalues_array, np.arange(0.0, 1.00001, 0.00001)) # range(m,)
        print("stage_2_pvalues_array_quantiles.shape", stage_1_min_pvalue_quantiles_array.shape) # range(m,)
        stage_2_pvalues_array                    = calc_p_values(stage_1_min_pvalue_quantiles_array, stage_1_min_pvalues_array) # range(n,)
        print("stage_2_pvalues_array.shape", stage_2_pvalues_array.shape) # range(n,)
        print("stage_2_pvalues_array: mean, min, max", np.mean(stage_2_pvalues_array), np.min(stage_2_pvalues_array), np.max(stage_2_pvalues_array))
        save_to_pickle(train_dataset_folder + "preprocess/" + "{}{}trip_multi_ad_scores.pkl".format(params_folder_name, subsample_label),                   trip_multi_ad_scores)
        save_to_pickle(train_dataset_folder + "preprocess/" + "{}{}train_multi_ad_scores_quantiles_array.pkl".format(params_folder_name, subsample_label),  train_multi_ad_scores_quantiles_array)
        save_to_pickle(train_dataset_folder + "preprocess/" + "{}{}stage_1_min_pvalue_quantiles_array.pkl".format(params_folder_name, subsample_label),     stage_1_min_pvalue_quantiles_array)
        save_to_pickle(train_dataset_folder + "preprocess/" + "{}{}stage_2_pvalues_array.pkl".format(params_folder_name, subsample_label),                  stage_2_pvalues_array)

        if ad_label == "LOF":
            sel_stage_pvalues = stage_1_multi_ad_pvalues_array[:,0]
        elif ad_label == "KNN":
            sel_stage_pvalues = stage_1_multi_ad_pvalues_array[:,1]
        elif ad_label == "SelKNN":
            sel_stage_pvalues = stage_1_multi_ad_pvalues_array[:,2]
        elif ad_label == "ensemble_stage_1_avg_pvalues":
            sel_stage_pvalues = stage_1_avg_pvalues_array
        elif ad_label == "ensemble_stage_1_min_pvalues":
            sel_stage_pvalues = stage_1_min_pvalues_array
        elif ad_label == "ensemble_stage_2_pvalues": 
            sel_stage_pvalues = stage_2_pvalues_array
        else:
            print("!!!!!!!!!!!!! ad_label has an invalid input")
            return 
        
        save_to_pickle(train_dataset_folder + "preprocess/" + "{}{}sel_stage_pvalues.pkl".format(params_folder_name, subsample_label), sel_stage_pvalues)


    # stage 3: NPSS scan
    stage_2_pvalues_array   = load_from_pickle(train_dataset_folder + "preprocess/" + "{}{}stage_2_pvalues_array.pkl".format(params_folder_name, subsample_label))
    sel_stage_pvalues = load_from_pickle(train_dataset_folder + "preprocess/" + "{}{}sel_stage_pvalues.pkl".format(params_folder_name, subsample_label))
    agent_2_train_npss_and_trip_indices_dict = npss_scan_proc(agent_id_list, sel_stage_pvalues, id_agent_X)

    save_to_pickle(train_dataset_folder + "preprocess/" + "{}{}agent_2_train_npss_and_trip_indices_dict.pkl".format(params_folder_name, subsample_label), agent_2_train_npss_and_trip_indices_dict)

    
    time2 = time.time()
    print("Whole process:---------------- ")
    str_running_time = print_time(time2-time1)
    # print("Agent specific process: -----------------")
    # print_time(time2-time_agents)
    save_to_pickle(train_dataset_folder + "preprocess/" + "{}{}training_time.pkl".format(params_folder_name, subsample_label), str_running_time)

    return


def histogram_plot(values, filepath, title = ""):

    create_folder_if_not_exists(os.path.dirname(filepath))
    if os.path.exists(filepath):
        # Remove the file
        os.remove(filepath)

    q01 = np.percentile(values, 1)
    q99 = np.percentile(values, 99)

    plt.hist(values, bins=50, range = (q01, q99), alpha=0.7, color='blue', edgecolor='black')

    # You can customize the plot with titles and labels if desired
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Save the plot to a file
    plt.savefig(filepath)  # You can use different formats like .png, .jpg, .pdf, etc.
    plt.close()


def two_histograms_plot(datalist_1, datalist_2, filepath, labels = ["", ""], title = ""):

    create_folder_if_not_exists(os.path.dirname(filepath))
    if os.path.exists(filepath):
        # Remove the file
        os.remove(filepath)

    # if len(datalist_1) > 1000:
    #     values = random.sample(values, 1000)
    # Create a histogram

    # q01 = np.percentile(datalist_1, 1)
    q00 = np.min(datalist_1 + datalist_2)
    q99 = np.percentile(datalist_1, 99)

    plt.hist(datalist_1, bins=50, range = (q00, q99), alpha=0.7, density=False, color='blue', label=labels[0], edgecolor='black')
    plt.hist(datalist_2, bins=50, range = (q00, q99), alpha=0.7, density=False, color='red',  label=labels[1], edgecolor='black')
    # You can customize the plot with titles and labels if desired
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Save the plot to a file
    plt.savefig(filepath)  # You can use different formats like .png, .jpg, .pdf, etc.
    plt.close()

class Table2Task(object):
    def __init__(self, parquet_fpath_list, agent_2_npss_diff_and_trip_indices_dict, id_agent_X_array, trip_df_idx, trip_anomaly_scores, abnormal_trip_ndarray):
        self.parquet_fpath_list = parquet_fpath_list
        self.agent_2_npss_diff_and_trip_indices_dict = agent_2_npss_diff_and_trip_indices_dict
        self.id_agent_X_array = id_agent_X_array
        self.trip_df_idx = trip_df_idx
        self.trip_anomaly_scores = trip_anomaly_scores
        self.abnormal_trip_ndarray = abnormal_trip_ndarray

    def __call__(self):
        table2_df_dict = dict()
        for i in range(len(self.parquet_fpath_list)):
            file_name = self.parquet_fpath_list[i]
            temp_df = pd.read_parquet(file_name)
            temp_df = temp_df.sort_values(by=['time_start'], ascending=[True]).reset_index(drop=True)
            temp_df = temp_df.rename(columns={'agent_id': 'agent'})
            temp_dt = pd.to_datetime(temp_df['time_start']).dt
            temp_df['timestamp'] = temp_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            temp_df['year'] = temp_dt.year
            temp_df['month'] = temp_dt.month
            temp_df['day'] = temp_dt.day
            temp_df['hour'] = temp_dt.hour
            temp_df.drop(columns=['time_start', 'time_stop', 'geometry', 'unique_stop_point', 'global_stop_points'], inplace=True)
            temp_df.insert(temp_df.shape[1], 'is_anomalous', False)
            temp_df.insert(temp_df.shape[1], 'anomaly_score', 0.0)
            temp_agent = temp_df.iloc[0]['agent']
            if temp_agent in self.agent_2_npss_diff_and_trip_indices_dict:
                temp_trips = np.where(self.id_agent_X_array == temp_agent)[0]
                for temp_trip in temp_trips:
                    temp_df.loc[self.trip_df_idx[temp_trip][0]:self.trip_df_idx[temp_trip][1] + 1, 'anomaly_score'] = \
                    self.trip_anomaly_scores[temp_trip]
                    if temp_trip in self.abnormal_trip_ndarray:
                        temp_df.loc[self.trip_df_idx[temp_trip][0]:self.trip_df_idx[temp_trip][1] + 1, 'is_anomalous'] = True
            else:
                temp_df['anomaly_score'] = 0.0
            temp_df = temp_df.sort_values(by=['timestamp', 'anomaly_score'], ascending=[True, False]).reset_index(
                drop=True)
            temp_df.drop_duplicates(subset=['agent', 'timestamp'], keep='first', inplace=True)
            ymdh_groups = temp_df.groupby([temp_df['year'], temp_df['month'], temp_df['day'], temp_df['hour']])
            for (y, m, d, h), group in ymdh_groups:
                temp_key = tuple([y, m, d, h])
                if temp_key not in table2_df_dict:
                    table2_df_dict[temp_key] = []
                table2_df_dict[temp_key].append(group)
        for key, value in table2_df_dict.items():
            table2_df_dict[key] = pd.concat(value)
        return table2_df_dict

    def __str__(self):
        return str(self.parquet_fpath_list)

def detection_test(train_dataset_folder, test_dataset_folder, parquet_fpath_list, abnormal_agent_id_list = None, anomaly_agent_index_dict = None, \
                   ad_label = "ensemble", subsample_label = "", trip_extraction_boolean = True, params = None, params_folder_name = ""):
    """
    this function conducts feature extraction, anomaly detection, and NPSS scan on the parquet files in the test folder and store the results on:
    agent_2_test_npss_dict.pkl: {agent_id: largest NPSS score in the test parquet file}
    agent_2_npss_difference_dict.pkl: {agent_id: the difference largest NPSS scores in the test and training parquet files}

    agent_2_event_idx_range_2_pvalue_dict {agent_id: {(start and end event indices of each trip): p-value}}
    """

    time1 = time.time()

    print("loading loc_coord_dict.pickle")
    if not os.path.exists(train_dataset_folder + 'preprocess/loc_coord_dict.pickle'):
        loc_coord_dict = new_create_location_coordinates(train_dataset_folder, test_dataset_folder)
    else:
        # loc_coord_dict = load_from_pickle(train_dataset_folder + 'preprocess/loc_coord_dict.pickle')
        loc_coord_dict = load_from_pickle(train_dataset_folder + 'preprocess/loc_coord_dict.pickle')


    """
    Step 1: Extract trips and their features from the trajectories stored in the parquet files in the file_list
    """
    

    if trip_extraction_boolean:
        agent_id_list                       = []
        X                                   = None # is None or an array of shape (n,d), where d is the number of features
        raw_X                               = None
        id_agent_X_array                    = None # is None or an array of shape (n,), that are agent_id values or -1.
        trip_start_end_datetimes_df         = None
        trip_df_idx                         = None
        agent_2_date_2_stopp_coords_dict    = dict()
        # print("filtered file_list size based on feature extraction done previously: {}".format(len(file_list)))
        n_agents        = len(parquet_fpath_list)
        if(n_agents == 0):
            return
        tasks           = multiprocessing.Queue()
        results         = multiprocessing.Queue()
        num_consumers   = 32
        n_agents        = len(parquet_fpath_list)

        if num_consumers > n_agents:
            num_consumers = int(np.ceil(n_agents * 0.5))

        K               = int(np.ceil(n_agents / num_consumers))
        consumers       = [Consumer(tasks, results)
                            for _ in range(num_consumers)]
        for w in consumers:
            w.start()

        agent_2_date_2_his_coord_set_dict = load_from_pickle(train_dataset_folder + "preprocess/" + "{}{}train_agent_2_date_2_his_coord_set_dict.pkl".format(params_folder_name, subsample_label))
        chunk_parquet_fpath_list = chunk_file_paths(parquet_fpath_list, K)
        num_jobs                 = len(chunk_parquet_fpath_list)
        count                    = 0
        debug_time_start         = time.time()
 
 
 
        for i_parquet_fpath_list in chunk_parquet_fpath_list:
            chunk_agent_id_list = []
            for idx, parquet_fname in enumerate(i_parquet_fpath_list):
                chunk_agent_id_list.append(np.int64(parquet_fname.split('/')[-1].split('.')[0]))
            chunk_agent_2_date_2_his_coord_set_dict = dict()
            for i_agent_id in chunk_agent_id_list:
                if i_agent_id in agent_2_date_2_his_coord_set_dict:
                    chunk_agent_2_date_2_his_coord_set_dict[i_agent_id] = agent_2_date_2_his_coord_set_dict[i_agent_id]
                else:
                    print("!!!!!!!!!!!!!!!!!!!!!!!! chunk_agent_2_date_2_his_coord_set_dict[i_agent_id]. Agent ID: {} has key error", i_agent_id)
                    chunk_agent_2_date_2_his_coord_set_dict[i_agent_id] = None
            tasks.put(FeatureExtractionTask(i_parquet_fpath_list, train_dataset_folder, test_dataset_folder, loc_coord_dict, chunk_agent_2_date_2_his_coord_set_dict, subsample_label, train_phase_boolean = False, have_tel = True, params = params, params_folder_name = params_folder_name))
            count += 1
            if count % 100 == 0:
                print(count, 'tasks generated')

        # Add a poison pill for each consumer
        for _ in range(num_consumers):
            tasks.put(None)
        while num_jobs:
            # chunk_agents_id, chunk_agent_vertex_dict, chunk_transition_count_dict, chunk_transition_agent_count_dict, chunk_duration_dict, chunk_duration_agent_dict, chunk_roadseg_stopp_duration_all_dict, chunk_travel_time_dict, chunk_travel_time_agent_dict,\
            #     chunk_trip_road_segments_list, chunk_X, chunk_hos_agent_KDEs_train_data_dict, chunk_hos_agent_faiss_indices_train_data_dict = results.get()
            chunk_agents_id, chunk_X, chunk_id_agent_X_array, _, _, chunk_trip_start_end_datetimes_df, chunk_trip_df_idx, \
                chunk_agent_2_date_2_stopp_coords_dict, chunk_raw_X = results.get()
            agent_id_list.extend(chunk_agents_id)

            if chunk_X.ndim == 0:
                print("!!!!!!!!!!!!!!!!!!!!!!! chunk_X has zero dimensions")
                continue

            if X is None:
                X                   = chunk_X
                raw_X               = chunk_raw_X
                id_agent_X_array    = chunk_id_agent_X_array
                trip_df_idx         = chunk_trip_df_idx
                trip_start_end_datetimes_df = chunk_trip_start_end_datetimes_df
            else:
                # print("X, chunk_X", X.shape, chunk_X.shape)
                X                   = np.concatenate((X, chunk_X), axis = 0)
                raw_X               = np.concatenate((raw_X, chunk_raw_X), axis = 0)
                id_agent_X_array    = np.concatenate((id_agent_X_array, chunk_id_agent_X_array), axis = 0)
                trip_df_idx.extend(chunk_trip_df_idx)
                trip_start_end_datetimes_df = pd.concat([trip_start_end_datetimes_df, chunk_trip_start_end_datetimes_df], axis = 0)

            agent_2_date_2_stopp_coords_dict = {**agent_2_date_2_stopp_coords_dict, **chunk_agent_2_date_2_stopp_coords_dict}

            num_jobs -= 1
            if num_jobs % 100 == 0:
                print(num_jobs, 'agents left')

        print("Start saving intermediate results for step 1")
        create_folder_if_not_exists(test_dataset_folder + "preprocess/{}".format(params_folder_name))
        save_to_pickle(test_dataset_folder + "preprocess/{}{}test_agent_id_list.pkl".format(params_folder_name, subsample_label),               agent_id_list)
        save_to_pickle(test_dataset_folder + "preprocess/{}{}test_X.pkl".format(params_folder_name, subsample_label),                           X)
        save_to_pickle(test_dataset_folder + "preprocess/{}{}test_raw_X.pkl".format(params_folder_name, subsample_label),                       raw_X)
        save_to_pickle(test_dataset_folder + "preprocess/{}{}test_trip_df_idx.pkl".format(params_folder_name, subsample_label),                 trip_df_idx)
        save_to_pickle(test_dataset_folder + "preprocess/{}{}test_id_agent_X_array.pkl".format(params_folder_name, subsample_label),            id_agent_X_array)
        save_to_pickle(test_dataset_folder + "preprocess/{}{}test_trip_start_end_datetimes_df.pkl".format(params_folder_name, subsample_label), trip_start_end_datetimes_df)
        save_to_pickle(test_dataset_folder + "preprocess/{}{}test_agent_2_date_2_stopp_coords_dict.pkl".format(params_folder_name, subsample_label), agent_2_date_2_stopp_coords_dict)
        print("Finished saving intermediate results for step 1")

    return []

    # return # *********************

    agent_id_list               = load_from_pickle(test_dataset_folder + "preprocess/{}{}test_agent_id_list.pkl".format(params_folder_name, subsample_label))
    X                           = load_from_pickle(test_dataset_folder + "preprocess/{}{}test_X.pkl".format(params_folder_name, subsample_label))
    raw_X                       = load_from_pickle(test_dataset_folder + "preprocess/{}{}test_raw_X.pkl".format(params_folder_name, subsample_label))
    # trip_df_idx: List (trip level) [start and end event indices of each trip]
    trip_df_idx                 = load_from_pickle(test_dataset_folder + "preprocess/{}{}test_trip_df_idx.pkl".format(params_folder_name, subsample_label))
    id_agent_X_array            = load_from_pickle(test_dataset_folder + "preprocess/{}{}test_id_agent_X_array.pkl".format(params_folder_name, subsample_label))
    trip_start_end_datetimes_df = load_from_pickle(test_dataset_folder + "preprocess/{}{}test_trip_start_end_datetimes_df.pkl".format(params_folder_name, subsample_label))

    step_2_bool = True
    if step_2_bool == True: 
        
        # X = raw_X
        X = agg_X_raw_X(X, raw_X, params)

        if X.shape[0] != id_agent_X_array.shape[0]:
            print("!!!!!!!!!!!!!!!! X is not alighed with id_agent_X_array on the number of rows")
            return

        train_multi_ad_scores_quantiles_array   = load_from_pickle(train_dataset_folder + "preprocess/{}{}train_multi_ad_scores_quantiles_array.pkl".format(params_folder_name, subsample_label))
        stage_1_min_pvalue_quantiles_array      = load_from_pickle(train_dataset_folder + "preprocess/{}{}stage_1_min_pvalue_quantiles_array.pkl".format(params_folder_name, subsample_label))
        scaler                                  = load_from_pickle(train_dataset_folder + 'preprocess/{}{}train_X_scaler.pkl'.format(params_folder_name, subsample_label))
        X_normalized                            = scaler.transform(X)

        print("train_multi_ad_scores_quantiles_array", train_multi_ad_scores_quantiles_array.shape)
        print("stage_1_min_pvalue_quantiles_array",    stage_1_min_pvalue_quantiles_array.shape)
        print("X_normalized.shape",                    X_normalized.shape)
        
        #pvalues_array, np.arange(0.0, 1.00001, 0.00001))

        clf_knn     = load_fast_knn(train_dataset_folder + 'preprocess/' + '{}{}FastKNN.pkl'.format(params_folder_name, subsample_label))
        # clf_sel_knn = load_fast_knn(train_dataset_folder + 'preprocess/' + '{}{}FastSelKNN.pkl'.format(params_folder_name, subsample_label))
        clf_lof     = load_fast_lof(train_dataset_folder + 'preprocess/' + '{}{}FastLOF.pkl'.format(params_folder_name, subsample_label))
        # clf_cblof   = load_from_pickle(train_dataset_folder + 'preprocess/' + '{}{}CBLOF.pkl'.format(params_folder_name, subsample_label))

        classifiers = {
            'FastLOF': clf_lof,
            'FastKNN': clf_knn
            # 'FastSelKNN': clf_sel_knn,
            # 'CBLOF': clf_cblof
        }

        multi_ad_stage_scores = []
        multi_ad_stage_1_pvalues = None
        for idx_ad, (clf_name, clf) in enumerate(classifiers.items()):
            if clf_name in ('FastSelKNN', 'CBLOF'):
                X_normalized_sel = X_normalized[:,[4, 5, 8]]
                ad_scores  = clf.decision_function(X_normalized_sel)
            else:
                ad_scores  = clf.decision_function(X_normalized)
            multi_ad_stage_scores.append(ad_scores)
            print("ad_scores shape: {}, mean: {}, min: {}, max: {}".format(ad_scores.shape, np.mean(ad_scores), np.min(ad_scores), np.max(ad_scores)))
            ad_pvalues = calc_p_values(train_multi_ad_scores_quantiles_array[:,idx_ad], ad_scores) # shape (n,)
            print("ad_pvalues, shape, mean, min, max", ad_pvalues.shape, np.mean(ad_pvalues), np.min(ad_pvalues), np.max(ad_pvalues))
            if idx_ad == 0:
                multi_ad_stage_1_pvalues = ad_pvalues.reshape(-1, 1) # reshape from (n,) to (n,1)
            else:
                multi_ad_stage_1_pvalues = np.column_stack((multi_ad_stage_1_pvalues, ad_pvalues.reshape(-1, 1))) # reshape from (n,) to (n,1)
        print("multi_ad_stage_1_pvalues.shape", multi_ad_stage_1_pvalues.shape)

        multi_ad_stage_scores_array = np.array(multi_ad_stage_scores).T
        stage_1_min_pvalues = np.min(multi_ad_stage_1_pvalues, axis=1) # shape (n,)
        stage_1_avg_pvalues = np.mean(multi_ad_stage_1_pvalues, axis=1) # shape (n,)
        print("stage_1_min_pvalues.shape: {}, mean {}, min: {}, max: {}".format(stage_1_min_pvalues.shape, np.mean(stage_1_min_pvalues), np.min(stage_1_min_pvalues), np.max(stage_1_min_pvalues)) )

        stage_2_pvalues = 1 - calc_p_values(stage_1_min_pvalue_quantiles_array, stage_1_min_pvalues) # shape (n, )
        print("stage_2_pvalues shape: {}, mean: {}, min: {}, max: {}".format(stage_2_pvalues.shape, np.mean(stage_2_pvalues), np.min(stage_2_pvalues), np.max(stage_2_pvalues)))

        save_to_pickle(test_dataset_folder + "preprocess/" + "{}{}test_multi_ad_stage_scores.pkl".format(params_folder_name, subsample_label),          multi_ad_stage_scores)
        save_to_pickle(test_dataset_folder + "preprocess/" + "{}{}test_multi_ad_stage_1_pvalues_array.pkl".format(params_folder_name, subsample_label), multi_ad_stage_1_pvalues)
        save_to_pickle(test_dataset_folder + "preprocess/" + "{}{}test_stage_2_pvalues_array.pkl".format(params_folder_name, subsample_label),          stage_2_pvalues)


        if ad_label == "LOF":
            sel_stage_pvalues = multi_ad_stage_1_pvalues[:,0]
        elif ad_label == "KNN":
            sel_stage_pvalues = multi_ad_stage_1_pvalues[:,1]
        elif ad_label == "SelKNN":
            sel_stage_pvalues = multi_ad_stage_1_pvalues[:,2]
        elif ad_label == 'CBLOF':
            sel_stage_pvalues = multi_ad_stage_1_pvalues[:,3]
        elif ad_label == "ensemble_stage_1_avg_pvalues":
            sel_stage_pvalues = stage_1_avg_pvalues
        elif ad_label == "ensemble_stage_1_min_pvalues":
            sel_stage_pvalues = stage_1_min_pvalues
        elif ad_label == "ensemble_stage_2_pvalues": 
            sel_stage_pvalues = stage_2_pvalues
        else:
            print("!!!!!!!!!!!!! ad_label has an invalid input")
            return 

        save_to_pickle(test_dataset_folder + "preprocess/" + "{}{}sel_stage_pvalues.pkl".format(params_folder_name, subsample_label), sel_stage_pvalues)
        

        agent_2_train_npss_and_trip_indices_dict     = load_from_pickle(train_dataset_folder + "preprocess/" + "{}{}agent_2_train_npss_and_trip_indices_dict.pkl".format(params_folder_name, subsample_label))
        agent_2_test_npss_and_trip_indices_dict      = dict()
        agent_2_npss_diff_and_trip_indices_dict      = dict()
        npss_diff_list                              = []
        true_anomaly_label_list                     = []
        agent_2_test_npss_and_trip_indices_dict      = npss_scan_proc(agent_id_list, sel_stage_pvalues, id_agent_X_array)
        # agent_2_event_idx_range_2_pvalue_dict       = dict()
        for idx, agent_id in enumerate(agent_id_list):
            # npss_set_idices, test_score_npss        = calc_npss(stage_2_pvalues[id_agent_X_array == agent_id])
            # agent_trip_start_end_datetimes_df       = trip_start_end_datetimes_df[id_agent_X_array == agent_id]
            # npss_set_start_date_time                = agent_trip_start_end_datetimes_df.loc[npss_set_idices[0],  'start_date_time']
            # npss_set_end_date_time                  = agent_trip_start_end_datetimes_df.loc[npss_set_idices[-1], 'end_date_time'  ]

            if agent_id not in agent_2_test_npss_and_trip_indices_dict:
                print("!!!!!!!!!!!!!!!!!! error: agent {} is not available from agent_2_test_npss_and_trip_indices_dict".format(agent_id))
                continue

            if agent_id not in agent_2_train_npss_and_trip_indices_dict:
                print("!!!!!!!!!!!!!!!!!! error: agent {} is not available from agent_2_train_npss_and_trip_indices_dict".format(agent_id))
                continue

            test_npss_score                                  = agent_2_test_npss_and_trip_indices_dict[agent_id][0]
            agent_npss_diff                                  = test_npss_score - agent_2_train_npss_and_trip_indices_dict[agent_id][0]
            npss_set_trip_indices                            = agent_2_test_npss_and_trip_indices_dict[agent_id][1]
            agent_2_npss_diff_and_trip_indices_dict[agent_id] = [agent_npss_diff, npss_set_trip_indices]

            if abnormal_agent_id_list is not None:
                if agent_id in abnormal_agent_id_list:
                    true_anomaly_label_list.append(1)
                else:
                    true_anomaly_label_list.append(0)
            npss_diff_list.append(agent_npss_diff)

            # print("agent_id: {}, test npss: {:.3f}, train npss: {:.3f}, difference: {:.3f}".format(agent_id, test_npss_score, agent_2_train_npss_and_trip_indices_dict[agent_id][0], agent_npss_diff))
            # print("npss-set time window: {} to {}".format(npss_set_trip_indices[0], npss_set_trip_indices[1]))

        create_folder_if_not_exists(test_dataset_folder + "preprocess/{}".format(params_folder_name))
        save_to_pickle(test_dataset_folder + "preprocess/" + "{}{}agent_2_test_npss_and_trip_indices_dict.pkl".format(params_folder_name, subsample_label),  agent_2_test_npss_and_trip_indices_dict)
        save_to_pickle(test_dataset_folder + "preprocess/" + "{}{}agent_2_npss_difference_dict.pkl".format(params_folder_name, subsample_label),             agent_2_npss_diff_and_trip_indices_dict)
    
    if abnormal_agent_id_list is not None:
        save_to_pickle(test_dataset_folder + "preprocess/" + "{}{}true_anomaly_label_list.pkl".format(params_folder_name, subsample_label),  true_anomaly_label_list)
        save_to_pickle(test_dataset_folder + "preprocess/" + "{}{}npss_diff_list_with_features.pkl".format(params_folder_name, subsample_label),  npss_diff_list)
        agent_aupr = pr_plot_proc(test_dataset_folder, np.array(true_anomaly_label_list), np.array(npss_diff_list), subsample_label, ad_label, 'Agent', params_folder_name)

    # Generate two tables
    all_agents_list = [np.int64(item.split('/')[-1].replace('.parquet','')) for item in parquet_fpath_list]
    test_agent_list = list(agent_2_npss_diff_and_trip_indices_dict.keys())
    # Table 1
    print("start generating table1")
    table1_df = pd.DataFrame(columns=['agent', 'is_anomalous', 'anomaly_score'],
                            index=list(range(len(all_agents_list))))
    anomaly_scores = np.array([value[0] for key, value in agent_2_npss_diff_and_trip_indices_dict.items()])
    min_score, max_score = anomaly_scores.min(axis=0), anomaly_scores.max(axis=0)
    for i, agent in enumerate(all_agents_list):
        table1_df.loc[i, 'agent'] = all_agents_list[i]
        if agent in test_agent_list:
            table1_df.loc[i, 'anomaly_score'] = agent_2_npss_diff_and_trip_indices_dict[agent][0]
        else:
            table1_df.loc[i, 'anomaly_score'] = min_score
    table1_df['anomaly_score'] = (table1_df['anomaly_score'] - min_score) / (max_score - min_score)
    mean = table1_df['anomaly_score'].values.mean()
    std = table1_df['anomaly_score'].values.std()
    threashold = mean + 3.5 * std
    table1_df['is_anomalous'] = False
    table1_df.loc[table1_df['anomaly_score'] >= threashold, ['is_anomalous']] = True
    table1_df.to_parquet(os.path.join(test_dataset_folder, "preprocess", params_folder_name, "table1.parquet"), index=False)
    print("finish generating table1, start generating table2")
    # Table 2
    def p2anomaly_score(x):
        return 1 - min((0.2 + x), 1.0)
    p2anomaly_score_vectorized = np.vectorize(p2anomaly_score)

    trip_anomaly_scores = p2anomaly_score_vectorized(sel_stage_pvalues)
    abnormal_agent_list = list(table1_df.loc[table1_df['is_anomalous'] == True]['agent'])
    abnormal_trip_list = []
    for temp_agent in abnormal_agent_list:
        abnormal_trip_list.append(agent_2_test_npss_and_trip_indices_dict[temp_agent][1])
    if len(abnormal_trip_list) > 0:
        abnormal_trip_ndarray = np.concatenate(abnormal_trip_list)
        trip_anomaly_scores[abnormal_trip_ndarray] = 1-(sel_stage_pvalues[abnormal_trip_ndarray])
    else:
        abnormal_trip_ndarray = np.empty(0)
    save_to_pickle(test_dataset_folder + "preprocess/" + "{}{}trip_anomaly_scores.pkl".format(params_folder_name,subsample_label),trip_anomaly_scores)

    tasks = multiprocessing.Queue()
    results = multiprocessing.Queue()
    num_consumers = 32
    n_files = len(parquet_fpath_list)
    if num_consumers > n_files:
        num_consumers = int(np.ceil(n_files * 0.5))
    K = int(np.ceil(n_files / num_consumers))
    consumers = [Consumer(tasks, results)
                 for _ in range(num_consumers)]
    for w in consumers:
        w.start()
    chunk_file_list = chunk_list(parquet_fpath_list, K)
    num_jobs = len(chunk_file_list)
    print("start generating chunk")
    for temp_file_list in chunk_file_list:
        tasks.put(Table2Task(temp_file_list, agent_2_npss_diff_and_trip_indices_dict, id_agent_X_array, trip_df_idx, trip_anomaly_scores, abnormal_trip_ndarray))
    print("finish generating chunk")
    # Add a poison pill for each consumer
    for _ in range(num_consumers):
        tasks.put(None)
    table2_df_dict = dict()
    count = 0
    time_table2_chunk_start = time.time()
    while num_jobs:
        chunk_table2_df_dict = results.get()
        for key, item in chunk_table2_df_dict.items():
            if key not in table2_df_dict:
                table2_df_dict[key] = []
            table2_df_dict[key].append(item)
        num_jobs -= 1
        count += 1
        if count % 5 == 1:
            temp_time = time.time()
            temp_used_time = temp_time - time_table2_chunk_start
            temp_rest_time = temp_used_time / count * (len(chunk_file_list) - count)
            print(count, "chunk took time:", temp_used_time, "rest:", temp_rest_time)
    print("chunk end")
    table2_folder = os.path.join(test_dataset_folder, "preprocess", params_folder_name, "table2")
    if not os.path.exists(table2_folder):
        os.makedirs(table2_folder)
    num_key = len(list(table2_df_dict.keys()))
    count = 0
    print("start generating table2")
    time_table2_gen_start = time.time()
    for key, value in table2_df_dict.items():
        temp_df = pd.concat(value)
        temp_df.to_parquet(os.path.join(table2_folder, "_".join([str(key[0]), str(key[1]), str(key[2]), str(key[3])]) + '.parquet'), index=False)
        count += 1
        if count % 20 == 1:
            temp_time = time.time()
            temp_used_time = temp_time - time_table2_gen_start
            temp_rest_time = temp_used_time / count * (num_key - count)
            print(count, "table took time:", temp_used_time, "rest:", temp_rest_time)

    time2 = time.time()
    print("Whole process:---------------- ")

    str_running_time = print_time(time2-time1)
    # print("Agent specific process: -----------------")
    # print_time(time2-time_agents)
    save_to_pickle(train_dataset_folder + "preprocess/" + "{}{}inference_time.pkl".format(params_folder_name, subsample_label), str_running_time)


def pr_plot_proc(test_dataset_folder, y, scores, subsample_label, ad_label, level_label, params_folder_name = ""):
    lr_precision, lr_recall, thresholds = precision_recall_curve(y, scores)
    aupr = auc(lr_recall, lr_precision)
    print("**************** Detction {} AUPR: {:.4f}".format(level_label, aupr))
    fontsize = 14
    plt.plot(lr_recall, lr_precision, marker='.', lw=2, label=aupr)
    plt.xlabel('Recall', fontsize=fontsize)
    plt.ylabel('Precision', fontsize=fontsize)
    plt.title('{} detector: PR Curve of {} Anomaly Detection'.format(ad_label, level_label))
    plt.legend(loc='upper right')
    plt.tight_layout()
    create_folder_if_not_exists(test_dataset_folder + "preprocess/plots/{}".format(params_folder_name))
    plt.savefig(os.path.join(test_dataset_folder, "preprocess/plots/{}{}Detector-{}-{}-PR.png".format(params_folder_name, subsample_label, ad_label, level_label)))
    plt.show()
    plt.close('all')
    return aupr



def npss_scan_proc(agent_id_list, stage_2_pvalues, id_agent_X_array):
    """
    INPUT
    agent_id_list:               A list containing the IDs of agents to be processed.
    stage_2_pvalues:             A numpy array of shape (n,) containing p-values for each trip.
    id_agent_X_array:            A numpy array of shape (n,) where each entry is agent IDs corresponding to
                                 the trip p-values in stage_2_pvalues. An entry of -1 indicates a trip that is ignored.
                                 In the training phase, if both training and test trips are included, we should ignore test
                                 trips when doing NPSS scan on the trianing trips
    trip_start_end_datetimes_df: An optional numpy array of shape (n,2) containing the start and end datetimes
                                 for each trip. It defaults to None if not provided.
    OUTPUT
    agent_2_test_npss_and_time_window_dict: {agent_id: [test npss score, [start date time, end datetime]]}
                                            A dictionary where each key is an agent ID, and each value is a list containing
                                            the agent's test NPSS score and a list with the start and end datetime of the
                                            evaluated time window.
    """
    tasks           = multiprocessing.Queue()
    results         = multiprocessing.Queue()
    num_consumers   = 32
    n_agents        = len(agent_id_list)
    if num_consumers > n_agents:
        num_consumers = int(np.ceil(n_agents * 0.5))
    K               = int(np.ceil(n_agents / num_consumers))
    consumers       = [Consumer(tasks, results)
                        for _ in range(num_consumers)]
    for w in consumers:
        w.start()
    chunk_agent_id_list      = chunk_list(agent_id_list, K)
    num_jobs                 = len(chunk_agent_id_list)
    count                    = 0

    for i_agent_id_list in chunk_agent_id_list:
        tasks.put(NPSSTask(i_agent_id_list, stage_2_pvalues, id_agent_X_array))
        count += 1
        if count % 100 == 0:
            print(count, 'tasks generated')
    # Add a poison pill for each consumer
    for _ in range(num_consumers):
        tasks.put(None)
    agent_2_npss_and_trip_indices_dict = dict()
    while num_jobs:
        chunk_agent_2_npss_and_trip_indices_dict = results.get()
        if chunk_agent_2_npss_and_trip_indices_dict is not None:
            agent_2_npss_and_trip_indices_dict = {**agent_2_npss_and_trip_indices_dict, **chunk_agent_2_npss_and_trip_indices_dict}
        else:
            print("!!!!!!!!!!!!!!!!!!! error the return one task is None")
        num_jobs -= 1
        if num_jobs % 100 == 0:
            print(num_jobs, 'agents left')
    return agent_2_npss_and_trip_indices_dict
    # agent_2_test_npss_and_time_window_dict = dict()
    # for agent_id in agent_id_list:
    #     npss_set_idices, test_npss_score        = calc_npss(stage_2_pvalues[id_agent_X_array == agent_id])
    #     agent_trip_start_end_datetimes_df       = trip_start_end_datetimes_df[id_agent_X_array == agent_id]
    #     npss_set_start_date_time                = agent_trip_start_end_datetimes_df.loc[npss_set_idices[0],  'start_date_time']
    #     npss_set_end_date_time                  = agent_trip_start_end_datetimes_df.loc[npss_set_idices[-1], 'end_date_time'  ]
    #     agent_2_test_npss_and_time_window_dict[agent_id] = [test_npss_score, [npss_set_start_date_time, npss_set_end_date_time]]



def chunk_list(lst, K):
    """
    Splits a list into chunks of size K or smaller.

    :param lst: The list to be split.
    :param K: The maximum size of each chunk.
    :return: A list of chunks, where each chunk is a list of size K or smaller.
    """
    return [lst[i:i + K] for i in range(0, len(lst), K)]

class NPSSTask(object):
    def __init__(self, agent_id_list, stage_2_pvalues, id_agent_X_array):
        self.agent_id_list                  = agent_id_list
        self.stage_2_pvalues                = stage_2_pvalues # shape(n,)
        self.id_agent_X_array               = id_agent_X_array # shape(n,)

    def __call__(self):
        agent_2_npss_and_trip_indices_dict = dict()
        # try:
        for idx, agent_id in enumerate(self.agent_id_list):
            if idx % 500 == 0: 
                print("idx, agent_id", idx, agent_id)
            # try:
            temp_agent_trip_indices = np.where(self.id_agent_X_array == agent_id)[0]
            sel_pvalues             = self.stage_2_pvalues[temp_agent_trip_indices]
            if len(sel_pvalues) == 0:
                continue
            npss_set_idices, npss_score = calc_npss(sel_pvalues)
            npss_set_all_trip_indices   = temp_agent_trip_indices[npss_set_idices]
            agent_2_npss_and_trip_indices_dict[agent_id] = [npss_score, npss_set_all_trip_indices]

        return agent_2_npss_and_trip_indices_dict

    def __str__(self):
        return str(self.agent_id_list)




def retrieve_abnormal_agent_id_list(test_dataset_folder, bool_trial_2):
    """
    """
    abnormal_agent_id_list = []
    if bool_trial_2:
        gts_folder = os.path.join(test_dataset_folder, "gts")
        for root, ds, fs in os.walk(gts_folder):
            for f in sorted(fs):
                if f.endswith('.parquet'):
                    temp_parquet_file = os.path.join(root, f)
                    temp_parquet_content = pd.read_parquet(temp_parquet_file)
                    temp_agent = np.int64(temp_parquet_content.iloc[0]['agent'])
                    abnormal_agent_id_list.append(temp_agent)
    else:
        for filename in os.listdir(os.path.join(test_dataset_folder, "truth/gts")):
            if filename.startswith('agent='):
                abnormal_agent_id_list.append(np.int64(filename.replace('agent=', '')))
    return abnormal_agent_id_list

def normalize_scores(scores):
    min_score = min(scores)
    max_score = max(scores)
    range_score = max_score - min_score

    # Avoid division by zero in case all scores are the same
    if range_score == 0:
        return [0] * len(scores)

    normalized_scores = [(score - min_score) / range_score for score in scores]
    return normalized_scores





from pytz import timezone
class Log:
    def __init__(self, log_fname):
        self.fname = log_fname
        self.logf = open(log_fname,"a") #append mode
        fmt = "%Y-%m-%d-%H-%M-%S"
        now_utc = datetime.datetime.now(timezone('UTC'))
        now_central = now_utc.astimezone(timezone('US/Central'))
        self.logf.write("\n***************************************")
        self.logf.write("******************{}******************\n".format(now_central.strftime(fmt)))

    def write(self, line):
        self.logf.write(line + "\n")
        print(line)

    def write_separator_line(self):
        self.logf.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    def close(self):
        self.logf.close()







def ST_departure_kde_analysis(train_dataset_folder, test_dataset_folder, abnormal_agent_id_list, anomaly_agent_index_dict, subsample_label = "", params = None, params_folder_name = ""):

    if not os.path.exists(train_dataset_folder + "preprocess/gdf_stopp.pkl"):
        gdf, stopp_df  = retrieve_gdf_stopp(train_dataset_folder)
        save_to_pickle(train_dataset_folder + "preprocess/gdf_stopp.pkl", [gdf, stopp_df])
    else:
        [gdf, stopp_df] = load_from_pickle(train_dataset_folder + "preprocess/gdf_stopp.pkl")

    print("loading loc_coord_dict.pickle")
    if not os.path.exists(train_dataset_folder + 'preprocess/loc_coord_dict.pickle'):
        loc_coord_dict = create_location_coordinates(train_dataset_folder, gdf, stopp_df)
    else:
        # loc_coord_dict = load_from_pickle(train_dataset_folder + 'preprocess/loc_coord_dict.pickle')
        loc_coord_dict = load_from_pickle(train_dataset_folder + 'preprocess/loc_coord_dict.pickle')

    # abnormal_agent_id_list  = retrieve_abnormal_agent_id_list(test_dataset_folder)
    for agent_id in abnormal_agent_id_list[:8]:
        print("!!!!!!!!!!! agent id: ", agent_id )
        test_parquet_fpath      = test_dataset_folder  + 'event_logs/{}.parquet'.format(agent_id)
        train_parquet_fpath     = train_dataset_folder + 'event_logs/{}.parquet'.format(agent_id)
        [a, b] = anomaly_agent_index_dict[agent_id]
        _, _, _, _, _, _, _, trip_df_idx_train, X_ST_train, _   = trip_feature_extraction(train_parquet_fpath, train_dataset_folder, test_dataset_folder, loc_coord_dict, subsample_label, train_phase_boolean = True,  abnormal_agent_id_list = abnormal_agent_id_list, anomaly_agent_index_dict = anomaly_agent_index_dict, params = params, params_folder_name = params_folder_name)
        agent_2_date_2_his_coord_set_dict                       = load_from_pickle(train_dataset_folder + "preprocess/" + "{}{}train_agent_2_date_2_his_coord_set_dict.pkl".format(params_folder_name, subsample_label))
        _, _, _, _, _, _, _, trip_df_idx_test,  X_ST_test, _    = trip_feature_extraction(test_parquet_fpath,  train_dataset_folder, test_dataset_folder, loc_coord_dict, subsample_label, agent_2_date_2_his_coord_set_dict, train_phase_boolean = False, abnormal_agent_id_list = abnormal_agent_id_list, anomaly_agent_index_dict = anomaly_agent_index_dict, params = params, params_folder_name = params_folder_name)
        [kde_depature_hour, kde_depature_ST, date_travel_time_kde, ST_scaler] = load_from_dill(train_dataset_folder + "preprocess/{}KDE/{}{}_KDEs.dill".format(params_folder_name, subsample_label, agent_id))
        trip_df_idx_test = np.array(trip_df_idx_test)
        # print(trip_df_idx_test.shape)
        starts_within_range   = trip_df_idx_test[:, 0] <= b
        ends_within_range     = trip_df_idx_test[:, 1] >= a
        ab_event_indices      = np.where(starts_within_range & ends_within_range)[0]
        
        if kde_depature_ST is not None:
            norm_X_ST_train = ST_scaler.transform(X_ST_train)
            norm_X_ST_test  = ST_scaler.transform(X_ST_test)
            densities_train = kde_depature_ST(norm_X_ST_train.T)
            densities_test  = kde_depature_ST(norm_X_ST_test.T)
            print("!!!!!!!!!!!!!! {}: train density: min: {:.4f}, max: {:.4f}, mean: {:.4f}, std: {:.4f}".format(agent_id, np.min(densities_train), np.max(densities_train), np.mean(densities_train), np.std(densities_train)))
            print("!!!!!!!!!!!!!! {}: test density: min: {:.4f}, max: {:.4f}, mean: {:.4f}, std: {:.4f}".format(agent_id, np.min(densities_test),  np.max(densities_test),  np.mean(densities_test),  np.std(densities_test)))

        # continue
        if kde_depature_ST is not None:
            lat_lon_time_KDE_plot(agent_id, kde_depature_ST, ST_scaler, X_ST_train, X_ST_test[ab_event_indices,:], train_dataset_folder, params_folder_name)
        else:
            print("!!!!!!!!!!!! agent id: {} has None deapture ST KDE".format(agent_id))







def genereate_y(abnormal_agent_id_list, anomaly_agent_index_dict, id_agent_X, trip_df_idx, n_trips):

    y = np.zeros((n_trips,))
    for ab_agent_id in abnormal_agent_id_list:
        # print("abnormal agent id:", ab_agent_id)
        [a, b] = anomaly_agent_index_dict[ab_agent_id]

        # Get the indices of trips for the current ab_agent_id
        ab_agent_indices     = np.where(id_agent_X == ab_agent_id)[0]
        # print("ab_agent_indices", ab_agent_indices)
        # print("trip_df_idx", trip_df_idx.shape)
        ab_agent_trip_df_idx_array = np.array(trip_df_idx)[ab_agent_indices, :]
        # print("ab_agent_trip_df_idx", ab_agent_trip_df_idx_array)

        # Determine which trips overlap with the anomaly range
        starts_within_range   = ab_agent_trip_df_idx_array[:, 0] <= b
        ends_within_range     = ab_agent_trip_df_idx_array[:, 1] >= a
        overlapping_indices   = np.where(starts_within_range & ends_within_range)[0]

        # Update y for the overlapping trips of the current ab_agent_id
        y[ab_agent_indices[overlapping_indices]] = 1

    return y



def postproc_X_y(train_dataset_folder, test_dataset_folder, abnormal_agent_id_list, anomaly_agent_index_dict, subsample_label, params_folder_name = ""):
    agent_id_list       = load_from_pickle(test_dataset_folder + "preprocess/{}{}test_agent_id_list.pkl".format(params_folder_name, subsample_label))
    test_X              = load_from_pickle(test_dataset_folder + "preprocess/{}{}test_X.pkl".format(params_folder_name, subsample_label))
    # trip_df_idx: List (trip level) [start and end event indices of each trip]
    trip_df_idx         = load_from_pickle(test_dataset_folder + "preprocess/{}{}test_trip_df_idx.pkl".format(params_folder_name, subsample_label))
    id_agent_X_array    = load_from_pickle(test_dataset_folder + "preprocess/{}{}test_id_agent_X_array.pkl".format(params_folder_name, subsample_label))
    test_y              = genereate_y(abnormal_agent_id_list, anomaly_agent_index_dict, id_agent_X_array, trip_df_idx, test_X.shape[0])

    train_X, _          = load_from_pickle(train_dataset_folder + "preprocess/" + "{}{}train_X_y.pkl".format(params_folder_name, subsample_label))
    train_id_agent_X    = load_from_pickle(train_dataset_folder + "preprocess/" + "{}{}train_id_agent_X.pkl".format(params_folder_name, subsample_label))
    train_X             = train_X[train_id_agent_X != -1,:] # this step will remove rows related to test trips
    n_train_trips       = train_X.shape[0]
    train_y             = np.zeros((n_train_trips,))
    X                   = np.concatenate((train_X, test_X), axis = 0)
    y                   = np.concatenate((train_y, test_y), axis = 0)

    save_to_pickle(train_dataset_folder + "preprocess/" + "{}{}train_raw_X_y.pkl".format(params_folder_name, subsample_label), [X, y])



def animation_map_proc(train_dataset_folder, test_dataset_folder, abnormal_agent_id_list, anomaly_agent_index_dict, subsample_label = "", params_folder_name = ""):
    
    if not os.path.exists(train_dataset_folder + "preprocess/gdf_stopp.pkl"):
        gdf, stopp_df  = retrieve_gdf_stopp(train_dataset_folder)
        save_to_pickle(train_dataset_folder + "preprocess/gdf_stopp.pkl", [gdf, stopp_df])
    else:
        [gdf, stopp_df] = load_from_pickle(train_dataset_folder + "preprocess/gdf_stopp.pkl")

    print("loading loc_coord_dict.pickle")
    if not os.path.exists(train_dataset_folder + 'preprocess/loc_coord_dict.pickle'):
        loc_coord_dict = create_location_coordinates(train_dataset_folder, gdf, stopp_df)
    else:
        # loc_coord_dict = load_from_pickle(train_dataset_folder + 'preprocess/loc_coord_dict.pickle')
        loc_coord_dict = load_from_pickle(train_dataset_folder + 'preprocess/loc_coord_dict.pickle')

    agent_ab_event_locations = dict()
    # abnormal_agent_id_list  = retrieve_abnormal_agent_id_list(test_dataset_folder)
    for agent_id in abnormal_agent_id_list[:]:
        print("!!!!!!!!!!! agent id: ", agent_id )
        test_parquet_fpath      = test_dataset_folder  + 'event_logs/{}.parquet'.format(agent_id)
        train_parquet_fpath     = train_dataset_folder + 'event_logs/{}.parquet'.format(agent_id)

        have_tel = True
        df                      = pd.read_parquet(test_parquet_fpath)
        if df['agent_id'].nunique() > 1:
            df                  = df.loc[df['agent_id'] == agent_id]
        df = df.drop_duplicates(keep='first')
        df = df.drop_duplicates(subset=["time_start"], keep='first')
        df['arrival_datetime'] = pd.to_datetime(df['time_start'])
        df['depart_datetime'] = pd.to_datetime(df['time_stop'])
        df['arrival_timestamp'] = df['arrival_datetime'].apply(lambda x: x.timestamp())
        df['depart_timestamp'] = df['depart_datetime'].apply(lambda x: x.timestamp())
        """
        trip_locations:     list (trip level) [list of locations (LocationUUID) for each trip]
        trip_df_idx:        List (trip level) [start and end event indices of each trip]
        """
        trip_features, trip_points, trip_count, _, trip_datetime_df, trip_locations, trip_df_idx = read_traveldis(df, have_tel)

        [a, b] = anomaly_agent_index_dict[agent_id]
        trip_df_idx           = np.array(trip_df_idx)
        starts_within_range   = trip_df_idx[:, 0] <= b
        ends_within_range     = trip_df_idx[:, 1] >= a
        ab_event_indices      = np.where(starts_within_range & ends_within_range)[0]
        print(ab_event_indices)
        agent_ab_event_locations[agent_id] = [trip_locations[index] for index in ab_event_indices]

    print(agent_ab_event_locations)
    save_to_pickle(train_dataset_folder + "preprocess/{}{}agent_ab_event_locations.pkl".format(params_folder_name, subsample_label), agent_ab_event_locations)




def debug(train_dataset_folder, test_dataset_folder, abnormal_agent_id_list, anomaly_agent_index_dict, params_folder_name, subsample_label):
        debug_folder = "debug/"
        create_folder_if_not_exists(train_dataset_folder + "preprocess/plots/{}".format(debug_folder))
        X_test  = load_from_pickle(test_dataset_folder + "preprocess/{}{}test_X.pkl".format(params_folder_name, subsample_label))
        raw_X_test  = load_from_pickle(test_dataset_folder + "preprocess/{}{}test_raw_X.pkl".format(params_folder_name, subsample_label))
        test_trip_df_idx = load_from_pickle(test_dataset_folder + "preprocess/{}{}test_trip_df_idx.pkl".format(params_folder_name, subsample_label))
        test_id_agent_X = load_from_pickle(test_dataset_folder + "preprocess/{}{}test_id_agent_X_array.pkl".format(params_folder_name, subsample_label))
        y_test = genereate_y(abnormal_agent_id_list, anomaly_agent_index_dict, test_id_agent_X, test_trip_df_idx, X_test.shape[0])
        two_histograms_plot(list(X_test[y_test==0,5]), list(X_test[y_test==1,5]), train_dataset_folder + "preprocess/plots/{}{}test_depature_ST_density_histogram.png".format(debug_folder, subsample_label),   ["all", "abnormal agents"], "Test: Depature Spatiotemporal Density")

        X_train = load_from_pickle(train_dataset_folder + "preprocess/" + "{}{}train_X.pkl".format(params_folder_name, subsample_label))
        raw_X_train = load_from_pickle(train_dataset_folder + "preprocess/" + "{}{}train_raw_X.pkl".format(params_folder_name, subsample_label))
        train_trip_df_idx = load_from_pickle(train_dataset_folder + "preprocess/" + "{}{}train_trip_df_idx.pkl".format(params_folder_name, subsample_label))
        train_id_agent_X = load_from_pickle(train_dataset_folder + "preprocess/" + "{}{}train_id_agent_X.pkl".format(params_folder_name, subsample_label))
        y_train = genereate_y(abnormal_agent_id_list, anomaly_agent_index_dict, train_id_agent_X * -1, train_trip_df_idx, X_train.shape[0])
        two_histograms_plot(list(X_train[y_train==0,5]), list(X_train[y_train==1,5]), train_dataset_folder + "preprocess/plots/{}{}train_depature_ST_density_histogram.png".format(debug_folder, subsample_label),   ["all", "abnormal agents"], "Train: Depature Spatiotemporal Density")

        two_histograms_plot(list(X_train[y_train==1,5]), list(X_test[y_test==1,5]), train_dataset_folder + "preprocess/plots/{}{}train_depature_ST_true_ab_density_histogram.png".format(debug_folder, subsample_label),   ["all", "abnormal agents"], "Train: Depature Spatiotemporal Density")


        indices_train = train_id_agent_X == -8089562 # -12385674
        indices_test = test_id_agent_X == 8089562 # 12385674
        # print(sum(indices_train), sum(indices_test))
        print(raw_X_train[indices_train, 0])
        print(raw_X_test[indices_test, 0])
        # return 

        print("|||||||||||||||||||||")
        print(list(X_train[indices_train,5][:]))
        print(list(X_test[indices_test,5][:]))
        print(len(list(X_train[indices_train,5][:])), len(list(X_test[indices_test,5][:])))
        print("|||||||||||||||||||||")

        print("\ntrain raw_X vs. test raw_X")
        print(list(raw_X_train[indices_train,0][:10]))
        print(list(raw_X_test[indices_test,0][:10]))

        # print("|||||||||||||||||||||")
        # print(test_id_agent_X[:30])
        # print(train_id_agent_X[:30])
        # print("|||||||||||||||||||||")

        
        # sel_raw_X_train = raw_X_train[train_id_agent_X < 0, :]

        # for i in range(3):
        #     print(sel_raw_X_train[i,:], raw_X_test[i,:])
        #     print("-------------------------------")


def analysis(train_dataset_folder, params_folder_name, subsample_label):
    trip_multi_ad_scores = load_from_pickle(train_dataset_folder + "preprocess/" + "{}{}trip_multi_ad_scores.pkl".format(params_folder_name, subsample_label))

    train_multi_ad_scores_quantiles_list    = []
    stage_1_multi_ad_pvalues_list           = []
    print("trip_multi_ad_scores.shape", trip_multi_ad_scores.shape) # shape (n,2)
    for ad_scores in trip_multi_ad_scores.T: # trip_multi_ad_scores: (n,2). trip_multi_ad_scores.T: (2,n)
        print("ad_scores.shape", ad_scores.shape) # shape (n,)
        train_ad_scores_quantiles = np.quantile(ad_scores, np.arange(0.0, 1.00001, 0.00001)) # shape (m,)
        print("train_ad_scores_quantiles.shape", train_ad_scores_quantiles.shape) # shape (m,)
        print("train_ad_scores_quantiles", train_ad_scores_quantiles) # shape (m, 0)
        stage_1_ad_pvalues_array = calc_p_values(train_ad_scores_quantiles, ad_scores) # shape (n,)
        print("stage 1 p-values", stage_1_ad_pvalues_array) # shape (n,0)
        stage_1_multi_ad_pvalues_list.append(stage_1_ad_pvalues_array)
        train_multi_ad_scores_quantiles_list.append(train_ad_scores_quantiles)

    train_multi_ad_scores_quantiles_array     = np.array(train_multi_ad_scores_quantiles_list).T # .T ensures (m,2) shape
    print("train_multi_ad_scores_quantiles_array.shape", train_multi_ad_scores_quantiles_array.shape) # range(m,2)
    stage_1_multi_ad_pvalues_array            = np.array(stage_1_multi_ad_pvalues_list).T  # range(n,2)
    print("stage_1_multi_ad_pvalues_array.shape", stage_1_multi_ad_pvalues_array.shape) # range(n,2)
    stage_1_min_pvalues_array                 = np.min(stage_1_multi_ad_pvalues_array, axis=1) # range(n,)
    stage_1_avg_pvalues_array                 = np.mean(stage_1_multi_ad_pvalues_array, axis=1) # shape (n,)
    print("stage_1_min_pvalues_array.shape", stage_1_min_pvalues_array.shape) # range(n,)
    print("stage_1_min_pvalues_array: mean, min, max", np.mean(stage_1_min_pvalues_array), np.min(stage_1_min_pvalues_array), np.max(stage_1_min_pvalues_array))
    print("!!!!!!!!!!! stage_1_min_pvalues_array", sorted(stage_1_min_pvalues_array)[:10])
    # print("train_min_pvalues_array", train_min_pvalues_array)
    stage_1_min_pvalue_quantiles_array       = np.quantile(stage_1_min_pvalues_array, np.arange(0.0, 1.00001, 0.00001)) # range(m,)
    print("stage_2_pvalues_array_quantiles.shape", stage_1_min_pvalue_quantiles_array.shape) # range(m,)
    stage_2_pvalues_array                    = calc_p_values(stage_1_min_pvalue_quantiles_array, stage_1_min_pvalues_array) # range(n,)
    print("stage_2_pvalues_array.shape", stage_2_pvalues_array.shape) # range(n,)
    print("stage_2_pvalues_array: mean, min, max", np.mean(stage_2_pvalues_array), np.min(stage_2_pvalues_array), np.max(stage_2_pvalues_array))
    
    save_to_pickle(train_dataset_folder + "preprocess/" + "{}{}train_multi_ad_scores_quantiles_array.pkl".format(params_folder_name, subsample_label),  train_multi_ad_scores_quantiles_array)
    save_to_pickle(train_dataset_folder + "preprocess/" + "{}{}stage_1_min_pvalue_quantiles_array.pkl".format(params_folder_name, subsample_label),     stage_1_min_pvalue_quantiles_array)
    save_to_pickle(train_dataset_folder + "preprocess/" + "{}{}stage_2_pvalues_array.pkl".format(params_folder_name, subsample_label),                  stage_2_pvalues_array)

    check_sel_stage_pvalues = load_from_pickle(train_dataset_folder + "preprocess/" + "{}{}sel_stage_pvalues.pkl".format(params_folder_name, subsample_label))
    print("\n\n!!!!!!!!!!!!! check_sel_stage_pvalues - stage_1_min_pvalues_array", np.sum(check_sel_stage_pvalues - stage_1_min_pvalues_array))
    print("!!!!!!!!!!!!! check_sel_stage_pvalues - stage_1_avg_pvalues_array", np.sum(check_sel_stage_pvalues - stage_1_avg_pvalues_array))


def agg_X_raw_X(X, raw_X, params):
    """
    columns in raw_X: [start_hour_of_day (0), end_hour_of_day (1),travel_time (2), travel_distance (3), start_lat (4), start_lon (5),
                       end_lat (6), end_lon (7), day_week (8), start_dur (9), end_dur (10)]

    columns in X: travel_time (0), Euclidan distance (1), distance of start stopp to historical stopps (2),
    distance of end stopp to historical stopps (3), depature time density (4), depature spatiotemporal density (5)                       
    """

    if params.agg_X_raw_X_option == 0:
        X_modified = raw_X
    elif params.agg_X_raw_X_option == 1:  # replaice end (lat, long) with differences in lat and long between end and start stopps
        diff_lat_col = raw_X[:, 6] - raw_X[:, 4]
        diff_lon_col = raw_X[:, 7] - raw_X[:, 5]
        X_modified = raw_X
        X_modified[:, 6] = diff_lat_col
        X_modified[:, 7] = diff_lon_col
    elif params.agg_X_raw_X_option == 2:  # replace day of week (0 to 6) with 0 (work days) and 1 (weekend)
        days = X[:, 8].astype(int)
        # Convert days to binary: 0 for workdays (0-4) and 1 for weekends (5-6)
        binary_days = np.where((days >= 5) & (days <= 6), 1, 0)  # may be useful for ST density
        # Replace the 8th column in X with the binary_days
        X_modified = X
        X_modified[:, 8] = binary_days
    elif params.agg_X_raw_X_option == 3:
        # Extract the day of the week column
        days = X[:, 8].astype(int)
        print("days", np.min(days), np.max(days))
        # Convert to one-hot encoding
        # Using np.eye(7) creates a 7x7 identity matrix, and fancy indexing selects the rows corresponding to each day
        one_hot_days = np.eye(7)[days]
        # Concatenate the columns before the day of the week column, the one-hot encoded day columns, and the columns after the day of the week column
        X_modified = np.concatenate([X[:, :8], one_hot_days, X[:, 9:]], axis=1)

        # X = np.concatenate((raw_X, X[:,8].reshape(-1, 1)), axis = 1)

    # sel_X = X[:,[2,3]] # select columns 2 and 3 corresponding to distance of start stopp to historical stopps and distance of end stopp to historical stopps
    # X_modified = np.concatenate((X_modified, sel_X), axis = 1)

    # X_modified = np.delete(X_modified, 3, axis=1)
    # # X = np.concatenate((raw_X, diff_lat_col.reshape(-1,1), diff_lon_col.reshape(-1,1)), axis = 1)

    # X = raw_X

    # print(raw_X[:5,:])

    # Select the 8th column
    # days = X[:, 8]

    return X_modified





from scipy.stats import entropy


def cross_entropy(p, q, smoothing=1e-15):
    p_smooth = p + smoothing
    p_smooth /= np.sum(p_smooth)
    q_smooth = q + smoothing
    q_smooth /= np.sum(q_smooth)
    return -np.sum(p_smooth * np.log(q_smooth))

def jensen_shannon_divergence(p, q):
    # Compute the average distribution
    m = 0.5 * (p + q)
    epsilon = 0.000000001
    epsilon = 1e-9
    epsilon = 0
    # Calculate the Jensen-Shannon Divergence
    return 0.5 * (cross_entropy(p + epsilon, m + epsilon) + cross_entropy(q + epsilon, m + epsilon))

def jensen_shannon_divergence_KDEs(kde1, kde2, sample_size=1000, grid_size=50):
    """
    Calculate Jensen-Shannon Divergence between two KDEs.

    kde1, kde2: Kernel Density Estimates (KDEs) to compare.
    sample_size: Number of samples to draw from the KDEs.
    grid_size: Number of grid points per dimension for evaluating KDEs.
    """

    # Sample points from the estimated KDEs
    data1 = kde1.resample(sample_size).T
    data2 = kde2.resample(sample_size).T

    # Determine the number of dimensions
    num_dims = data1.shape[1]

    # Generate a grid of points in each dimension
    grid_points = []
    for i in range(num_dims):
        min_val = min(data1[:,i].min(), data2[:,i].min())
        max_val = max(data1[:,i].max(), data2[:,i].max())
        grid = np.linspace(min_val, max_val, grid_size)
        grid_points.append(grid)

    # Create meshgrid for evaluating KDEs at the grid of points
    meshgrid = np.meshgrid(*grid_points)
    grid_points_flat = np.vstack([grid.flatten() for grid in meshgrid])

    # Evaluate KDEs at the grid of points
    p = kde1(grid_points_flat)
    q = kde2(grid_points_flat)

    # Compute JSD between p and q
    jsd = jensen_shannon_divergence(p, q)

    if np.isinf(jsd) or jsd > 10: 
        jsd = 10

    return jsd

def wasserstein_distance_KDEs(kde1, kde2, sample_size=10000):
    # Sample points from the estimated KDEs
    data1 = kde1.resample(sample_size).T
    data2 = kde2.resample(sample_size).T    
    return wasserstein_distance(data1.flatten(), data2.flatten())



def normal_agents_ST_departure_kde_analysis(train_dataset_folder, test_dataset_folder, abnormal_agent_id_list, subsample_label = "", params = None, params_folder_name = ""):

    if abnormal_agent_id_list == None:
        abnormal_agent_id_list = []

    subsampled_filenames = os.listdir(test_dataset_folder + "event_logs/")
    normal_agent_id_list = []
    for idx, parquet_fname in enumerate(subsampled_filenames[:]):
        agent_id = int(parquet_fname.split('/')[-1].split('.')[0])
        if agent_id not in abnormal_agent_id_list:
            normal_agent_id_list.append(agent_id)            
        if len(normal_agent_id_list) > 10:
            break 

    if not os.path.exists(train_dataset_folder + "preprocess/gdf_stopp.pkl"):
        gdf, stopp_df  = retrieve_gdf_stopp(train_dataset_folder)
        save_to_pickle(train_dataset_folder + "preprocess/gdf_stopp.pkl", [gdf, stopp_df])
    else:
        [gdf, stopp_df] = load_from_pickle(train_dataset_folder + "preprocess/gdf_stopp.pkl")

    print("loading loc_coord_dict.pickle")
    if not os.path.exists(train_dataset_folder + 'preprocess/loc_coord_dict.pickle'):
        loc_coord_dict = create_location_coordinates(train_dataset_folder, gdf, stopp_df)
    else:
        # loc_coord_dict = load_from_pickle(train_dataset_folder + 'preprocess/loc_coord_dict.pickle')
        loc_coord_dict = load_from_pickle(train_dataset_folder + 'preprocess/loc_coord_dict.pickle')

    # abnormal_agent_id_list  = retrieve_abnormal_agent_id_list(test_dataset_folder)
    for agent_id in normal_agent_id_list[:]:
        print("!!!!!!!!!!! agent id: ", agent_id )

        train_kdes_file_path = train_dataset_folder + "preprocess/{}KDE/{}{}_KDEs.dill".format(params_folder_name, subsample_label, agent_id)
        train_kdes_file_path = train_kdes_file_path.replace(subsample_label, "full_")
        [train_kde_depature_hour, train_kde_depature_ST, train_date_travel_time_kde, train_ST_Scaler, train_check_kdes_file_path] = load_from_dill(train_kdes_file_path)

        test_kdes_file_path = test_dataset_folder + "preprocess/{}KDE/{}{}_KDEs.dill".format(params_folder_name, subsample_label, agent_id)
        [test_kde_depature_hour, test_kde_depature_ST, test_date_travel_time_kde, test_ST_Scaler, test_check_kdes_file_path] = load_from_dill(test_kdes_file_path)


        # Sample points from the estimated KDEs
        norm_X_ST_train = train_kde_depature_ST.resample(1000).T
        X_ST_train      = train_ST_Scaler.inverse_transform(norm_X_ST_train)
        norm_X_ST_test  = test_kde_depature_ST.resample(1000).T
        X_ST_test       = test_ST_Scaler.inverse_transform(norm_X_ST_test)

        # continue
        if train_kde_depature_ST is not None:
            lat_lon_time_KDE_plot(agent_id, train_kde_depature_ST, train_ST_Scaler, X_ST_train, None, train_dataset_folder, params_folder_name)
        else:
            print("!!!!!!!!!!!! agent id: {} has None deapture ST KDE".format(agent_id))

        if test_kde_depature_ST is not None:
            lat_lon_time_KDE_plot(agent_id, test_kde_depature_ST, test_ST_Scaler, X_ST_test, None, test_dataset_folder, params_folder_name)
        else:
            print("!!!!!!!!!!!! agent id: {} has None deapture ST KDE".format(agent_id))


        # test_parquet_fpath      = test_dataset_folder  + 'event_logs/{}.parquet'.format(agent_id)
        # train_parquet_fpath     = train_dataset_folder + 'event_logs/{}.parquet'.format(agent_id)
        # [a, b] = anomaly_agent_index_dict[agent_id]
        # _, _, _, _, _, _, _, _, X_ST_train, _   = trip_feature_extraction(train_parquet_fpath, train_dataset_folder, test_dataset_folder, loc_coord_dict, gdf, stopp_df, subsample_label, train_phase_boolean = True,  abnormal_agent_id_list = None, anomaly_agent_index_dict = None, params = params, params_folder_name = params_folder_name)
        # agent_2_date_2_his_coord_set_dict       = load_from_pickle(train_dataset_folder + "preprocess/" + "{}{}train_agent_2_date_2_his_coord_set_dict.pkl".format(params_folder_name, subsample_label))
        # _, _, _, _, _, _, _, _,  X_ST_test, _   = trip_feature_extraction(test_parquet_fpath,  train_dataset_folder, test_dataset_folder, loc_coord_dict, gdf, stopp_df, subsample_label, agent_2_date_2_his_coord_set_dict, train_phase_boolean = False, abnormal_agent_id_list = None, anomaly_agent_index_dict = None, params = params, params_folder_name = params_folder_name)
        # [kde_depature_hour, kde_depature_ST, date_travel_time_kde, ST_scaler] = load_from_dill(train_dataset_folder + "preprocess/{}KDE/{}{}_KDEs.dill".format(params_folder_name, subsample_label, agent_id))
        # trip_df_idx_test = np.array(trip_df_idx_test)
        # # print(trip_df_idx_test.shape)
        # starts_within_range   = trip_df_idx_test[:, 0] <= b
        # ends_within_range     = trip_df_idx_test[:, 1] >= a
        # ab_event_indices      = np.where(starts_within_range & ends_within_range)[0]
        
        # if kde_depature_ST is not None:
        #     norm_X_ST_train = ST_scaler.transform(X_ST_train)
        #     norm_X_ST_test  = ST_scaler.transform(X_ST_test)
        #     densities_train = kde_depature_ST(norm_X_ST_train.T)
        #     densities_test  = kde_depature_ST(norm_X_ST_test.T)
        #     print("!!!!!!!!!!!!!! {}: train density: min: {:.4f}, max: {:.4f}, mean: {:.4f}, std: {:.4f}".format(agent_id, np.min(densities_train), np.max(densities_train), np.mean(densities_train), np.std(densities_train)))
        #     print("!!!!!!!!!!!!!! {}: test density: min: {:.4f}, max: {:.4f}, mean: {:.4f}, std: {:.4f}".format(agent_id, np.min(densities_test),  np.max(densities_test),  np.mean(densities_test),  np.std(densities_test)))

            
class JensenShannonDivgTask(object):
    def __init__(self, train_dataset_folder, test_dataset_folder, params_folder_name, subsample_label, subsample_size, filenames, ST_KDE_bool = True):
        self.train_dataset_folder   = train_dataset_folder
        self.test_dataset_folder    = test_dataset_folder 
        self.params_folder_name     = params_folder_name 
        self.subsample_label        = subsample_label 
        self.subsample_size         = subsample_size 
        self.filenames              = filenames
        self.ST_KDE_bool            = ST_KDE_bool

    def __call__(self):
        jsd_KDE_list = []
        for parquet_fname in self.filenames:
            try: 
                agent_id = int(parquet_fname.split('/')[-1].split('.')[0]) # agent ID should be integer format by default
                train_kdes_file_path = self.train_dataset_folder + "preprocess/{}KDE/{}{}_KDEs.dill".format(self.params_folder_name, self.subsample_label, agent_id)
                train_kdes_file_path = train_kdes_file_path.replace(self.subsample_label, "full_")
                [train_kde_depature_hour, train_kde_depature_ST, train_date_travel_time_kde, train_ST_Scaler, train_check_kdes_file_path] = load_from_dill(train_kdes_file_path)

                test_kdes_file_path = self.test_dataset_folder + "preprocess/{}KDE/{}{}_KDEs.dill".format(self.params_folder_name, self.subsample_label, agent_id)
                [test_kde_depature_hour, test_kde_depature_ST, test_date_travel_time_kde, test_ST_Scaler, test_check_kdes_file_path] = load_from_dill(test_kdes_file_path)

                if self.ST_KDE_bool == True: 
                    jsd_KDE = jensen_shannon_divergence_KDEs(train_kde_depature_ST, test_kde_depature_ST)
                else:
                    jsd_KDE = jensen_shannon_divergence_KDEs(train_kde_depature_hour, test_kde_depature_hour)

                print(parquet_fname, jsd_KDE)
                jsd_KDE_list.append(jsd_KDE)
            except Exception as error:
                print(error, parquet_fname)
        return jsd_KDE_list

    def __str__(self):
        return str(self.agent_id_list)




def get_fis_group(fis_folder, num_subsample_group=None):
    group_agent_list = []
    group_list = os.listdir(fis_folder)
    if num_subsample_group is not None:
        num_group = num_subsample_group
    else:
        num_group = len(group_list)
    for i in range(num_group):
        temp_group_folder = group_list[i]
        temp_parquet_list = os.listdir(os.path.join(fis_folder, temp_group_folder))
        temp_agent_list = [np.int64(temp_file.replace('.parquet', '')) for temp_file in temp_parquet_list]
        group_agent_list.append(temp_agent_list)
    return group_agent_list

def get_trajectory_parquet_files(directory):
    agent_trajectory_path_dict = dict()
    for root, _, files in os.walk(directory):
        if root in [os.path.join(directory, 'preprocess'), os.path.join(directory, 'plots')]:
            continue
        for file in files:
            if file.endswith('.parquet'):
                temp_head_str = file.replace('.parquet', '')
                if temp_head_str.isdigit():
                    agent_trajectory_path_dict[np.int64(temp_head_str)] = os.path.join(root, file)
    return agent_trajectory_path_dict


def copy_features(train_dataset_folder, test_dataset_folder, first_conf_params_folder, params_folder_name, subsample_label):
    source_train_folder = os.path.join(train_dataset_folder, "preprocess", "{}".format(first_conf_params_folder))
    source_test_folder = os.path.join(test_dataset_folder, "preprocess", "{}".format(first_conf_params_folder))
    target_train_folder = os.path.join(train_dataset_folder, "preprocess", "{}".format(params_folder_name))
    target_test_folder  = os.path.join(test_dataset_folder, "preprocess", "{}".format(params_folder_name))
    shutil.copyfile(os.path.join(source_train_folder, '{}train_agent_id_list.pkl'.format(subsample_label)),
                    os.path.join(target_train_folder, '{}train_agent_id_list.pkl'.format(subsample_label)))
    shutil.copyfile(os.path.join(source_train_folder, '{}train_id_agent_X.pkl'.format(subsample_label)),
                    os.path.join(target_train_folder, '{}train_id_agent_X.pkl'.format(subsample_label)))
    shutil.copyfile(os.path.join(source_train_folder, '{}train_X.pkl'.format(subsample_label)),
                    os.path.join(target_train_folder, '{}train_X.pkl'.format(subsample_label)))
    shutil.copyfile(os.path.join(source_train_folder, '{}train_raw_X.pkl'.format(subsample_label)),
                    os.path.join(target_train_folder, '{}train_raw_X.pkl'.format(subsample_label)))
    shutil.copyfile(os.path.join(source_train_folder, '{}train_trip_df_idx.pkl'.format(subsample_label)),
                    os.path.join(target_train_folder, '{}train_trip_df_idx.pkl'.format(subsample_label)))
    shutil.copyfile(os.path.join(source_train_folder, '{}train_agent_2_his_coord_set_dict.pkl'.format(subsample_label)),
                    os.path.join(target_train_folder, '{}train_agent_2_his_coord_set_dict.pkl'.format(subsample_label)))
    shutil.copyfile(os.path.join(source_train_folder, '{}train_agent_2_date_2_his_coord_set_dict.pkl'.format(subsample_label)),
                    os.path.join(target_train_folder, '{}train_agent_2_date_2_his_coord_set_dict.pkl'.format(subsample_label)))
    shutil.copyfile(os.path.join(source_train_folder, '{}train_agent_2_date_2_stopp_coords_dict.pkl'.format(subsample_label)),
                    os.path.join(target_train_folder, '{}train_agent_2_date_2_stopp_coords_dict.pkl'.format(subsample_label)))

    shutil.copyfile(
        os.path.join(source_test_folder, '{}test_agent_id_list.pkl'.format(subsample_label)),
        os.path.join(target_test_folder, '{}test_agent_id_list.pkl'.format(subsample_label)))
    shutil.copyfile(
        os.path.join(source_test_folder, '{}test_X.pkl'.format(subsample_label)),
        os.path.join(target_test_folder, '{}test_X.pkl'.format(subsample_label)))
    shutil.copyfile(
        os.path.join(source_test_folder, '{}test_raw_X.pkl'.format(subsample_label)),
        os.path.join(target_test_folder, '{}test_raw_X.pkl'.format(subsample_label)))
    shutil.copyfile(
        os.path.join(source_test_folder, '{}test_trip_df_idx.pkl'.format(subsample_label)),
        os.path.join(target_test_folder, '{}test_trip_df_idx.pkl'.format(subsample_label)))
    shutil.copyfile(
        os.path.join(source_test_folder, '{}test_id_agent_X_array.pkl'.format(subsample_label)),
        os.path.join(target_test_folder, '{}test_id_agent_X_array.pkl'.format(subsample_label)))
    shutil.copyfile(
        os.path.join(source_test_folder, '{}test_trip_start_end_datetimes_df.pkl'.format(subsample_label)),
        os.path.join(target_test_folder, '{}test_trip_start_end_datetimes_df.pkl'.format(subsample_label)))




def pr_plots(train_dataset_folder, test_dataset_folder, test_parquet_fpath_list, abnormal_agent_id_list, anomaly_agent_index_dict, ad_label, subsample_label, trip_extraction_boolean,  params, params_folder_name, fis_group_agent_list):
    
    
    plot_folder = os.path.join(test_dataset_folder, 'preprocess', params_folder_name, 'plots')
    create_folder_if_not_exists(plot_folder)
    agent_2_npss_diff_and_trip_indices_dict = load_from_pickle(test_dataset_folder + "preprocess/" + "{}{}agent_2_npss_difference_dict.pkl".format(params_folder_name,
                                                                                            subsample_label))
    score_list = []
    label_list = []
    for temp_group_list in fis_group_agent_list:
        temp_group_len = len(temp_group_list)
        temp_group_score_list = []
        temp_group_label_list = []
        for i in range(temp_group_len):
            temp_agent = temp_group_list[i]
            if temp_agent in agent_2_npss_diff_and_trip_indices_dict:
                temp_group_score_list.append(agent_2_npss_diff_and_trip_indices_dict[temp_agent][0])
                if temp_agent in abnormal_agent_id_list:
                    temp_group_label_list.append(1)
                else:
                    temp_group_label_list.append(0)
        if len(temp_group_score_list) > 0:
            temp_group_score = np.array(temp_group_score_list)
            temp_group_label = np.array(temp_group_label_list)
            score_list.append(temp_group_score)
            label_list.append(temp_group_label)
    # normal_concat_score_list = []
    # for temp_group_score in score_list:
    #     normal_concat_score_list.append((temp_group_score - temp_group_score.min()) / (temp_group_score.max() - temp_group_score.min()))
    # normal_concat_score = np.concatenate(normal_concat_score_list)
    concat_unnormal_score = np.concatenate(score_list)
    concat_normal_score = (concat_unnormal_score - concat_unnormal_score.min()) / (concat_unnormal_score.max() - concat_unnormal_score.min())
    labels = np.concatenate(label_list)
    mean = concat_normal_score.mean()
    std = concat_normal_score.std()
    threashold = mean + 3.5 * std
    pred_mean_std = np.where(concat_normal_score > threashold, 1, 0)
    pred_group_first = np.where(concat_normal_score == 1, 1, 0)
    fontsize = 14


    # concat_normal
    lr_precision, lr_recall, thresholds = precision_recall_curve(labels, concat_normal_score)
    aupr = auc(lr_recall, lr_precision)
    plt.plot(lr_recall, lr_precision, marker='.', lw=2, label=aupr)
    plt.xlabel('Recall', fontsize=fontsize)
    plt.ylabel('Precision', fontsize=fontsize)
    plt.title('Agent Level PR Curve')
    plt.legend(loc='upper right')
    plt.tight_layout()
    os.path.join(test_dataset_folder, 'plots')
    plt.savefig(os.path.join(plot_folder, "Agent-PR.png"))
    plt.show()
    plt.close('all')

    # roc mean + 3 * std
    fpr, tpr, _ = roc_curve(labels, concat_normal_score)
    auroc = auc(fpr, tpr)
    plt.plot(fpr, tpr, marker='.', lw=2, label=auroc)
    plt.xlabel('FPR', fontsize=fontsize)
    plt.ylabel('TPR', fontsize=fontsize)
    plt.title('Agent Level ROC Curve')
    plt.legend(loc='upper right')
    plt.tight_layout()
    os.path.join(test_dataset_folder, 'plots')
    plt.savefig(os.path.join(plot_folder, "Agent-ROC.png"))
    plt.show()
    plt.close('all')


    au_dict = {'aupr':aupr, 'auroc':auroc}
    save_to_pickle(os.path.join(test_dataset_folder, 'preprocess', params_folder_name, 'au_dict.pkl'), au_dict)


class cls_params: 
    def __init__(self):
        self.bw_ST_KDE = 0.1
        self.K_KNN = 5
        self.K_LOF = 20
        self.ab_trip_removal_ST_KDE_bool = False
        self.raw_X_bool = False
        self.bool_combine_train_test = False
        self.agg_bool = False
        self.agg_X_raw_X_option = 0
    def to_str(self):
        return "bw-{}-K-{}-{}-rawX-{}-agg-{}-STKDE-{}-comb-{}-aggX-{}".format(self.bw_ST_KDE, self.K_KNN, self.K_LOF, int(self.raw_X_bool), \
                                                                        int(self.agg_bool), int(self.ab_trip_removal_ST_KDE_bool), int(self.bool_combine_train_test), self.agg_X_raw_X_option)


import itertools
import cv2
@profile
def main():
# terminal 1: 1k, bw-0.1-K-10-25-rawX-0-STKDE-0-comb-0
# terminal 2: 2k, bw-0.1-K-10-25-rawX-0-STKDE-0-comb-0
    
    time1 = time.time()
    datasets = {'singapore-T2-new_format-test': ['/home/jxl220096/data/hay/new_format/trial2/singapore/train_stops/', 
                                                '/home/jxl220096/data/hay/new_format/trial2/singapore/test_stops/'],
                'sanfrancisco-T2-new_format-test': ['/home/jxl220096/data/hay/new_format/trial2/sanfrancisco/train_stops/', 
                                                '/home/jxl220096/data/hay/new_format/trial2/sanfrancisco/test_stops/'],
                'knoxville-T2-new_format-test': ['/home/jxl220096/data/hay/new_format/trial2/knoxville/train_stops/', 
                                                '/home/jxl220096/data/hay/new_format/trial2/knoxville/test_stops/'],
                'losangeles-T2-new_format-test': ['/home/jxl220096/data/hay/new_format/trial2/losangeles/train_stops/', 
                                                '/home/jxl220096/data/hay/new_format/trial2/losangeles/test_stops/']}

    fis_folder = {
                'singapore-T2-new_format-test': '/home/jxl220096/data/hay/new_format/trial2/singapore/test_stops_valsplit',
                'sanfrancisco-T2-new_format-test': '/home/jxl220096/data/hay/new_format/trial2/sanfrancisco/test_stops_valsplit',
                'knoxville-T2-new_format-test': '/home/jxl220096/data/hay/new_format/trial2/knoxville/test_stops_valsplit',
                'losangeles-T2-new_format-test': '/home/jxl220096/data/hay/new_format/trial2/losangeles/test_stops_valsplit'}

    label_dict = {-1: "full_", 200: "200_", 500: "500_", 800: "800_", 1000: "1k_", 2000: "2k_", 5000: "5k_", 10000: "10k_", 50000: "50k_", 100000: "100k_", 150000: "150k_", 200000: "200k_"}

    K_KNN = [5, 10][1:2]
    K_LOF = [20, 25][1:2]
    agg_X_raw_X_options = [0, 1, 2, 3]#[1:2]
    params = cls_params()

    bw_grid_ST_KDE = [0.1]
    hyper_param_combinations = list(itertools.product(bw_grid_ST_KDE, K_KNN, K_LOF, agg_X_raw_X_options))
    hyper_param_combinations = hyper_param_combinations[1:]
    subsample_size_set       = [50000]

    dataset_names = ['singapore-T2-new_format-test'] # 'singapore-T2-new_format-test', 'sanfrancisco-T2-new_format-test',  'losangeles-T2-new_format-test', 'knoxville-T2-new_format-test'
    anomaly_detector_labels  = ["ensemble_stage_1_avg_pvalues", "ensemble_stage_1_min_pvalues", "LOF", "KNN", "ensemble_stage_2_pvalues"][:1]
    bool_trial_2 = True

    # return
    for subsample_size in subsample_size_set:
        subsample_label = label_dict[subsample_size]
        print("subsample_size", subsample_size)

        for dataset_name in dataset_names: 
            # print("!!!!!!!!!!!! Dataset name: ", dataset_namec)
            [train_dataset_folder, test_dataset_folder] = datasets[dataset_name]
            # train_dataset_folder = test_dataset_folder

            abnormal_agent_id_list      = retrieve_abnormal_agent_id_list(test_dataset_folder, bool_trial_2)
            anomaly_agent_index_dict    = gen_anomaly_agent_index_dict(test_dataset_folder, bool_trial_2)

            # list of lists of 300 agents, where each 300 agents corresponding to a fis group [[300 agents], [300 agents],...]
            fis_group_agent_list = get_fis_group(fis_folder[dataset_name], num_subsample_group=None)
            all_agents_list = sum(fis_group_agent_list, []) # combine all the fis groups to get one list of all the agents in all the fis groups
            all_agents_list = all_agents_list[:50000]
            train_parquet_fpath_list    = [os.path.join(train_dataset_folder, str(agent)+'.parquet')  for agent in all_agents_list]
            test_parquet_fpath_list     = [os.path.join(test_dataset_folder, str(agent)+'.parquet')  for agent in all_agents_list]

            first_conf_params_folder = None

            for (bw_ST_KDE, K_KNN, K_LOF, agg_X_raw_X_option) in hyper_param_combinations:
                # (bw_ST_KDE, K_KNN, K_LOF) = (0.1, 10, 25)
                print("!!!!!! (bw_ST_KDE, K_KNN, K_LOF)", bw_ST_KDE, K_KNN, K_LOF)
                params.bw_ST_KDE  = bw_ST_KDE
                params.K_KNN      = K_KNN
                params.K_LOF      = K_LOF
                params.agg_X_raw_X_option = agg_X_raw_X_option
                if params is not None: 
                    params_folder_name = params.to_str() + "/"
                else:
                    params_folder_name = ""
                    
                create_folder_if_not_exists(train_dataset_folder + "preprocess/" + "{}".format(params_folder_name))
                create_folder_if_not_exists(test_dataset_folder + "preprocess/" + "{}".format(params_folder_name))
                save_to_pickle(train_dataset_folder + "preprocess/" + "{}{}experiment_setting.pkl".format(params_folder_name, subsample_label), [params, subsample_size, subsample_label, hyper_param_combinations, anomaly_detector_labels])
                print("!!!!!!!!!!!!!! ", params_folder_name)

                for ad_label in anomaly_detector_labels:
                    if first_conf_params_folder is None:
                        first_conf_params_folder = params_folder_name
                        trip_extraction_boolean = True
                    else:
                        # the files in first_conf_params_folder_name to the new folder: params_folder_name
                        copy_features(train_dataset_folder, test_dataset_folder, first_conf_params_folder, params_folder_name, subsample_label)
                        trip_extraction_boolean = False
                    
                    detection_train(train_dataset_folder, test_dataset_folder, train_parquet_fpath_list, abnormal_agent_id_list, \
                                    anomaly_agent_index_dict, ad_label, subsample_label, trip_extraction_boolean, params, params_folder_name)
                    detection_test(train_dataset_folder, test_dataset_folder, test_parquet_fpath_list, abnormal_agent_id_list, \
                                    anomaly_agent_index_dict, ad_label, subsample_label, trip_extraction_boolean,  params, params_folder_name)
                    # pr_plots(train_dataset_folder, test_dataset_folder, test_parquet_fpath_list, abnormal_agent_id_list, \
                    #                 anomaly_agent_index_dict, ad_label, subsample_label, trip_extraction_boolean,  params, params_folder_name, fis_group_agent_list)

    time2 = time.time()
    print_time(time2 - time1)    

def evaluation_with_without_features(train_dataset_folder, test_dataset_folder, params_folder_name, subsample_label, ad_label):
    true_anomaly_label_list         = load_from_pickle(test_dataset_folder + "preprocess/" + "{}{}true_anomaly_label_list.pkl".format(params_folder_name, subsample_label))
    npss_diff_list_without_features = load_from_pickle(test_dataset_folder + "preprocess/" + "{}{}npss_diff_list_without_features.pkl".format(params_folder_name, subsample_label))
    npss_diff_list_with_features    = load_from_pickle(test_dataset_folder + "preprocess/" + "{}{}npss_diff_list_with_features.pkl".format(params_folder_name, subsample_label))
    npss_diff_list_with_3_features  = load_from_pickle(test_dataset_folder + "preprocess/" + "{}{}npss_diff_list_with_3_features.pkl".format(params_folder_name, subsample_label))
    # agent_aupr = pr_plot_proc(train_dataset_folder, np.array(true_anomaly_label_list), np.array(npss_diff_list), subsample_label, ad_label, 'Agent', params_folder_name)

    lr_precision, lr_recall, thresholds = precision_recall_curve(true_anomaly_label_list, npss_diff_list_with_3_features)
    aupr = auc(lr_recall, lr_precision)
    fontsize = 14
    plt.plot(lr_recall, lr_precision, marker='.', lw=2, label= "Detection with travel- and min-distance features (AUPR score: {:.4f})".format(aupr))

    lr_precision, lr_recall, thresholds = precision_recall_curve(true_anomaly_label_list, npss_diff_list_with_features)
    aupr = auc(lr_recall, lr_precision)
    fontsize = 14
    plt.plot(lr_recall, lr_precision, marker='.', lw=2, label= "Detection with travel-distance feature (AUPR score: {:.4f})".format(aupr))

    lr_precision, lr_recall, thresholds = precision_recall_curve(true_anomaly_label_list, npss_diff_list_without_features)
    aupr = auc(lr_recall, lr_precision)
    fontsize = 14
    plt.plot(lr_recall, lr_precision, marker='.', lw=2, label= "Detection without travel- and min-distance features (AUPR score: {:.4f})".format(aupr))

    plt.xlabel('Recall', fontsize=fontsize)
    plt.ylabel('Precision', fontsize=fontsize)
    plt.title('Find Method: PR Curve for Agent-level Detection', pad=70)
    # plt.legend(loc='upper right')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.30))
    plt.tight_layout()
    create_folder_if_not_exists(train_dataset_folder + "preprocess/plots/{}".format(params_folder_name))
    plt.savefig(os.path.join(train_dataset_folder, "preprocess/plots/{}{}Detector-{}-PR-Comparison.png".format(params_folder_name, subsample_label, ad_label)))
    plt.show()
    plt.close('all')

def pr_plots_site_visit_apr_12(train_dataset_folder, test_dataset_folder, params_folder_name, subsample_label, ad_label, dataset_name):

    dataset_name_2_prec_recall_list = {'Kitware':    [[0.0,1.0],[0.08,1.0],[0.081,0.88],[0.2,0.94],[0.201,0.9],[0.22,0.91],[0.24,0.7],[0.25,0.72],[0.251,0.7],[0.27,0.72],[0.271,0.66],[0.28,0.67],[0.281,0.56],[0.34,0.58],[0.35,0.53],[0.44,0.54],[0.441,0.48],[0.46,0.5],[0.461,0.46],[0.47,0.47],[0.471,0.38],[0.49,0.36],[0.5,0.36],[0.51,0.32],[0.61,0.32],[0.69,0.21],[0.691,0.15],[0.73,0.14],[0.96,0.04],[1.0,0.0]],
                                       'Novateur':   [[0.0,1.0],[0.001,0.0],[0.01,0.5],[0.02,0.67],[0.03,0.75],[0.05,0.67],[0.06,0.69],[0.07,0.75],[0.08,0.78],[0.089,0.8],[0.09,0.73],[0.1,0.75],[0.11,0.64],[0.13,0.69],[0.131,0.65],[0.17,0.7],[0.18,0.6],[0.2,0.62],[0.201,0.5],[0.21,0.52],[0.22,0.38],[0.23,0.4],[0.24,0.36],[0.241,0.35],[0.25,0.3],[0.27,0.32],[0.271,0.22],[0.46,0.06],[1.0,0.0]],
                                       'Baseline': [[0.0,1.0],[0.001,0.0],[0.03,0.1],[0.04,0.1],[1.0,0.0]],
                                       'L3Harris':   [[0.0,1.0],[0.11,1.0],[0.111,0.84],[0.14,0.87],[0.141,0.72],[0.15,0.74],[0.16,0.56],[0.21,0.56],[0.211,0.5],[0.22,0.52],[0.221,0.48],[0.24,0.5],[0.241,0.4],[0.26,0.38],[0.261,0.36],[0.28,0.38],[0.3,0.28],[0.32,0.28],[0.321,0.18],[0.42,0.18],[0.421,0.14],[0.54,0.06],[1.0,0.0]]
                                       }

    dataset_name_2_aupr = {'Kitware':     0.4405,
                            'Novateur':   0.1945,
                            'Baseline': 0.0121,
                            'L3Harris':   0.2675
                            }
    recall_prec_list = dataset_name_2_prec_recall_list[dataset_name]
    recall_prec_arr  = np.array(recall_prec_list)
    true_anomaly_label_list         = load_from_pickle(test_dataset_folder + "preprocess/" + "{}{}true_anomaly_label_list.pkl".format(params_folder_name, subsample_label))
    npss_diff_list_with_features    = load_from_pickle(test_dataset_folder + "preprocess/" + "{}{}npss_diff_list_with_features.pkl".format(params_folder_name, subsample_label))
    # agent_aupr = pr_plot_proc(train_dataset_folder, np.array(true_anomaly_label_list), np.array(npss_diff_list), subsample_label, ad_label, 'Agent', params_folder_name)

    plt.plot(recall_prec_arr[:,0], recall_prec_arr[:,1], marker='.', lw=2, label= "Trial-1 Find Method (AUPR score: {:.4f})".format(dataset_name_2_aupr[dataset_name]))
    
    lr_precision, lr_recall, thresholds = precision_recall_curve(true_anomaly_label_list, npss_diff_list_with_features)
    aupr = auc(lr_recall, lr_precision)
    fontsize = 14
    plt.plot(lr_recall, lr_precision, marker='.', lw=2, label= "Trial-2 Find Method (AUPR score: {:.4f})".format(aupr))

    plt.xlabel('Recall', fontsize=fontsize)
    plt.ylabel('Precision', fontsize=fontsize)
    plt.title('PR Curve for Agent-level Detection', pad=5)
    plt.legend(loc='upper right')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.30))
    plt.tight_layout()
    create_folder_if_not_exists(test_dataset_folder + "preprocess/plots/{}".format(params_folder_name))
    plt.savefig(os.path.join(test_dataset_folder, "preprocess/plots/{}{}Detector-{}-PR-Comparison-site-visit.png".format(params_folder_name, subsample_label, ad_label)))
    plt.show()
    plt.close('all')


def draw_plots():
    datasetnames = ['Knoxville', 'SanFrancisco', 'LosAngeles']
    folders = ["/home/kxj200023/data/dataset/trial2/metrics/baseline", "/home/kxj200023/data/dataset/trial2/metrics/l3harris", "/home/kxj200023/data/dataset/trial2/metrics/novateur"]
    for i in range(3):
        datasetname = datasetnames[i]
        folder = folders[i]
        info_table = pd.read_csv(os.path.join(folder, "anomaly_detection/tables/agents.csv"))
        fontsize = 14
        # pr
        lr_recall = info_table['recall'].values
        lr_precision = info_table['precision'].values
        aupr = auc(lr_recall, lr_precision)
        plt.plot(lr_recall, lr_precision, marker='.', lw=2, label=aupr)
        plt.xlabel('Recall', fontsize=fontsize)
        plt.ylabel('Precision', fontsize=fontsize)
        plt.title(datasetname + ' Agent Level PR Curve')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join("/home/kxj200023/data/dataset/trial2/metrics", datasetname + "-Agent-PR.png"))
        plt.show()
        plt.close('all')
        # roc
        tp = info_table['tp'].values
        tn = info_table['tn'].values
        fp = info_table['fp'].values
        fn = info_table['fn'].values
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        auroc = auc(fpr, tpr)
        plt.plot(fpr, tpr, marker='.', lw=2, label=auroc)
        plt.xlabel('FPR', fontsize=fontsize)
        plt.ylabel('TPR', fontsize=fontsize)
        plt.title(datasetname + ' Agent Level ROC Curve')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join("/home/kxj200023/data/dataset/trial2/metrics", datasetname + "-Agent-ROC.png"))
        plt.show()
        plt.close('all')

def post_analysis(train_dataset_folder, test_dataset_folder, subsample_label, params_folder_name):

    train_trip_multi_ad_scores              = load_from_pickle(train_dataset_folder + "preprocess/" + "{}{}trip_multi_ad_scores.pkl".format(params_folder_name, subsample_label))
    train_multi_ad_scores_quantiles_array   = load_from_pickle(train_dataset_folder + "preprocess/" + "{}{}train_multi_ad_scores_quantiles_array.pkl".format(params_folder_name, subsample_label))
    stage_1_min_pvalue_quantiles_array      = load_from_pickle(train_dataset_folder + "preprocess/" + "{}{}stage_1_min_pvalue_quantiles_array.pkl".format(params_folder_name, subsample_label))
    stage_2_pvalues_array                   = load_from_pickle(train_dataset_folder + "preprocess/" + "{}{}stage_2_pvalues_array.pkl".format(params_folder_name, subsample_label))
    train_trip_multi_ad_scores_array        = np.array(train_trip_multi_ad_scores).T

    test_multi_ad_stage_scores          = load_from_pickle(test_dataset_folder + "preprocess/"  + "{}{}test_multi_ad_stage_scores.pkl".format(params_folder_name, subsample_label))
    test_multi_ad_stage_1_pvalues_array = load_from_pickle(test_dataset_folder + "preprocess/"  + "{}{}test_multi_ad_stage_1_pvalues_array.pkl".format(params_folder_name, subsample_label))
    test_stage_2_pvalues_array          = load_from_pickle(test_dataset_folder + "preprocess/"  + "{}{}test_stage_2_pvalues_array.pkl".format(params_folder_name, subsample_label))
    [test_X, test_y]                    = load_from_pickle(train_dataset_folder + "preprocess/" + "{}{}test_X_y.pkl".format(params_folder_name, subsample_label))
    test_multi_ad_stage_scores_array    = np.array(test_multi_ad_stage_scores).T
    
    two_histograms_plot(list(test_multi_ad_stage_scores_array[test_y==0,3]), list(test_multi_ad_stage_scores_array[test_y==1,3]), train_dataset_folder + "preprocess/plots/{}{}test_cblof_scores_histogram.png".format(params_folder_name, subsample_label),                ["all", "abnormal agents"], "Test: CBLOF Scores")
    print(np.mean(list(test_multi_ad_stage_scores_array[test_y==0,3])))
    print(np.mean(list(test_multi_ad_stage_scores_array[test_y==1,3])))
    print("test_multi_ad_stage_scores_array", test_multi_ad_stage_scores_array.shape)
    print("test_y", test_y.shape)
    anomaly_detector_labels = ["LOF", "KNN", "SelKNN", "CBLOF"]
    for i, ad_label in enumerate(anomaly_detector_labels):
        trip_aupr  = pr_plot_proc(train_dataset_folder, test_y, test_multi_ad_stage_scores_array[:,i], subsample_label, ad_label, 'Trip', params_folder_name)
        print(ad_label, trip_aupr)

def delete_files_in_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # List all files and directories in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                # If it's a file, delete it
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                # If it's a directory, you can optionally extend this to remove the directory and its contents
                elif os.path.isdir(file_path):
                    # os.rmdir(file_path)  # Use for empty directories
                    # shutil.rmtree(file_path)  # Use to delete a directory and all its contents
                    pass  # Currently, do nothing with directories
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        print(f"The folder {folder_path} does not exist.")
            
def genereate_side_by_side_plots(train_dataset_folder, ad_label = 'ensemble', subsample_label = "", params_folder_name = ""):
    images = []
    labels = ["travel_time", 'euclidean_distance', 'dist_to_start_stopp', 'dist_to_end_stopp', \
                'depature_hour_density', 'depature_ST_density', 'start_stopp_duration', 'end_stopp_duration', 'date_travel_distance',\
                    'lof_scores', 'knn_scores', '{}_sel_stage_pvalues'.format(ad_label), 'stage_1_min_pvalues_array', 'stage_2_pvalues_array']
    # 'sel_knn_scores', 'cblof_scores', 
    for label in labels:
        images.append(cv2.imread(train_dataset_folder + "preprocess/plots/{}raw/{}train_{}_histogram.png".format(params_folder_name, subsample_label, label)))
        images.append(cv2.imread(train_dataset_folder + "preprocess/plots/{}raw/{}test_{}_histogram.png".format(params_folder_name, subsample_label, label)))
        # print(train_dataset_folder + "preprocess/plots/{}raw/{}train_{}_histogram.png".format(params_folder_name, subsample_label, label))
    # Concatenate images in pairs to form four rows
    # print("images", len(images))
    # print(images)
    rows = [np.hstack(images[i:i+2]) for i in range(0, len(images), 2)]
    # Concatenate rows to form the final grid
    grid = np.vstack(rows)
    # Save the concatenated image
    cv2.imwrite(train_dataset_folder + 'preprocess/plots/{}{}{}_train_test_side_by_side_histograms.jpg'.format(params_folder_name, subsample_label, ad_label), grid)
    print(train_dataset_folder + 'preprocess/plots/{}{}{}_train_test_side_by_side_histograms.jpg'.format(params_folder_name, subsample_label, ad_label))


if __name__ == "__main__":
    main()


