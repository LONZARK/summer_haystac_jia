import copy
try:
    import faiss
except ImportError:
    faiss = None  # or handle it in another way

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


# Function to get current process memory usage
def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss  # Return the Resident Set Size (RSS) in bytes

# Decorator function to profile memory usage of a function
def profile(func):
    def wrapper(*args, **kwargs):
        mem_before = process_memory()  # Memory before function execution
        result = func(*args, **kwargs)  # Execute the function
        mem_after = process_memory()  # Memory after function execution
        print("{}: Consumed memory: {:.2f} MiB".format(
            func.__name__,
            (mem_after - mem_before) / (1024 ** 2)))  # Convert bytes to MiB and print

        return result
    return wrapper

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in sorted(fs):
            if f.endswith('.parquet'):
                fullname = os.path.join(root, f)
                yield fullname

def getAllFile(base):
    result = []
    for root, ds, fs in os.walk(base):
        for f in sorted(fs):
            if f.endswith('.parquet'):
                fullname = os.path.join(root, f)
                result.append(fullname)
    return result


def diff_hour(start,end):
    start_min = int(start/60)
    end_min = int(end/60)
    start_hour = start_min/60
    end_hour = end_min/60
    result = end_hour - start_hour
    return result

def str_pointlist(str):
    pointlist = []
    str = str.replace("LINESTRING (","").replace(")","").replace(", ",",")
    str_list =str.split(",")
    for item in str_list:
        pair_list = item.split(" ")
        pointlist.append([float(pair_list[1]),float(pair_list[0])])
    return pointlist

def str_linestr(str):
    line_list = []
    str = str.replace("LINESTRING (","").replace(")","").replace(", ",",")
    str_list =str.split(",")
    for item in str_list:
        pair_list = item.split(" ")
        line_list.append([float(pair_list[0]),float(pair_list[1])])
    linestr = LineString(line_list)
    return linestr

def pre_tripdatetime(file_name, output_filename):
    trip_datetime = pd.read_csv(file_name)
    # trip_datetime['start_datetime'] = trip_datetime['start_datetime'].values.astype(str)
    # trip_datetime['end_datetime'] = trip_datetime['end_datetime'].values.astype(str)
    trip_datetime['weekday'] = trip_datetime['start_datetime'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').weekday())
    trip_datetime['dayofweek'] = trip_datetime['weekday'].apply(lambda x: 0 if x<4.5 else 1)
    trip_datetime['start_time'] = trip_datetime['start_datetime'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').time())
    trip_datetime['end_time'] = trip_datetime['end_datetime'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').time())
    trip_datetime['start_date'] = trip_datetime['start_datetime'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').date())
    trip_datetime['end_weekday'] = trip_datetime['end_datetime'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').weekday())
    trip_datetime['end_dayofweek'] = trip_datetime['end_weekday'].apply(lambda x: 0 if x<4.5 else 1)
    trip_datetime.to_csv(output_filename)

def graph_freq(file_name, save_file_name):
    trans_dict = np.load(file_name, allow_pickle=True)[()]
    freq_dict = dict()
    for point_key in trans_dict.keys():
        ori_point_dict = trans_dict[point_key]
        sum_value = sum(ori_point_dict.values())
        point_freq_dict = dict()
        for trans_key in ori_point_dict.keys():
            point_freq_dict[trans_key] = ori_point_dict[trans_key] / sum_value
        freq_dict[point_key] = point_freq_dict
    np.save(save_file_name, freq_dict)


import os 




# Convert the sets to numpy arrays for FAISS
def build_faiss_index(his_coord_array):
    """
    INPUT
    his_coord_array: (n,2), each row is one coordinate (latitude, longitude)

    OUTPUT
    index: the FAISS index of his_coord_array. 

    """

    if isinstance(his_coord_array, set):
        his_coord_array = np.array(list(his_coord_array))
        
    his_coord_array = his_coord_array.astype('float32')
    
    # Dimension of the vectors
    d = his_coord_array.shape[1]
    
    # ngpus = faiss.get_num_gpus()
    # print("number of GPUs:", ngpus)    
    # # Build and train an index
    # cpu_index = faiss.IndexFlatL2(d)  # Use the L2 norm (Euclidean distance)
    # faiss_index = faiss.index_cpu_to_all_gpus(  # build the index
    #     cpu_index
    # )    

    faiss_index = faiss.IndexFlatL2(d)

    faiss_index.add(his_coord_array)  # Add the his_coords vectors to the index
    return faiss_index 


def find_nearest_neighbor_distances_faiss(query_coord_array, his_coord_array, index = None):
    """
    INPUT
    query_coord_array: a (n,2) array of query coordinates for the rows (latitude, longitude)
    his_coord_array: a (m,2) array of histoical coordinates for the rows (latitude, longitude)
    index: the index of his_coord_set is trained before. In this case, it ignores his_coord_set within this function

    OUTPUT
    nn_distances_list: a list of nn distances corresponding to the rows in query_coord_array
    """
    if index == None: 
        index = build_faiss_index(his_coord_array)
    query_coords_array = query_coord_array.astype('float32')
    # Search the index for the nearest neighbors of coords
    k    = 1  # Number of nearest neighbors to find
    D, _ = index.search(query_coords_array, k)  # D contains the distances of nearest neighbors (n, 1)
    
    # Convert distances to a more convenient format (e.g., a list)
    nn_distances_list = D.flatten().tolist()   # Since k=1, each query point has only one nearest neighbor, so we flatten the array (n,1) to (n,), and then to a list
    
    # Create a mapping from coords to their nearest neighbor distances
    # coord_2_nearest_neighbor_distances_dict = {tuple(query_coords_array[i]): distances[i] for i in range(query_coords_array.shape[0])}
    
    return nn_distances_list, index


def find_nearest_neighbor_indices_faiss(query_coord_array, data_coord_array, index = None, k = 1):
    """
    INPUT
    query_coord_set: a set of query coordinate tuples (latitude, longitude)
    his_coord_set: a set of histoical coordinate tupples (latitude, longitude)
    index: the index of his_coord_set is trained before. In this case, it ignores his_coord_set within this function

    We need to convert query_coord_set and his_coord_set to arrays first in order to use FAISS. 

    OUTPUT
    I: array (n,1) of indices

    """
    if index is None: 
        index = build_faiss_index(data_coord_array)

    # Ensure coords is a float32 numpy array
    query_coord_array = query_coord_array.astype('float32')

    # Search the index for the nearest neighbors of coords
    _, I = index.search(query_coord_array, k)  # I contains the indices of nearest neighbors for each point

    # I is already an array of indices, so you can return it directly
    return I, index


def get_hos_agent_stopp_dict(dataset_folder):
    """
    This function aims to read a set of JSON files containing data about HOS events for different agents, and 
    then compile a dictionary that maps each agent to a list of unique locations associated with their stop points. 

    INPUT
    dataset_folder: the folder of the HOS files 

    OUTPUT
    hos_stopp_dict: {hos string ID: list of stopps}

    """
    
    hos_file_list = os.listdir(dataset_folder + 'HOS/')
    hos_agents_dict = get_hos_agents(dataset_folder + 'HOS/')
    json_files = [file for file in hos_file_list if file.startswith('hos_') and file.endswith('.json')]
 
    hos_sp_dict = {}
    for json_file in json_files:
        file_path = os.path.join(dataset_folder + 'HOS/', json_file)
        with open(file_path, 'r') as file:
            HOS_data = json.load(file)
 
        HOS_sp_location = []
        for i in range(len(HOS_data['events'])):
            HOS_sp_location.append(HOS_data['events'][i]['location'])
        HOS_sp_location        = remove_duplicates(HOS_sp_location)
        hos_sp_dict[hos_agents_dict[json_file.replace('hos_', '').replace('.json', '')]] = []
        hos_sp_dict[hos_agents_dict[json_file.replace('hos_', '').replace('.json', '')]] = HOS_sp_location
    return hos_sp_dict 

def remove_duplicates(ids):
    seen = set()
    result = []
    for id in ids:
        if id not in seen:
            result.append(id)
            seen.add(id)
    return result



def create_location_coordinates(dataset_dir, gdf, stopp):
    """
    The function create_location_coordinates takes a dataset directory, a GeoDataFrame gdf, and another DataFrame stopp as inputs 
    to generate a dictionary mapping location UUIDs to their respective coordinates. This dictionary, loc_coord_dict, contains pairs 
    of start and end coordinates for each road segment or Spatial Point (SP) represented in the GeoDataFrame. The function iterates 
    through gdf to extract geometry information and through stopp to add latitude and longitude data for each location. 
    The function normalizes these coordinates, stores both the original and normalized coordinate dictionaries as pickled files in 
    a preprocessing directory within the given dataset directory, and returns the normalized coordinates along with their min-max normalization information.
    
    INPUT
    dataset_dir: dataset directory
    gdf: GeoDataFrame
    stopp: pandas data frame

    OUTPUT
    loc_coord_dict: {LocationUUID: coordinates_list}, where coordinates_list is a list of start and end coordinates of this location (road segment or SP)
    """
    # create location-coordinates dict
    loc_coord_dict = dict()
    road_start_time = time.time()
    for i in range(gdf.shape[0]):
        linestr = gdf.iloc[i]['geometry']
        coordinates_list = str_pointlist(linestr)
        loc_coord_dict[gdf.iloc[i]['LocationUUID']] = coordinates_list
        if i % 100000 == 1:
            temp_time = time.time()
            used_time = temp_time - road_start_time
            print(str(i), 'roads finished with', used_time, 'seconds')
            print('estimated rest time:', used_time/i*(gdf.shape[0]-i))
    stopp_start_time = time.time()
    for i in range(stopp.shape[0]):
        loc_coord_dict[stopp.iloc[i]['LocationUUID']] = [stopp.iloc[i]['Latitude'], stopp.iloc[i]['Longitude']]
        if i % 10000 == 1:
            temp_time = time.time()
            used_time = temp_time - stopp_start_time
            print(str(i), 'stopps finished with', used_time, 'seconds')
            print('estimated rest time:', used_time/i*(gdf.shape[0]-i))
    normalized_dict, min_max_info = normalize_coordinates(loc_coord_dict)

    save_to_pickle(dataset_dir + 'preprocess/loc_coord_dict.pickle', loc_coord_dict)
    save_to_pickle(dataset_dir + 'preprocess/norm_loc_coord_dict.pickle', [normalized_dict, min_max_info])

    return loc_coord_dict
    # return normalized_dict, min_max_info


def new_create_location_coordinates(train_dataset_dir, test_dataset_dir):
    """
    The function create_location_coordinates takes a dataset directory, and DataFrame stopp as inputs
    to generate a dictionary mapping location UUIDs to their respective coordinates. This dictionary, loc_coord_dict, contains pairs
    of start and end coordinates for stop Point (SP) represented in the GeoDataFrame. The function iterates through stopp to add latitude and longitude data for each location.
    The function normalizes these coordinates, stores both the original and normalized coordinate dictionaries as pickled files in
    a preprocessing directory within the given dataset directory, and returns the normalized coordinates along with their min-max normalization information.

    INPUT
    dataset_dir: dataset directory
    stopp: pandas data frame

    OUTPUT
    loc_coord_dict: {LocationUUID: coordinates_list}, where coordinates_list is a list of start and end coordinates of this location (road segment or SP)
    """
    # load stop point meta files
    train_set_unique_stop_points = pd.read_csv(os.path.join(train_dataset_dir, 'set_unique_stop_points.csv'))
    test_set_unique_stop_points = pd.read_csv(os.path.join(test_dataset_dir, 'set_unique_stop_points.csv'))
    stopp = pd.concat([train_set_unique_stop_points, test_set_unique_stop_points], ignore_index=True)
    # create location-coordinates dict
    loc_coord_dict = dict()
    stopp_start_time = time.time()
    for i in range(stopp.shape[0]):
        temp_coord_strs = stopp.iloc[i]['point'].replace('POINT (', '').replace(')', '').split(' ')
        temp_lon = float(temp_coord_strs[0])
        temp_lat = float(temp_coord_strs[1])
        loc_coord_dict[stopp.iloc[i]['global_stop_points']] = [temp_lat, temp_lon]
        if i % 10000 == 1:
            temp_time = time.time()
            used_time = temp_time - stopp_start_time
            print(str(i), 'stopps finished with', used_time, 'seconds')
            print('estimated rest time:', used_time / i * (stopp.shape[0] - i))
    normalized_dict, min_max_info = normalize_coordinates(loc_coord_dict)

    save_to_pickle(train_dataset_dir + 'preprocess/loc_coord_dict.pickle', loc_coord_dict)
    save_to_pickle(train_dataset_dir + 'preprocess/norm_loc_coord_dict.pickle', [normalized_dict, min_max_info])

    return loc_coord_dict

def bool_list_of_list(data):
    if all(isinstance(item, list) for item in data):
        return True
    else: 
        return False
    
def normalize_coordinates(coord_dict):
    # Initialize min and max values with the first coordinate
    min_lat, max_lat = float('inf'), -float('inf')
    min_lon, max_lon = float('inf'), -float('inf')
    
    # print(coord_dict)
    # Find the min and max values for latitude and longitude
    for coords in coord_dict.values():
        if bool_list_of_list(coords): 
            for lat, lon in coords:
                min_lat = min(min_lat, lat)
                max_lat = max(max_lat, lat)
                min_lon = min(min_lon, lon)
                max_lon = max(max_lon, lon)
        else:
            [lat, lon] = coords
            min_lat    = min(min_lat, lat)
            max_lat    = max(max_lat, lat)
            min_lon    = min(min_lon, lon)
            max_lon    = max(max_lon, lon)

    # Function to normalize a value
    def normalize(value, min_value, max_value):
        return (value - min_value) / (max_value - min_value)

    # Normalize the coordinates and update the dictionary
    normalized_dict = {}
    for key, coords in coord_dict.items(): 
        if bool_list_of_list(coords): 
            normalized_coords = [[normalize(lat, min_lat, max_lat), normalize(lon, min_lon, max_lon)] for lat, lon in coords]
            normalized_dict[key] = normalized_coords
        else:
            [lat, lon] = coords
            normalized_dict[key] = [normalize(lat, min_lat, max_lat), normalize(lon, min_lon, max_lon)]
    return normalized_dict, (min_lat, max_lat, min_lon, max_lon)



def unnormalize_coordinates(normalized_dict, min_max_info):

    (min_lat, max_lat, min_lon, max_lon) = min_max_info
    # Function to unnormalize a value
    def unnormalize(value, min_value, max_value):
        return (value * (max_value - min_value)) + min_value

    # Unnormalize the coordinates and update the dictionary
    unnormalized_dict = {}
    for key, coords in normalized_dict.items():
        if bool_list_of_list(coords): 
            unnormalized_coords = [(unnormalize(lat, min_lat, max_lat), unnormalize(lon, min_lon, max_lon)) for lat, lon in coords]
            unnormalized_dict[key] = unnormalized_coords
        else:
            [lat, lon] = coords
            unnormalized_dict[key] = [unnormalize(lat, min_lat, max_lat), unnormalize(lon, min_lon, max_lon)]
    return unnormalized_dict


    
def retrieve_gdf_stopp(train_dataset_folder):

    train_dataset_dir = train_dataset_folder
    dataset_dir = train_dataset_folder    
    if not os.path.exists(dataset_dir + 'metadata'):
        copy_tree(train_dataset_dir + 'metadata', dataset_dir + 'metadata')
    if not os.path.exists(dataset_dir+'preprocess/'):
        os.makedirs(dataset_dir+'preprocess/')
    if not os.path.exists(dataset_dir+'preprocess/gdd.csv') or not os.path.exists(dataset_dir + 'preprocess/gdf.csv'):
        edge_file = gpd.read_file(dataset_dir + 'metadata/roads.shp')
    loc_id_name = 'id'
    # if 'Kitware' in dataset_dir:
    #     loc_id_name = 'uuid' 
    # else:
    #     loc_id_name = 'id'
    if not os.path.exists(dataset_dir + 'preprocess/gdd.csv'):
        gdd = pd.DataFrame(edge_file).rename(columns={loc_id_name: 'LocationUUID'}) # id for Baseline, L3 and Nov, uuid for Kitware
        gdd[['LocationUUID']] = gdd[['LocationUUID']].astype(str)
        gdd.to_csv(dataset_dir + 'preprocess/gdd.csv')
    # else:
    gdd = pd.read_csv(dataset_dir + 'preprocess/gdd.csv')
    gdd[['LocationUUID']] = gdd[['LocationUUID']].astype(str)
    if not os.path.exists(dataset_dir + 'preprocess/gdf.csv'):
        gdf = pd.DataFrame(edge_file[[loc_id_name,"geometry"]]).rename(columns={loc_id_name: 'LocationUUID'}) # id for Baseline, L3 and Nov, uuid for Kitware
        gdf[['LocationUUID']] = gdf[['LocationUUID']].astype(str)
        gdf.to_csv(dataset_dir + 'preprocess/gdf.csv')
    # else:
    gdf = pd.read_csv(dataset_dir + 'preprocess/gdf.csv')#.rename(columns = {'osmid': 'LocationUUID'})
    gdf[['LocationUUID']] = gdf[['LocationUUID']].astype(str)
    stopp_file = dataset_dir + 'merged_stopp.parquet'
    stopp = pd.read_parquet(stopp_file)
    stopp[stopp.select_dtypes(np.float64).columns] = stopp.select_dtypes(np.float64).astype(np.float32)
    stopp[['LocationUUID']] = stopp[['LocationUUID']].astype(str)

    return gdf, stopp
    


        

def chunk_file_paths(file_path_list, K):
    """
    It is designed to take a list of file paths and an integer K, and return a list of lists (chunks) 
    where each sublist contains up to K file paths from the original list. This function is useful for 
    processing or handling large lists of file paths in smaller, more manageable batches.

    INPUT
    file_path_list: A list of strings, where each string represents a path to a file. This list contains 
    all the file paths that you want to divide into chunks.
    K: An integer representing the maximum number of file paths that should be included in each chunk (sublist).


    OUTPUT 
    chunked_list: A list of lists (sublists). Each sublist contains up to K file paths from the original file_path_list. 
    The last chunk may contain fewer than K file paths if the total number of file paths in file_path_list is not exactly divisible by K.
    """
    # Initialize the list to hold chunks of file names
    chunked_list = []
    
    # Iterate over the list and create sublists of size K or less
    for i in range(0, len(file_path_list), K):
        # Slice the list and append the chunk to the chunked_list
        chunked_list.append(file_path_list[i:i+K])
        
    return chunked_list


def save_fast_lof(clf, filename):
    import faiss 
    # import pickle 
    faiss_index = clf.detector_.faiss_index
    # Assuming `faiss_index` is your original GPU index
    cpu_index = faiss.index_gpu_to_cpu(faiss_index)    
    faiss.write_index(cpu_index, filename + '_faiss_index.bin')
    del clf.detector_.faiss_index
    # Serialize the rest of the model
    with open(filename, 'wb') as f:
        pickle.dump(clf, f)
    clf.detector_.faiss_index = faiss_index

def load_fast_lof(filename):
    import faiss 
    # import pickle 
    # Deserialize the model without the FAISS index
    with open(filename, 'rb') as f:
        clf = pickle.load(f)    
    # Deserialize the FAISS index
    clf.detector_.faiss_index = faiss.read_index(filename + '_faiss_index.bin')
    return clf

def load_fast_lof(filename):
    import faiss
    import pickle

    # Deserialize the model without the FAISS index
    with open(filename, 'rb') as f:
        clf = pickle.load(f)
    
    # Deserialize the FAISS index
    cpu_index = faiss.read_index(filename + '_faiss_index.bin')
    
    # Move the FAISS index to all available GPUs
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    
    # Assign the GPU index back to the detector within the classifier
    clf.detector_.faiss_index = gpu_index
    
    return clf

def print_time(seconds):
    hours = seconds // 3600  # 3600 seconds in an hour
    minutes = (seconds % 3600) // 60  # Remaining seconds converted to minutes
    seconds_remaining = seconds % 60  # Remaining seconds
    str_running_time = f"{hours} hour(s), {minutes} minute(s), {seconds_remaining} second(s)"
    print(str_running_time)
    return str_running_time


def save_fast_knn(clf, filename):
    faiss_index = clf.faiss_index
    # Assuming `faiss_index` is your original GPU index
    cpu_index = faiss.index_gpu_to_cpu(faiss_index)    
    faiss.write_index(cpu_index, filename + '_faiss_index.bin')
    del clf.faiss_index
    # Serialize the rest of the model
    with open(filename, 'wb') as f:
        pickle.dump(clf, f)
    clf.faiss_index = faiss_index

def load_fast_knn(filename):
    # Deserialize the model without the FAISS index
    with open(filename, 'rb') as f:
        clf = pickle.load(f)
    
    # Deserialize the FAISS index
    cpu_index = faiss.read_index(filename + '_faiss_index.bin')
    
    # Move the FAISS index to all available GPUs
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    
    # Assign the GPU index back to the classifier
    clf.faiss_index = gpu_index
    
    return clf


def normalize_data(X):
    """
    Normalize the data matrix X using StandardScaler from sklearn.
    
    Parameters:
    - X: A 2D numpy array or pandas DataFrame where each column is a feature.
    
    Returns:
    - X_normalized: The normalized data as a numpy array.
    - scaler: The fitted StandardScaler instance used for normalization.
    """
    scaler = StandardScaler()
    # scaler       = RobustScaler()
    X_normalized = scaler.fit_transform(X)
    
    return X_normalized, scaler

def unnormalize_data(X_normalized, scaler):
    """
    Unnormalize the data matrix X_normalized using the provided StandardScaler instance.
    
    Parameters:
    - X_normalized: A (n,d) numpy array of the normalized data, where d is the number of features. 
    - scaler: The fitted StandardScaler instance used for the original normalization.
    
    Returns:
    - X_unnormalized: The unnormalized data as a numpy array (n, d)
    """

    X_unnormalized = scaler.inverse_transform(X_normalized)
    
    return X_unnormalized


def calc_p_values(quantiles, scores):
    """
    INPUT
    quantiles: a (m,) array of quantiles that used to calculate p-values
    scores: a (n,) array of scores that will be converted to p-values based on quantiles

    OUTPUT
    pvalues: a (n,) array of p-values
    """
    
    indices = np.searchsorted(quantiles, scores, side='left')
    pvalues = 1 - (indices - 1) / (len(quantiles) + 1)
    return pvalues


"""
This function splits the provided file path into the directory path and the file name, 
and then returns them as two separate strings. The directory path includes the trailing '/'
"""
def split_path(file_path):
    # Split the path by '/'
    parts = file_path.split('/')
    
    # Extract the file name (last part)
    file_name = parts[-1]
    
    # Rejoin the path without the file name
    dir_path = '/'.join(parts[:-1]) + '/'
    
    return dir_path, file_name


def save_to_dill(file_path, obj):
    """
    Saves a Python object to a file using dill.

    :param file_path: Path to the file where the object should be saved.
    :param obj: The Python object to save.
    """
    with open(file_path, 'wb') as file:
        dill.dump(obj, file)

def load_from_dill(file_path):
    """
    Loads a Python object from a file using dill.

    :param file_path: Path to the file from which the object should be loaded.
    :return: The Python object loaded from the file.
    """
    with open(file_path, 'rb') as file:
        obj = dill.load(file)
    return obj

def save_to_pickle(folder, obj):
    """
    Saves a Python object to a file using pickle.

    Parameters:
    - file_path: The full path (including filename) where the object should be saved.
    - obj: The Python object to be saved.
    """
    with open(folder, 'wb') as fp:
        pickle.dump(obj, fp)       

def load_from_pickle(file_path):
    """
    Loads a Python object from a pickle file.

    Parameters:
    - file_path: The full path to the pickle file from which the object should be loaded.

    Returns:
    - The Python object loaded from the pickle file.
    """
    with open(file_path, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

def create_folder_if_not_exists(folder_path):
    """
    Check if a folder exists, and if not, create it.

    Parameters:
    - folder_path (str): The path to the folder you want to check and possibly create.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    # else:
    #     print(f"Folder '{folder_path}' already exists.")


def get_hos_agents(hos_dir):
    hos_agents_dict = dict()
    for root, dirs, files in os.walk(hos_dir):
        for file in files:
            if file.startswith('hos_') and file.endswith('.json'):
                with open(os.path.join(root, file), "r") as f:
                    temp_json_dict = json.load(f)
                temp_agent = temp_json_dict['events'][0]['agents'][0]
                temp_hos = file.replace('hos_', '').replace('.json', '')
                hos_agents_dict[temp_hos] = int(temp_agent)
    return hos_agents_dict

