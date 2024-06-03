import os
import pandas as pd
import numpy as np
import pickle 
import time as time
import sys
from distutils.dir_util import copy_tree
import geopandas as gpd
import random
from datetime import datetime
import math
from tqdm import tqdm

def agent_anomaly_days_dict(train_dataset_folder, folder_path):
    agent_anomaly_days_dict = {}
    for subdir in os.listdir(folder_path):
        subdir_path = os.path.join(folder_path, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.endswith('.parquet'):
                    file_path = os.path.join(subdir_path, filename)
                    df = pd.read_parquet(file_path)
                    df = df.drop_duplicates(keep='first')
                    df = df.drop_duplicates(subset=["timestamp", "instruction"], keep='first')
                    df['datetime'] = pd.to_datetime(df['timestamp'])
                    df['day'] = df['datetime'].dt.date
                    df = df.sort_values(by=['timestamp'], ascending=[True]).reset_index()
                    agent_id = df['agent'][0]
                    if agent_id not in agent_anomaly_days_dict:
                        agent_anomaly_days_dict[agent_id] = set()
                        agent_anomaly_days_dict[agent_id].update(df[df['agent'] == agent_id]['day'])

    for agent_id in agent_anomaly_days_dict:
        agent_anomaly_days_dict[agent_id] = list(agent_anomaly_days_dict[agent_id])
    npy_file_path = os.path.join(train_dataset_folder + 'preprocess/agent_anomaly_days_dict.npy')   # Set the output file path
    np.save(npy_file_path, agent_anomaly_days_dict)  # Save the dictionary

    return agent_anomaly_days_dict



def retrieve_gdf_stopp(train_dataset_folder, test_dataset_folder):

   
    if not os.path.exists(test_dataset_folder + 'metadata'):
        print('11111111111111111')
        copy_tree(train_dataset_folder + 'metadata', test_dataset_folder + 'metadata')
    if not os.path.exists(test_dataset_folder+'preprocess/'):
        print('22222222222222222')
        os.makedirs(test_dataset_folder+'preprocess/')
    if not os.path.exists(test_dataset_folder+'preprocess/gdd.csv') or not os.path.exists(test_dataset_folder + 'preprocess/gdf.csv'):
        print('33333333333333333')
        edge_file = gpd.read_file(test_dataset_folder + 'metadata/roads.shp')
    loc_id_name = 'id'
    # if 'Kitware' in dataset_dir:
    #     loc_id_name = 'uuid' 
    # else:
    #     loc_id_name = 'id'
    if not os.path.exists(test_dataset_folder + 'preprocess/gdd.csv'):
        print('444444444444444444')
        gdd = pd.DataFrame(edge_file).rename(columns={loc_id_name: 'LocationUUID'}) # id for Baseline, L3 and Nov, uuid for Kitware
        gdd[['LocationUUID']] = gdd[['LocationUUID']].astype(str)
        gdd.to_csv(test_dataset_folder + 'preprocess/gdd.csv')
    # else:
    gdd = pd.read_csv(test_dataset_folder + 'preprocess/gdd.csv')
    gdd[['LocationUUID']] = gdd[['LocationUUID']].astype(str)
    if not os.path.exists(test_dataset_folder + 'preprocess/gdf.csv'):
        gdf = pd.DataFrame(edge_file[[loc_id_name,"geometry"]]).rename(columns={loc_id_name: 'LocationUUID'}) # id for Baseline, L3 and Nov, uuid for Kitware
        gdf[['LocationUUID']] = gdf[['LocationUUID']].astype(str)
        gdf.to_csv(test_dataset_folder + 'preprocess/gdf.csv')
    # else:
    gdf = pd.read_csv(test_dataset_folder + 'preprocess/gdf.csv')#.rename(columns = {'osmid': 'LocationUUID'})
    gdf[['LocationUUID']] = gdf[['LocationUUID']].astype(str)
    stopp_file = test_dataset_folder + 'StopPoints.parquet'
    stopp = pd.read_parquet(stopp_file)
    stopp[stopp.select_dtypes(np.float64).columns] = stopp.select_dtypes(np.float64).astype(np.float32)
    stopp[['LocationUUID']] = stopp[['LocationUUID']].astype(str)

    return gdf, stopp

def str_pointlist(str):
    pointlist = []
    str = str.replace("LINESTRING (","").replace(")","").replace(", ",",")
    str_list =str.split(",")
    for item in str_list:
        pair_list = item.split(" ")
        pointlist.append([float(pair_list[1]),float(pair_list[0])])
    return pointlist

def save_to_pickle(folder, obj):
    """
    Saves a Python object to a file using pickle.

    Parameters:
    - file_path: The full path (including filename) where the object should be saved.
    - obj: The Python object to be saved.
    """
    with open(folder, 'wb') as fp:
        pickle.dump(obj, fp)

def create_location_coordinates(test_dataset_folder, gdf, stopp):
    """
    The function create_location_coordinates takes a dataset directory, a GeoDataFrame gdf, and another DataFrame stopp as inputs 
    to generate a dictionary mapping location UUIDs to their respective coordinates. This dictionary, loc_coord_dict, contains pairs 
    of start and end coordinates for each road segment or Spatial Point (SP) represented in the GeoDataFrame. The function iterates 
    through gdf to extract geometry information and through stopp to add latitude and longitude data for each location. 
    The function normalizes these coordinates, stores both the original and normalized coordinate dictionaries as pickled files in 
    a preprocessing directory within the given dataset directory, and returns the normalized coordinates along with their min-max normalization information.
    
    INPUT
    test_dataset_folder: dataset directory
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

    save_to_pickle(test_dataset_folder + 'preprocess/loc_coord_dict.pickle', loc_coord_dict)




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

def to_datetime(timestamp):
    """Ensure the timestamp is converted to datetime object."""
    if not isinstance(timestamp, datetime):
        # Assuming the timestamp format is standard; adjust the format as needed.
        return pd.to_datetime(timestamp)
    return timestamp

def diff_hour(start_time, end_time):
    """
    Calculate the difference in hours between two datetime objects.

    Parameters:
    start_time (datetime or convertible): The start time.
    end_time (datetime or convertible): The end time.

    Returns:
    float: The difference in hours between start_time and end_time.
    """
    # Ensure both times are datetime objects
    start_time = to_datetime(start_time)
    end_time = to_datetime(end_time)

    # Calculate the time difference and convert it to hours
    time_diff = end_time - start_time
    hours_diff = time_diff.total_seconds() / 3600  # Convert seconds to hours
    return hours_diff

def process_dataframe(df):
    """Process the dataframe to extract trip dates, coordinates and other features."""
    df['arrival_datetime'] = pd.to_datetime(df['time_start'])
    # df['depart_datetime'] = pd.to_datetime(df['time_stop'])
    # Assuming the format is Year-Month-Day Hours:Minutes:Seconds
    df['depart_datetime'] = pd.to_datetime(df['time_stop'], format='%Y-%m-%d %H:%M:%S')

    df['arrival_timestamp'] = df['arrival_datetime'].apply(lambda x: x.timestamp())
    df['depart_timestamp'] = df['depart_datetime'].apply(lambda x: x.timestamp())
    
    # Assuming 'read_traveldis' processes df and returns trip details
    trip_features, _, _, _, trip_datetime_df, _, _ = read_traveldis(df, have_tel=True)
    
    # Extracting start and end times, days
    # trip_datetime_df['start_datetime'] = pd.to_datetime(df['start_datetime'], format='%Y-%m-%d %H:%M:%S')

    trip_datetime_df['start_date'] = trip_datetime_df['start_datetime'].dt.date
    trip_datetime_df['hour_of_day'] = trip_datetime_df['start_datetime'].dt.hour
    
    return trip_features, trip_datetime_df

def calculate_trip_features(trip_features):
    """Calculate travel times and Euclidean distances for trips."""
    n_trips = len(trip_features)
    raw_X = np.array(trip_features).astype(float)  # start_hour_of_day,end_hour_of_day,travel_time,travel_distance,start_lat,start_lon,end_lat,end_lon,day_week,start_dur,end_dur
    X = np.zeros((n_trips, 9))
    # X[:, 0] = raw_X[:, 2]  # travel times
    # X[:, 1] = np.sqrt((raw_X[:, 4] - raw_X[:, 6]) ** 2 + (raw_X[:, 5] - raw_X[:, 7]) ** 2)  # Euclidean distances
    X[:, 0] = raw_X[:, 2] 
    X[:, 0] = raw_X[:, 2] 
    X[:, 0] = raw_X[:, 2] 
    X[:, 0] = raw_X[:, 2] 
    X[:, 0] = raw_X[:, 2] 
    X[:, 0] = raw_X[:, 2] 
    return X


def compile_agent_data(folder_path, agent_id, filename):

    # Read parquet files
    file_path = os.path.join(folder_path, filename)
    df = pd.read_parquet(file_path)

    if df.empty:
        print("No data found for the specified agent ID.")
        return {}

    # Process dataframe
    trip_features, trip_datetime_df = process_dataframe(df)

    # Calculate trip features
    # X = calculate_trip_features(trip_features)

    # Adding coordinates and time data to trip data
    trip_datetime_df['start_lat'] = np.array(trip_features).astype(float)[:, 4]
    trip_datetime_df['start_lon'] = np.array(trip_features).astype(float)[:, 5]

    date_to_coords_dict = trip_datetime_df.groupby('start_date').apply(
        lambda group: group[['start_lat', 'start_lon', 'hour_of_day']].values.tolist()
    ).to_dict()

    return date_to_coords_dict

import os
import random


def subsample_data(parquet_folder, abnormal_agent_id_list, subsample_size):
    """
    Create a subsample of agent IDs and their corresponding filenames from a dataset folder,
    ensuring all specified abnormal agent IDs are included, with a fixed random seed for consistency.

    Parameters:
    - parquet_folder: Path to the folder containing the dataset files.
    - abnormal_agent_id_list: List of abnormal agent IDs to include mandatorily in the subsample.
    - subsample_size: Desired size of the subsample, which must be at least as large as the number
                      of abnormal agent IDs provided.

    Returns:
    - subsampled_agents: List of tuples, each containing an agent ID and its corresponding filename.
    """
    # Validate the subsample size
    num_abnormal_agents = len(abnormal_agent_id_list)
    if subsample_size < num_abnormal_agents:
        raise ValueError("Subsample size must be at least as large as the number of abnormal agent IDs.")

    # Collect all valid training agent filenames
    all_filenames = [filename for filename in os.listdir(parquet_folder) if filename.endswith('.parquet')]
    all_agent_ids = [filename.replace('.parquet', '') for filename in all_filenames if filename.replace('.parquet', '').isdigit()]

    # Prepare initial subsamples with abnormal agent IDs and corresponding filenames
    subsampled_agents = [(agent_id, f"{agent_id}.parquet") for agent_id in abnormal_agent_id_list if f"{agent_id}.parquet" in all_filenames]

    # Determine the remaining size needed for the subsample
    remaining_size = subsample_size - num_abnormal_agents
    available_agent_ids = list(set(all_agent_ids) - set(abnormal_agent_id_list))
    available_filenames = [filename for filename in all_filenames if filename.replace('.parquet', '') in available_agent_ids]

    if len(available_agent_ids) < remaining_size or len(available_filenames) < remaining_size:
        raise ValueError("Not enough available agent IDs or filenames to reach the desired subsample size.")

    # Randomly pick additional agent IDs and filenames to complete the subsample
    random.seed(42)
    additional_agent_ids = random.sample(available_agent_ids, remaining_size)
    additional_filenames = [f"{agent_id}.parquet" for agent_id in additional_agent_ids]

    # Append each individually to subsampled_agents
    subsampled_agents.extend(zip(additional_agent_ids, additional_filenames))

    print('remaining_size', remaining_size)
    print('subsampled_agents', len(subsampled_agents))

    return subsampled_agents





def agent_date_time_sp_coords_dict(train_dataset_folder, test_dataset_folder, subsample_size):

    agent_anomaly_days_dict = np.load(os.path.join(train_dataset_folder, "preprocess/agent_anomaly_days_dict.npy"), allow_pickle=True).item()
    abnormal_agent_id_list = list(agent_anomaly_days_dict.keys())
    
    subsample_size = int(subsample_size)
    train_agent_list = subsample_data(train_dataset_folder, abnormal_agent_id_list, subsample_size)
    # test_agent_list = subsample_data(test_dataset_folder, abnormal_agent_id_list, subsample_size)

    # print('train_agent_list', train_agent_list[-1])
    # print('test_agent_list', test_agent_list[-1])

    label_dict = {-1: "full_", 200: "200_", 500: "500_", 800: "800_", 1000: "1k_", 2000: "2k_", 5000: "5k_", 10000: "10k_", 50000: "50k_", 100000: "100k_", 150000: "150k_", 200000: "200k_"}

    all_train_agent_data = {}
    for item in tqdm(train_agent_list, desc="Training Processing agents"):
        agent_data = compile_agent_data(train_dataset_folder, item[0], item[1])
        all_train_agent_data[item[0]] = agent_data
    save_to_pickle(f"{train_dataset_folder}/preprocess/train_{label_dict[subsample_size]}agent_date_time_sp_coords_dict.pkl", all_train_agent_data)

    all_test_agent_data = {}
    for item in tqdm(train_agent_list, desc="Test Processing agents"):
        agent_data = compile_agent_data(test_dataset_folder, item[0], item[1])
        all_test_agent_data[item[0]] = agent_data
    save_to_pickle(f"{test_dataset_folder}/preprocess/test_{label_dict[subsample_size]}agent_date_time_sp_coords_dict.pkl", all_test_agent_data)


if __name__ == '__main__':

    start = time.time()
    args = sys.argv
    if len(args) == 4:
        train_dataset_folder = args[1]
        test_dataset_folder = args[2]
        subsample_size = args[3]
    else:
        raise Exception("no dataset")


    # TODO: generate agent_anomaly_days_dict, key is agent_id and value is dates

    # gts_path_dict = {
    #     '/data/jxl220096/hay/trial2_oldformat/losangeles_eventlogs_train/': '/data/jxl220096/hay/trial2_oldformat/gts/la_gts',
    #     '/data/jxl220096/hay/trial2_oldformat/knoxville_unjoined_train/': '/data/jxl220096/hay/trial2_oldformat/gts/kx_gts',
    #     '/data/jxl220096/hay/trial2_oldformat/sanfrancisco_train_eventlogs/': '/data/jxl220096/hay/trial2_oldformat/gts/sf_gts', 
    #     '/home/jxl220096/data/hay/trial2_oldformat/singapore_test_event_logs/': '/home/jxl220096/data/hay/new_format/trial2/gts/sp_gts',
    # }

    gts_path_dict = {
        '/home/jxl220096/data/hay/new_format/trial2/sanfrancisco/test_stops/' : '/home/jxl220096/data/hay/new_format/trial2/gts/sf_gts',
        '/home/jxl220096/data/hay/new_format/trial2/knoxville/test_stops/' : '/home/jxl220096/data/hay/new_format/trial2/gts/kx_gts',
        '/home/jxl220096/data/hay/new_format/trial2/losangeles/test_stops/' : '/home/jxl220096/data/hay/new_format/trial2/gts/la_gts',
        '/home/jxl220096/data/hay/new_format/trial2/singapore/test_stops/' : '/home/jxl220096/data/hay/new_format/trial2/gts/sp_gts',
    }

    gts_path = gts_path_dict[test_dataset_folder]
    agent_anomaly_days_dict(train_dataset_folder, gts_path)

    
    # # TODO: generate loc_coord_dict
    # gdf, stopp = retrieve_gdf_stopp(train_dataset_folder, test_dataset_folder)
    # create_location_coordinates(test_dataset_folder, gdf, stopp)
    
    # # TODO: generate agent_date_time_sp_coords_dict
    # agent_date_time_sp_coords_dict(train_dataset_folder, test_dataset_folder, subsample_size)

# python data_gen.py '/home/jxl220096/data/hay/new_format/trial2/sanfrancisco/train_stops/' '/home/jxl220096/data/hay/new_format/trial2/sanfrancisco/test_stops/'
# python data_gen.py '/home/jxl220096/data/hay/new_format/trial2/knoxville/train_stops/' '/home/jxl220096/data/hay/new_format/trial2/knoxville/test_stops/'
# python data_gen.py '/home/jxl220096/data/hay/new_format/trial2/losangeles/train_stops/' '/home/jxl220096/data/hay/new_format/trial2/losangeles/test_stops/'
# python data_gen.py '/home/jxl220096/data/hay/new_format/trial2/singapore/train_stops/' '/home/jxl220096/data/hay/new_format/trial2/singapore/test_stops/'


# python data_gen.py /home/jxl220096/data/hay/trial2_oldformat/singapore_train_event_logs/ /home/jxl220096/data/hay/trial2_oldformat/singapore_test_event_logs/