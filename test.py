import numpy as np
import os
import pandas as pd
from geopy.distance import geodesic
from shapely.wkt import loads
from math import radians, cos, sin, asin, sqrt, atan2, degrees
import time
import sys
import tilemapbase
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch, Arrow


def calculate_distance(row):
    prev_point = row['geometry']
    next_point = row['geometry_shifted']
    if pd.notna(prev_point) and pd.notna(next_point):
        # Convert WKT format to Shapely Point objects
        prev_point = loads(prev_point)
        next_point = loads(next_point)
        # Extract latitude and longitude from the Point objects
        prev_lat, prev_lon = prev_point.y, prev_point.x
        next_lat, next_lon = next_point.y, next_point.x
        # Calculate and return the geodesic distance
        return geodesic((prev_lat, prev_lon), (next_lat, next_lon)).meters
    return np.nan



def calculate_daily_distance(df):
    # Convert WKT POINT format to latitude and longitude using shapely
    # df['geometry'] = df['geometry'].apply(lambda x: loads(x) if pd.notna(x) else None)
    
    # # Create new columns for latitude and longitude
    # df['latitude'] = df['geometry'].apply(lambda x: x.y if x is not None else None)
    # df['longitude'] = df['geometry'].apply(lambda x: x.x if x is not None else None)
    
    # Sort by agent_id and time_start
    # df.sort_values(by=['agent_id', 'time_start'], inplace=True)
    df.sort_values(by=['agent', 'timestamp'], inplace=True)
    
    # Calculate distances between consecutive stops
    df['next_latitude'] = df.groupby('agent')['latitude'].shift(-1)
    df['next_longitude'] = df.groupby('agent')['longitude'].shift(-1)
    df['distance_to_next'] = df.apply(lambda row: geodesic(
        (row['latitude'], row['longitude']), 
        (row['next_latitude'], row['next_longitude'])
    ).meters if pd.notna(row['next_latitude']) else 0, axis=1)
    
    # Extract date from time_start and sum distances by agent_id and date
    # df['date'] = pd.to_datetime(df['time_start']).dt.date
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    # daily_distance = df.groupby(['agent_id', 'date'])['distance_to_next'].sum().reset_index()
    daily_distance = df.groupby(['agent', 'date'])['distance_to_next'].sum().reset_index()
    
    return daily_distance

def count_daily_stops(df):
    # Ensure the time_start column is in datetime format
    # df['time_start'] = pd.to_datetime(df['time_start'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract date from time_start for daily grouping
    # df['date'] = df['time_start'].dt.date
    df['date'] = df['timestamp'].dt.date
    
    # Group by agent_id and date, then count the number of stops (records) per group
    # daily_stops = df.groupby(['agent_id', 'date']).size().reset_index(name='number_of_stops')
    daily_stops = df.groupby(['agent', 'date']).size().reset_index(name='number_of_stops')
    
    return daily_stops


def haversine(lon1, lat1, lon2, lat2):
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))

    # Radius of earth in kilometers. Use 6371 for kilometers
    r = 6371
    return c * r

def calculate_bearing(lon1, lat1, lon2, lat2):
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Bearing calculation
    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    initial_bearing = atan2(x, y)
    
    # Convert bearing from radians to degrees
    initial_bearing = degrees(initial_bearing)
    bearing = (initial_bearing + 360) % 360

    return bearing

def calculate_bearing_changes(df):
    # Assuming latitude and longitude are already extracted as separate columns
    # Calculate bearings between consecutive points
    df['next_latitude'] = df['latitude'].shift(-1)
    df['next_longitude'] = df['longitude'].shift(-1)
    df['bearing'] = df.apply(lambda row: calculate_bearing(row['longitude'], row['latitude'], row['next_longitude'], row['next_latitude']), axis=1)
    
    # Calculate change in bearing
    df['previous_bearing'] = df['bearing'].shift(1)
    df['change_in_bearing'] = (df['bearing'] - df['previous_bearing']).abs()
    df['change_in_bearing'] = df['change_in_bearing'].apply(lambda x: min(x, 360-x) if x is not None else None)

    # return df[['agent_id', 'time_start', 'bearing', 'change_in_bearing']]
    return df[['agent', 'timestamp', 'bearing', 'change_in_bearing']]



def print_function(train_dataset_folder, agent_anomaly_days_dict):

    # parquet_file_path = train_dataset_folder + '229918.parquet'
    parquet_file_path = train_dataset_folder + '2194.parquet'
    temp_parquet = pd.read_parquet(parquet_file_path)
    temp_parquet.sort_values(by=['time_start'], inplace=True)

    print(temp_parquet.head())

    temp_parquet['duration_at_stop'] = (pd.to_datetime(temp_parquet['time_stop']) - pd.to_datetime(temp_parquet['time_start'])).dt.total_seconds()
    temp_parquet['travel_time_to_next'] = (pd.to_datetime(temp_parquet['time_start']).shift(-1) - pd.to_datetime(temp_parquet['time_stop'])).dt.total_seconds()
    temp_parquet['geometry_shifted'] = temp_parquet['geometry'].shift(-1)
    temp_parquet['distance_to_next'] = temp_parquet.apply(calculate_distance, axis=1)
    print(temp_parquet.head())


    daily_distances = calculate_daily_distance(temp_parquet)
    print(daily_distances.head())

    daily_stops = count_daily_stops(temp_parquet)
    print(daily_stops.head())

    bearing_changes = calculate_bearing_changes(temp_parquet)
    print(bearing_changes.head())

    # print(agent_anomaly_days_dict[229918])
    print(agent_anomaly_days_dict[2194])


def daily_animation(frame, ax_train, temp_lat, temp_lon, tiles, extent):
    plotter = tilemapbase.Plotter(extent, tiles, height=600)
    plotter.plot(ax_train, tiles, alpha=0.5)


    temp_projected_lon, temp_projected_lat = project_coordinates_to_df(temp_lat, temp_lon)
    num_coords_to_show = min(len(temp_projected_lon), frame + 1)
    num_coords_to_show = len(temp_projected_lon)

    ax_train.plot(temp_projected_lon[:num_coords_to_show], temp_projected_lat[:num_coords_to_show],
                    color='blue', linewidth=2.7, alpha=1.0, zorder=1)

    if num_coords_to_show > 1:
        arrow_start_lon = temp_projected_lon[num_coords_to_show - 2]
        arrow_start_lat = temp_projected_lat[num_coords_to_show - 2]
        arrow_end_lon = temp_projected_lon[num_coords_to_show - 1]
        arrow_end_lat = temp_projected_lat[num_coords_to_show - 1]

        arrow = FancyArrowPatch((arrow_start_lon, arrow_start_lat), (arrow_end_lon, arrow_end_lat),
                                arrowstyle='->', color='blue', linewidth=2, mutation_scale=20, zorder=2)
        ax_train.add_patch(arrow)

    return ax_train.figure


def project_coordinates_to_df(latitude_list, longitude_list):
    """
    Project latitude and longitude coordinates to a projected coordinate system.

    Parameters:
        latitude_list (list): List of latitude coordinates.
        longitude_list (list): List of longitude coordinates.

    Returns:
        pd.DataFrame: DataFrame with columns "x" and "y" containing the projected coordinates.
    """
    tilemapbase.init(create=True)
    data = {'lat': latitude_list, 'lon': longitude_list}
    df = pd.DataFrame(data)
    projected_coordinates = df.apply(lambda row: pd.Series(tilemapbase.project(row['lon'], row['lat'])), axis=1)
    projected_coordinates.columns = ["x", "y"]
    projected_coordinates_x = projected_coordinates["x"]
    projected_coordinates_y = projected_coordinates["y"]

    return projected_coordinates_x, projected_coordinates_y

def still_plots(train_dataset_folder, folder_path):
    # read parquet files

    output_folder_path = os.path.join(train_dataset_folder, 'plots/anomaly_agents_gts_trajectories/')
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    print(output_folder_path)

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

                    temp_agent = df['agent'][0]

                    # print('=======================================')
                    # print('temp_agent', temp_agent)
                    # pd.set_option('display.max_columns', None)
                    # print(df)

                    # daily_distances = calculate_daily_distance(df)
                    # print(daily_distances)

                    # daily_stops = count_daily_stops(df)
                    # print(daily_stops)

                    # bearing_changes = calculate_bearing_changes(df)
                    # print(bearing_changes)

                    # print('=======================================')

                    all_coords_lat = df['latitude']
                    all_coords_lon = df['longitude']
                    lat_min, lat_max, lon_min, lon_max = all_coords_lat.min(), all_coords_lat.max(), all_coords_lon.min(), all_coords_lon.max()
                    tilemapbase.init(create=True)  # This initializes the tilemap library. The create=True argument suggests that it's creating a new map instance.
                    lat_expand = 0.3 * (lat_max - lat_min)  # It calculates a latitude expansion value by taking 30% of the difference between the maximum and minimum latitude values.
                    lon_expand = 0.3 * (lon_max - lon_min)  # Similarly, it calculates a longitude expansion value by taking 30% of the difference between the maximum and minimum longitude values.
                    extent = tilemapbase.Extent.from_lonlat(
                        lon_min - lon_expand,
                        lon_max + lon_expand,
                        lat_min - lat_expand,
                        lat_max + lat_expand,
                    )
                    tiles = tilemapbase.tiles.build_OSM()

                    # --- draw animation ---
                    plt.clf() 
                    fig, ax_train = plt.subplots(1, 1, figsize=(10, 10), sharex=False, sharey=False)
                    ani = FuncAnimation(fig, daily_animation,
                                        frames=1,
                                        interval=500, repeat=False,
                                        fargs=(ax_train, all_coords_lat, all_coords_lon, tiles, extent))

                    fig.legend(fontsize=20)
                    fig.tight_layout()
                    fig.suptitle('agent ' + str(temp_agent), fontsize=20)
                    output_filename = str(temp_agent) + '.gif'
                    gifoutput_path = os.path.join(output_folder_path, output_filename)
                    print('output_filename:', output_filename)
                    ani.save(gifoutput_path, writer='pillow')
                    plt.close('all')


# parquet_file_path = '/home/jxl220096/data/hay/new_format/trial2/gts/la_gts/gts_b46df1d2-ab87-4117-9472-86e8901aaec7/part-fd232366-61db-42f8-9e01-4ba09d6f87b8.zstd.parquet'
# temp_parquet = pd.read_parquet(parquet_file_path)
# print(temp_parquet.head())

# a = [229918, 216682, 268103, 107633, 142555, 195714, 215821, 65023, 87153, 132533, 12818, 95678, 288017, 93160, 39922, 232502, 138530, 246024, 53165, 56793, 224687, 53124, 148066, 290834, 192153, 145919, 268801, 52010, 134765, 101830, 98240, 49055, 125152, 109189, 121234, 227510, 19725, 89609, 40675, 89811, 39177, 221200, 145414, 287431, 249207, 219482, 233941, 69647, 250428, 123378, 67551, 190012, 271189, 106533, 62016, 73445, 87921, 114080, 52589, 52750, 70320, 38795, 287264, 168145, 141275, 174935, 132829, 70827, 186980, 221824, 245941, 216848, 155074, 202417, 16370, 108074, 94005, 75292, 283307, 35845, 179548, 181919, 265218, 45288, 288209, 127567, 37855, 246613, 140445, 153292, 266021, 73384, 279975, 8835, 47025, 101104, 103739, 232082, 157801, 250251, 248047, 149457, 52653, 157757, 65541, 194505, 219839, 60486, 183780, 283687, 209952, 291796, 78673, 163075, 96175, 2540, 51204, 151324, 90735, 215297, 158762, 115011, 70739, 22134, 53973, 242621, 132311, 134276, 76956, 156931, 209081, 141664, 134754, 3441, 184623, 288142, 290172, 196404, 271205, 230478, 73153, 84003, 266631, 168835, 39956, 60791, 2194]
# print(len(a))


# train_dataset_folder = '/home/jxl220096/data/hay/new_format/trial2/losangeles/train_stops/'
# test_dataset_folder = '/home/jxl220096/data/hay/new_format/trial2/losangeles/test_stops/'
# agent_anomaly_days_dict = np.load(os.path.join(train_dataset_folder, "preprocess/agent_anomaly_days_dict.npy"), allow_pickle=True).item()
# abnormal_agent_id_list = list(agent_anomaly_days_dict.keys())
# print('abnormal_agent_id_list', abnormal_agent_id_list)
# # print('abnormal_agent_id_list', type(abnormal_agent_id_list[0]))  # 229918, 'numpy.int32'

# exit()

# print_function(train_dataset_folder, agent_anomaly_days_dict)
# print('======================')
# print_function(test_dataset_folder, agent_anomaly_days_dict)

if __name__ == '__main__':

    start = time.time()
    args = sys.argv
    if len(args) == 3:
        train_dataset_folder = args[1]
        test_dataset_folder = args[2]
    else:
        raise Exception("no dataset")


    gts_path_dict = {
        '/home/jxl220096/data/hay/new_format/trial2/sanfrancisco/test_stops/' : '/home/jxl220096/data/hay/new_format/trial2/gts/sf_gts',
        '/home/jxl220096/data/hay/new_format/trial2/knoxville/test_stops/' : '/home/jxl220096/data/hay/new_format/trial2/gts/kx_gts',
        '/home/jxl220096/data/hay/new_format/trial2/losangeles/test_stops/' : '/home/jxl220096/data/hay/new_format/trial2/gts/la_gts',
        '/home/jxl220096/data/hay/new_format/trial2/singapore/test_stops/' : '/home/jxl220096/data/hay/new_format/trial2/gts/sp_gts',
    }

    gts_path = gts_path_dict[test_dataset_folder]
    still_plots(train_dataset_folder, gts_path)

# python test.py '/home/jxl220096/data/hay/new_format/trial2/sanfrancisco/train_stops/' '/home/jxl220096/data/hay/new_format/trial2/sanfrancisco/test_stops/'
# python test.py '/home/jxl220096/data/hay/new_format/trial2/knoxville/train_stops/' '/home/jxl220096/data/hay/new_format/trial2/knoxville/test_stops/'
# python test.py '/home/jxl220096/data/hay/new_format/trial2/losangeles/train_stops/' '/home/jxl220096/data/hay/new_format/trial2/losangeles/test_stops/'
# python test.py '/home/jxl220096/data/hay/new_format/trial2/singapore/train_stops/' '/home/jxl220096/data/hay/new_format/trial2/singapore/test_stops/'
