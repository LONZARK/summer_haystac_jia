import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch, Arrow
import os
import numpy as np
import pandas as pd
import tilemapbase
import pickle
import json
import time
import sys
from datetime import datetime
import math

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

def find_parquet_files(directory):
    """
    Find and return a list of full paths to .parquet files in the given directory.

    :param directory: The path to the directory to search.
    :return: A list of full paths to .parquet files.
    """
    parquet_files = []
    # Walk through all files and folders within the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".parquet"):
                full_path = os.path.join(root, file)
                parquet_files.append(full_path)
    return parquet_files


def daily_animation(frame, ax_train, dates_coords_list, date_group, colors, tiles, extent):
    plotter = tilemapbase.Plotter(extent, tiles, height=600)
    plotter.plot(ax_train, tiles, alpha=0.5)

    for i, coords_list in enumerate(dates_coords_list):
        if len(coords_list) == 0:
            continue  # Skip if no coordinates
        
        print('coords_list', coords_list)
        exit()
        temp_lat = [coords[0] for coords in coords_list]
        temp_lon = [coords[1] for coords in coords_list]
        
        temp_projected_lon, temp_projected_lat = project_coordinates_to_df(temp_lat, temp_lon)
        num_coords_to_show = min(len(temp_projected_lon), frame + 1)

        i_date = list(date_group.groups)[i]
        color = colors[i_date]

        ax_train.plot(temp_projected_lon[:num_coords_to_show], temp_projected_lat[:num_coords_to_show],
                      color=color, linewidth=2.7, alpha=1.0, zorder=1)

        if num_coords_to_show > 1:
            arrow_start_lon = temp_projected_lon[num_coords_to_show - 2]
            arrow_start_lat = temp_projected_lat[num_coords_to_show - 2]
            arrow_end_lon = temp_projected_lon[num_coords_to_show - 1]
            arrow_end_lat = temp_projected_lat[num_coords_to_show - 1]

            arrow = FancyArrowPatch((arrow_start_lon, arrow_start_lat), (arrow_end_lon, arrow_end_lat),
                                    arrowstyle='->', color=color, linewidth=2, mutation_scale=20, zorder=2)
            ax_train.add_patch(arrow)

    return ax_train.figure

def anomaly_agents_daily_animation(train_dataset_folder, test_dataset_folder):

    output_folder_path = os.path.join(test_dataset_folder, 'plots/anomaly_agents_daily_trajectories/')
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    print(output_folder_path)

    #------------ read file ----------------
    with open(os.path.join(test_dataset_folder + "preprocess/loc_coord_dict.pickle"), 'rb') as fp:
        loc_coord_dict = pickle.load(fp)
    agent_anomaly_days_dict = np.load(os.path.join(train_dataset_folder, "preprocess/agent_anomaly_days_dict.npy"), allow_pickle=True).item()
    abnormal_agent_id_list = list(agent_anomaly_days_dict.keys())

    #------------ read parquet file for each agents ------------
    # abnormal_agent_id_list = abnormal_agent_id_list[:10]
    for temp_agent in abnormal_agent_id_list:

        # --- read full trajectory ---
        temp_parquet_path = os.path.join(test_dataset_folder, 'event_logs', str(temp_agent) + '.parquet')
        df = pd.read_parquet(temp_parquet_path)
        if df['agent_id'].nunique() > 1:
            df = df.loc[df['agent_id'] == temp_agent]
        df = df.drop_duplicates(keep='first')
        df = df.drop_duplicates(subset=["timestamp", "EventType"], keep='first')
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df['day'] = df['datetime'].dt.date
        df = df.sort_values(by=['timestamp', 'EventType'], ascending=[True, False]).reset_index()

        # --- read anomaly trajectory ---
        anomaly_days = set(agent_anomaly_days_dict[temp_agent])
        all_days = set(df['day'])
        # print('anomaly_days', anomaly_days)
        # print('all_days', all_days)

        # --- generate different color for different day ---
        date_group = df.groupby(by='day')
        num_date = len(date_group)
        colors = {}
        for i_date, _ in date_group:
            if i_date in anomaly_days:
                colors[i_date] = np.array([1.0, 0.0, 0.0])  # Red for anomaly days
            else:
                color = np.random.random(3)
                while color[0] > 0.8 and color[1] < 0.3 and color[2] < 0.3:  # Avoid generating colors that are too close to red
                    color = np.random.random(3)
                colors[i_date] = color
        # --- get daily coordinates ---
        max_tra_len = 0
        dates_coords_list = []
        dates_anomaly_coords_list = []
        for i_date, i_group in date_group:
            temp_coords = []
            anomaly_coords = []
            for i in range(0, len(i_group), 2):
                temp_coord = loc_coord_dict[i_group.iloc[i]['LocationUUID']]
                if i_date in anomaly_days:
                    anomaly_coord = loc_coord_dict[i_group.iloc[i]['LocationUUID']]
                else:
                    anomaly_coord = []
                if isinstance(temp_coord[0], list):
                    temp_coords += temp_coord
                    anomaly_coords += anomaly_coord
                else:
                    temp_coords.append(temp_coord)
                    anomaly_coords.append(temp_coord) 
              
            temp_coords = np.array(temp_coords)
            dates_coords_list.append(temp_coords)
            anomaly_coords = np.array(anomaly_coords)
            dates_anomaly_coords_list.append(anomaly_coords)
            temp_tra_len = len(temp_coords)
            if temp_tra_len > max_tra_len:
                max_tra_len = temp_tra_len
        max_tra_len = min(max_tra_len, 7)
        
        
        # --- calculate extend ---
        all_coords = np.concatenate(dates_coords_list)
        all_coords_lat = all_coords[:, 0]
        all_coords_lon = all_coords[:, 1]
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
                            frames=7,
                            interval=500, repeat=False,
                            fargs=(ax_train, dates_coords_list, date_group, colors, tiles, extent))

        fig.legend(fontsize=20)
        fig.tight_layout()
        fig.suptitle('agent ' + str(temp_agent), fontsize=20)
        output_filename = str(temp_agent) + '.gif'
        gifoutput_path = os.path.join(output_folder_path, output_filename)
        print('output_filename:', output_filename)
        ani.save(gifoutput_path, writer='pillow')
        plt.close('all')

if __name__ == '__main__':

    start = time.time()
    args = sys.argv
    if len(args) == 3:
        train_dataset_folder = args[1]
        test_dataset_folder = args[2]
    else:
        raise Exception("no dataset")

    anomaly_agents_daily_animation(train_dataset_folder, test_dataset_folder)

    print(f"total runtime: {time.time() - start}s")


# python daily_animation_jia.py '/home/jxl220096/data/hay/new_format/trial2/sanfrancisco/train_stops/' '/home/jxl220096/data/hay/new_format/trial2/sanfrancisco/test_stops/'
# python daily_animation_jia.py '/home/jxl220096/data/hay/new_format/trial2/knoxville/train_stops/' '/home/jxl220096/data/hay/new_format/trial2/knoxville/test_stops/'
# python daily_animation_jia.py '/home/jxl220096/data/hay/new_format/trial2/losangeles/train_stops/' '/home/jxl220096/data/hay/new_format/trial2/losangeles/test_stops/'
# python daily_animation_jia.py '/home/jxl220096/data/hay/new_format/trial2/singapore/train_stops/' '/home/jxl220096/data/hay/new_format/trial2/singapore/test_stops/'
