import pandas as pd
from tqdm import tqdm
import os
from multiprocessing import Pool, Queue
import click

def writer(queue):

    while True:
        args = queue.get(block=True)
        if args is None:
            return

        df, agent, output_file = args
        df = df[df.agent_id==agent]
        df.to_parquet(output_file)

@click.command()
@click.argument('stop_point_csv')
@click.argument('event_log_csv')
@click.argument('output_folder')
def main(stop_point_csv, event_log_csv, output_folder):
    
    
    os.makedirs(output_folder, exist_ok=True)
    print('Loading Data')
    stop_points_df = pd.read_csv(stop_point_csv)
    event_log_df = pd.read_csv(event_log_csv)
    event_log_df['id'] = event_log_df['id'].astype(str)
    event_log_df['timestamp'] = pd.to_datetime(event_log_df['timestamp'], unit='s')
    stop_points_df['global_stop_points'] = stop_points_df['global_stop_points'].astype(str)
    agents = list(set(stop_points_df.agent_id.unique()).union(event_log_df.agent_id.unique()))

    print('Start Converting')
    timestamp = []
    agent_id = []
    event_type = []
    location_type = []
    location_uuid = []
    print('Converting Road Arrivals')
    arr = event_log_df[event_log_df.EventType == 'arrival']
    timestamp += list(arr.timestamp.values)
    agent_id += list(arr.agent_id.values)
    event_type += ['ARRIVE']*len(arr)
    location_type += ['ROAD_EDGE']*len(arr)
    location_uuid += list(arr.id.values)

    print('Converting Road Departures')
    dep = event_log_df[event_log_df.EventType == 'departure']
    timestamp += list(dep.timestamp.values)
    agent_id += list(dep.agent_id.values)
    event_type += ['DEPART']*len(dep)
    location_type += ['ROAD_EDGE']*len(dep)
    location_uuid += list(dep.id.values)

    print('Converting Stop Points')
    timestamp += list(stop_points_df.time_start.values)
    timestamp += list(stop_points_df.time_stop.values)
    agent_id += list(stop_points_df.agent_id.values)
    agent_id += list(stop_points_df.agent_id.values)
    event_type += ['ARRIVE']*len(stop_points_df)
    event_type += ['DEPART']*len(stop_points_df)
    location_type += ['STOP_POINT']*len(stop_points_df)*2
    location_uuid += list(stop_points_df.global_stop_points.values)
    location_uuid += list(stop_points_df.global_stop_points.values)
    
    print('Creating DataFrame')
    df = pd.DataFrame({
        "agent_id": agent_id,
        "timestamp": timestamp,
        "EventType": event_type,
        "LocationType": location_type,
        "LocationUUID": location_uuid,

    })
    print('Converting timestamps')
    df['timestamp'] = pd.to_datetime(df.timestamp, errors='coerce')
    df = df.dropna()
    df['timestamp'] = df['timestamp'].astype(str)
    print('Sorting DataFrame')
    df = df.sort_values(['agent_id','timestamp'])

    df.to_pickle('test.pkl')
    #df = pd.read_pickle('test.pkl')
    df = df.reset_index(drop=True)
    queue = Queue(64)
    pool = Pool(32, writer, (queue,))
    print('Writing To Disk')
    agent_index = pd.Index(df.agent_id)
    agents = df.agent_id.unique()
    for agent in tqdm(agents):
        try:
            sub = df.loc[agent_index.get_loc(agent)]#.to_parquet(os.path.join(output_folder, f'{agent}.parquet'))
        except:
            sub = df[df.agent_id==agent]#.to_parquet(os.path.join(output_folder, f'{agent}.parquet'))
        queue.put((sub.copy().reindex(), agent, os.path.join(output_folder, f'{agent}.parquet')))

    for i in range(32):
        queue.put(None)
    
    pool.close()
    pool.join()













if __name__ == "__main__":


    main()
