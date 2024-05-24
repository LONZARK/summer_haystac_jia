import pandas as pd
import click
from datetime import datetime,timedelta
from tqdm import tqdm
import os

@click.command()
@click.argument('agent_stops')
@click.argument('set_unique_stop_point_map')
@click.argument('output_file')
def calculate_encounter_frequency(agent_stops, set_unique_stop_point_map, output_file):
    
    df = pd.read_csv(agent_stops)
    df_map = pd.read_csv(set_unique_stop_point_map)
    df = df.merge(df_map[['unique_stop_point','global_stop_points']], on=['unique_stop_point'])
    
    # Compute day/hour bins for calculating encounters
    df['time_start'] = pd.to_datetime(df['time_start'])
    df['time_stop'] = pd.to_datetime(df['time_stop'])
    df['day_start'] = [x.day for x in df.time_start.dt.date.values]
    df['hour_start'] = [x.hour for x in df.time_start.dt.time.values]
    df['day_stop'] = [x.day for x in df.time_stop.dt.date.values]
    df['hour_stop'] = [x.hour for x in df.time_stop.dt.time.values]

    data = {'agent_id':[],'day':[],'hour':[],'location':[]}
    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        cur_time = row.time_start
        while cur_time<row.time_stop:
            data['agent_id'].append(row.agent_id)
            data['day'].append(cur_time.date().day)
            data['hour'].append(cur_time.time().hour)
            data['location'].append(row.global_stop_points)
            cur_time+=timedelta(hours=1)
    df = pd.DataFrame(data)

    #Calculate potential encounters
    encounters = df.merge(df, on=['day','hour','location'])
    encounters = encounters[encounters.agent_id_x!=encounters.agent_id_y]
    encounters = encounters[['agent_id_x','agent_id_y']].groupby(['agent_id_x','agent_id_y']).value_counts().to_frame()
    encounters = encounters[[x[0]<x[1] for x in encounters.index]]
    if len(os.path.dirname(output_file)) > 0:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    encounters.to_pickle(output_file)

if __name__ == "__main__":

    calculate_encounter_frequency()