import click
import pandas as pd
from collections import defaultdict
import os
from tqdm import tqdm
from datetime import datetime
import pickle
@click.command()
@click.argument('filename')
@click.argument('output_file')
@click.option('--rank', default=2)
def temporal_variability(filename, output_file, rank):

    # Load df of unique stop points
    df = pd.read_csv(filename)
    df['time_start'] = pd.to_datetime(df['time_start'])
    df['time_stop'] = pd.to_datetime(df['time_stop'])
    df['arrival_time_s'] = [
                x.hour *
                60 *
                60 +
                x.minute *
                60 +
                x.second for x in df.time_start.dt.time.values]
    df['departure_time_s'] = [
                x.hour *
                60 *
                60 +
                x.minute *
                60 +
                x.second for x in df.time_stop.dt.time.values]
    # Get List of Agent Ids
    agent_ids = sorted(df.agent_id.unique())

    # Get visit count for each stop point
    loc_counts = df.groupby(['agent_id', 'unique_stop_point'])[
        'unique_stop_point'].value_counts().sort_values(ascending=False)

    data = defaultdict(list)
    for agent in tqdm(agent_ids):
        locations = loc_counts.loc[agent, :].head(rank).index.values
        data['agent_id'].append(agent)

        agent_sub = df[df.agent_id == agent]
        for i, location in enumerate(locations):
            location_sub = agent_sub[agent_sub.unique_stop_point == location]
            std_a = location_sub.arrival_time_s.std()
            std_d = location_sub.departure_time_s.std()
            data[f'std-arr-{i}'].append(std_a)
            data[f'std-dep-{i}'].append(std_d)
    if len(os.path.dirname(output_file)) > 0:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pickle.dump(data, open(output_file, 'wb'))


if __name__ == "__main__":

    temporal_variability()
