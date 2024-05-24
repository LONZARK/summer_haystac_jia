import pandas as pd
import geopandas as gpd
from glob import glob
import os
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import click
import pytz


@click.command()
@click.argument('parquet_folder')
@click.argument('road_network_shape_file')
@click.argument('stop_point_table')
@click.argument('sim_time_start_utc')
@click.argument('sim_time_stop_utc')
def validate_data(
        parquet_folder,
        road_network_shape_file,
        stop_point_table,
        sim_time_start_utc,
        sim_time_stop_utc):
    """
    Validate the event files for the semantic
    TA-2 representations.

    Parameters
    ----------
    parquet_folder : str
        Path to directory containing TA-2 semantic trajectory parquets
    road_network_shape_file : str
        Path to road network .shp file
    stop_point_table : str
        Path to StopPoints.parquet
    sim_time_start_utc : int
        UTC timestamp of simulation start time
    sim_time_stop_utc : int
        UTC timestamp of simulation stop time
    """

    # Convert time bounds to datetimes
    sim_time_start_utc = datetime.fromtimestamp(
        sim_time_start_utc, tz=pytz.UTC)
    sim_time_stop_utc = datetime.fromtimestamp(sim_time_start_utc, tz=pytz.UTC)

    # Load files into memeory
    road_net = gpd.read_file(road_network_shape_file, crs='EPSG:4326')
    parquet_files = glob(os.path.join(parquet_folder, '*.parquet'))
    stop_point_table = pd.read_parquet(stop_point_table)

    # Build a fast mapping which returns true
    # for all valid LocationUUID and reaod network ids
    valid_uuid = defaultdict(lambda: False)
    for location_uuid in stop_point_table.LocationUUID.unique():
        valid_uuid[str(location_uuid)] = True
    for location_uuid in road_net['id'].values:
        valid_uuid[str(location_uuid)] = True

    # Validate the parquet files
    for event_log in tqdm(parquet_files):
        # event_log is a parquet file for a single
        # trajectory

        # Load parquet
        event_log = Path(event_log)
        events = pd.read_parquet(event_log)

        # Assert one agent per file
        assert (len(events.agent_id.unique()) == 1)
        assert (str(events.agent_id.unique()[0]) == event_log.stem)

        # Assert timestamps ascending and in test zone
        ts = pd.to_datetime(events['timestamp'].values, utc=True)
        assert (all(ts[1:] >= ts[:-1]))
        assert (all(ts >= sim_time_start_utc))
        assert (all(ts <= sim_time_stop_utc))

        # Assert all UUID are valid
        assert (all([valid_uuid[str(x)] for x in events.LocationUUID.values]))


if __name__ == "__main__":

    validate_data()
