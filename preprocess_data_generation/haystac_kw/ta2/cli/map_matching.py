import click
from haystac_kw.utils.data.map_matching import *
import os

@click.command()
@click.argument('stop_point_csv')
@click.argument('ullt_parquet_folder')
@click.argument('road_shp_file')
@click.argument('output_directory')
def main(stop_point_csv, ullt_parquet_folder, road_shp_file, output_directory):
    """
    Script for matching ULLT data to a road network and exporting it in an event
    based file format.

    stop_point_csv - .csv file containing stop points for all agents
    ullt_parquet_folder - directory containing ULLT parquet (one per agent)
    road_shp_file - .shp file describing the road network
    output_directory - location to save results

    The output file will be located at {output_directory}/event_log.csv
    and it will be a csv file with columns:
    'agent_id','timestamp', 'EventType', 'id'

    agent_id - UUID of the agent
    timestamp - unix epoch of event
    EventType - Arrival or Departure
    id - UUID of road edge
    """ 
    
    os.makedirs(output_directory, exist_ok=True)
    convert_traj_fmm(stop_point_csv, ullt_parquet_folder, output_directory)
    run_map_matching(
        road_shp_file,
        os.path.join(output_directory, 'gps.csv'),
        os.path.join(output_directory, 'matches.txt'),
        working_dir=os.path.join(output_directory, 'fmm'))
    convert_matches_to_event(os.path.join(output_directory, 'fmm', 'matches.txt'),
                             os.path.join(output_directory, 'path_map.pkl'),
                             os.path.join(output_directory, 'event_log.csv')
                             )


if __name__ == "__main__":
    main()