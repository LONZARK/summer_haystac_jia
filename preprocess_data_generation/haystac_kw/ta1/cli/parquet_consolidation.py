import click
import os
from glob import glob
from multiprocessing import Pool
from haystac_kw.utils.data_utils.loggers import log_consolidator
from collections import defaultdict
from tqdm import tqdm
import psutil

@click.command()
@click.argument('input_directory')
@click.argument('output_directory')
@click.option('--workers', default=psutil.cpu_count(), help='Number of processes to use.')
def cleanup(input_directory: str, output_directory: str, workers: int):
    """
    Script to consolidate a folder of parquet files writen
    out by agent_behavior_sim's logs to a single parquet per
    agent in the simulation.

    Assumes files are in the format {agent_id}.{#}.parquet

    Parameters
    ----------
    input_director : str
        Directory containing logs to be consolidated.
    output_directory : str
        Directory to save consolidated log files.
    workers : int
        Number of processes to use (defaults to number of system cpus).
    """
    

    print('This script will delete the original logs as it consolidates.')
    resp = input('Do you wish to continue? (type Yes): ')
    if resp != 'Yes':
        print('Confirmation not recieved to continue.')
        print('Exiting')
        exit()

    # Group parquets in input_directory
    # by agent id
    parquets = defaultdict(list)
    for parquet_file in glob(os.path.join(input_directory, '*.parquet')):
        agent_id = os.path.basename(parquet_file).split('.')[0]
        parquets[agent_id].append(parquet_file)

    # Create inputs for multiprocessed file consolidation
    args = []
    for agent_id, parquet_files in parquets.items():
        output_file = os.path.join(output_directory, f'{agent_id}.parquet')
        args.append((parquet_files, output_file,))

    # Make sure our output folder exists
    os.makedirs(output_directory, exist_ok=True)
    with Pool(workers) as pool:
        list(tqdm(pool.imap(log_consolidator, args), total=len(args)))
    print('Parquet Consolidation Complete')


if __name__ == "__main__":
    cleanup()
