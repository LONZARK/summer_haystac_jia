import click
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
import os
import pandas as pd
from haystac_kw.utils.data_utils.stop_points import (
    calculate_stop_points,
    calculate_agent_unique_stop_points,
    calculate_set_unique_stop_points
)
import geopandas as gpd
from shapely import wkt


def process_unique_stop_points(args):
    agent_id, sub, distance_thresh = args
    gdf_updated, agent_unique_stop_points = \
        calculate_agent_unique_stop_points(agent_id, sub, distance_thresh)
    return gdf_updated, agent_unique_stop_points


def process_stop_points(args):
    try:
        fname, distance_thresh, id_format = args
        # this format of agent id is used internally to this project
        if id_format == "internal":
            agent_id = os.path.basename(fname).split('.')[0].split('_')[1]
        else:
            # delta lake format - used in submissions
            agent_id = fname.split('/')[-2].split('=')[-1]
        # print(f"agent_id = {agent_id}")
        df = pd.read_parquet(fname)
        df = calculate_stop_points(agent_id, df,
                                   distance_threshold=distance_thresh)
        return df
    except BaseException as ex:
        print(f"{fname} : exception: {ex.with_traceback(None)}")
        return

@click.group()
def cli():
    pass

@cli.command()
@click.argument('input_folder')
@click.argument('output_file')
@click.option('--distance-thresh', default=37.5)
@click.option('--num-workers', default=16)
def stop_points(input_folder, output_file, distance_thresh, num_workers):
    parquets = list(glob(os.path.join(input_folder, '*.parquet')))
    id_format = "internal"  # default format for the ids of agents
    if len(parquets) < 1:
        # look for parquets in the sub folders in the case of deltatable
        parquets = list(glob(os.path.join(input_folder, '*/*.parquet')))
        id_format = "deltatable"  # deltalake table format for ids of agents
    parquets = [(x, distance_thresh, id_format) for x in parquets]
    dfs = []
    with Pool(num_workers) as pool:
        proc_iter = pool.imap_unordered(process_stop_points, parquets)
        first = True  # save the header the first time only
        for val in tqdm(proc_iter, total=len(parquets)):
            if val is None:
                continue
            dfs.append(val)
            # save the first 100 to check formatting
            if len(dfs) == 100 and first:
                dfs = pd.concat(dfs)
                dfs.to_csv(output_file, mode='a', header=first, index=False)
                dfs = []
                first = False
    print(f"dataframes: {len(dfs)}")
    dfs = pd.concat(dfs)
    dfs.to_csv(output_file, mode='a', header=first, index=False)


@cli.command()
@click.argument('filename')
@click.argument('output_folder')
@click.option('--distance-thresh', default=37.5)
@click.option('--num-workers', default=16)
def unique_stop_points(filename, output_folder, distance_thresh, num_workers):

    if filename.endswith('.csv'):
        df = pd.read_csv(filename)
        df['geometry'] = [wkt.loads(x) for x in df.geometry]
        df = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    else:
        raise ValueError

    processed_gdfs, unique_stop_points = [], []

    def loader():
        for agent_id in df.agent_id.unique():
            yield ((agent_id, df[df.agent_id == agent_id], distance_thresh))

    with Pool(num_workers) as pool:
        for result in tqdm(
            pool.imap(
                process_unique_stop_points, loader()), total=len(
                df.agent_id.unique())):
            gdf_updated, agent_unique_stop_points = result
            processed_gdfs.append(gdf_updated)
            unique_stop_points.append(agent_unique_stop_points)

    processed_gdf = pd.concat(processed_gdfs)
    unique_stop_points = pd.concat(unique_stop_points)

    os.makedirs(output_folder, exist_ok=True)
    processed_gdf.to_csv(os.path.join(output_folder,
                                      os.path.basename(filename)))
    unique_stop_points.to_csv(
        os.path.join(output_folder,
                     'agent_unique_stop_point_table.csv'))


@cli.command()
@click.argument('filename')
@click.argument('output_folder')
@click.option('--distance-thresh', default=37.5)
@click.option('--stop-point-file',
              help='Stop point file to merge with', default=None)
def set_unique_stop_points(filename, output_folder,
                           distance_thresh, stop_point_file):

    if filename.endswith('.csv'):
        print('Loading Data Frame')
        df = pd.read_csv(filename)
        df['point'] = [wkt.loads(x) for x in df.point]
        df = gpd.GeoDataFrame(df, geometry='point', crs='EPSG:4326')
    else:
        raise ValueError

    gdf, set_unique_stop_points = calculate_set_unique_stop_points(
        df, distance_thresh)

    gdf.to_csv(
        os.path.join(output_folder,
                     'agent_set_unique_stop_point_table.csv'))
    set_unique_stop_points.to_csv(
        os.path.join(output_folder,
                     'set_unique_stop_points.csv'))
    
    if stop_point_file is not None:
        stop_points = pd.read_csv(stop_point_file)
        stop_points = stop_points.merge(
            gdf[['unique_stop_point', 'global_stop_points']],
            on='unique_stop_point')
        stop_points.to_csv(stop_point_file)


if __name__ == "__main__":
    cli()