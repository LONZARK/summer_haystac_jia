from tqdm import tqdm
import pandas as pd
from importlib.resources import path
import os
from glob import glob
from tempfile import TemporaryDirectory
from shutil import copyfile
import docker
from docker.types import Mount
import pickle
from shapely.geometry import Point, LineString
from haystac_kw.utils.data_utils.data import get_unixtime
import re
import numpy as np
from multiprocessing import Pool

def build_fmm(force_rebuild=False):
    """
    Builds the fmm image for map matching, this function
    can also be used to assert that the image exists.
    If force_rebuild is set to False (default), then
    the function will simply return immediately if the
    image already exists.

    :param force_rebuild: Flag to force image rebuild, defaults to False
    :type force_rebuild: bool, optional
    """

    # Initialize docker client
    client = docker.client.from_env()

    # Determine if the image already exists
    try:
        _ = client.images.get('fmm')
        image_exists = True
    except docker.errors.ImageNotFound:
        # Image does not exist
        image_exists = False

    if image_exists and not force_rebuild:
        # Image already build and we don't want
        # to rebuild
        return

    # At this point we have established that the image
    # either doesn't exist already or we have specified
    # that we want to rebuild it

    # Get path to the docker file in the repo
    with path('haystac_kw.utils.data_utils.fmm', 'Dockerfile') as p:
        docker_file_path = p
    with path('haystac_kw.utils.data_utils.fmm', 'fmm.xml') as p:
        fmm_file_path = p
    with path('haystac_kw.utils.data_utils.fmm', 'ubodt.xml') as p:
        ubodt_file_path = p

    # Open a temporary directory to work in
    with TemporaryDirectory() as tmp_dir:

        # Copy docker file to temporary directory
        docker_file = os.path.join(tmp_dir, 'Dockerfile')
        copyfile(docker_file_path, docker_file)
        fmm_file = os.path.join(tmp_dir, 'fmm.xml')
        copyfile(fmm_file_path, fmm_file)
        ubodt_file = os.path.join(tmp_dir, 'ubodt.xml')
        copyfile(ubodt_file_path, ubodt_file)

        # Build the docker file
        client.images.build(path=tmp_dir, tag='fmm', quiet=False)


def run_map_matching(
        shape_file: str,
        gps_csv: str,
        output_file: str,
        working_dir=None):
    """
    Runs map matching of ULLT trajectories against a road network.
    For repeated computations of the same area specify `working_dir`
    to be some directory on your machine. This will allow a precomputed
    routing table to be persistent.

    :param shape_file: .shp file describing the road network
    :type shape_file: str
    :param gps_csv: csv file of trajectories as linestrings as WKT with
    columns id, timestamp, geometry
    :type gps_csv: str
    :param output_file: Location to save matching result as a text file.
    :type output_file: str
    :param working_dir: Location of working directory if you want
    a persistent routing table., defaults to None
    :type working_dir: str, optional
    """
    # Initialize docker client
    client = docker.client.from_env()

    # Ensure fmm image exists
    build_fmm(force_rebuild=True)

    with TemporaryDirectory() as working_dir_tmp:

        if working_dir is None:
            working_dir = working_dir_tmp
        else:
            os.makedirs(working_dir, exist_ok=True)
            working_dir = os.path.abspath(working_dir)

        # Initialize container with mounted dir
        mount_point = Mount('/home/workspace', working_dir, type='bind')
        container = client.containers.run(
            'fmm',
            entrypoint='tail -f /dev/null',
            detach=True,
            mounts=[mount_point],
            working_dir='/home/workspace')

        # Copy shape file to working directory
        shape_base = os.path.splitext(shape_file)[0]
        shape_files = glob(shape_base + '.*')
        for shape_file in shape_files:
            ext = os.path.splitext(shape_file)[-1]
            copyfile(shape_file, os.path.join(working_dir, 'roads' + ext))

        if not os.path.isfile(os.path.join(working_dir, 'ubodt.txt')):
            # Computing ubodt file
            container.exec_run('ubodt_gen /home/ubodt.xml')
        else:
            print('Using precomputed routes file.')

        # Copy trajectories into working dir
        copyfile(gps_csv, os.path.join(working_dir, 'gps.csv'))
        # Run matching
        container.exec_run('fmm /home/fmm.xml')
        # Copy matches back out of working dir to output file
        copyfile(os.path.join(working_dir, 'matches.txt'), output_file)

        container.stop()
        container.remove()


def convert_traj_fmm_sub(args):

    agent_id, fpath, stop_table = args
    # Read trajectories into memory
    df = pd.read_parquet(fpath)
    df = df.sort_values('timestamp')
    df['timestamp'] = get_unixtime(df['timestamp'].values)

    
    start_times = stop_table.time_stop.values[:-1]
    stop_times = stop_table.time_start.values[1:]
    res = []
    # Grab trajectory for each agent trip
    for start, stop in zip(start_times, stop_times):
        sub = df[(df.timestamp > start) & (df.timestamp < stop)]
        if len(sub) < 10:
            # Trip too short, skip it
            continue
        
        # Convert geometry to a linestring
        points = [
            Point(
                x, y) for x, y in zip(
                sub.longitude.values, sub.latitude.values)]
        
        res.append((LineString(points).wkt, ','.join(list(map(str, sub.timestamp))), agent_id, start, stop))
    return res
        

def convert_traj_fmm(stop_point_csv: str, traj_parquet_dir: str, output_dir: str):
    """
    Method for converting ULLT data into the input format for the FMM map matching
    algorithm

    :param stop_point_csv: CSV file containg stop points for all agents (agent_id, time_start, time_stop, point)
    :type stop_point_csv: str
    :param traj_parquet_dir: Directory containing parquet files for all of the agents ULLT data
    :type traj_parquet_dir: str
    :param output_dir: Directory to write gps.csv and path_map.pkl to
    :type output_dir: str
    """

    # Data structure for recovering trajectories in ULLT after matching
    path_map = {}

    # Data structure for writing to format for FMM
    data = {
        'id': [],
        'timestamp': [],
        'geometry': []
    }

    # Load stop points from csv
    stop_points = pd.read_csv(stop_point_csv)
    stop_points['time_start'] = get_unixtime(stop_points.time_start.values)
    stop_points['time_stop'] = get_unixtime(stop_points.time_stop.values)
    stop_points = stop_points.set_index(['agent_id'])

    # Build mapping of agent_id to trajectory parquet files
    print(f"looking for parquet files in {traj_parquet_dir}")
    trajectories = {
        int(
            re.findall(
                r'(\d+)',
                os.path.basename(x).split('.')[0])[0]): x for x in glob(
            os.path.join(
                traj_parquet_dir,
                '*.parquet'))}

    # if nothing found in the internal format - look for delta format
    if len(trajectories) < 1:
        # change to using the delta table format
        items = glob(traj_parquet_dir + '/*/*.parquet')
        print(f"deltatable parquet files: {len(items)}")
        trajectories = {}

        for itm in items:
            if itm.split('/')[-2] == '_delta_log':
                continue  # skip this info for delta table folder
            agent_id = itm.split('/')[-2].split('=')[-1]
            #print(agent_id)
            trajectories[int(agent_id)] = itm
    print(f"Processing {len(trajectories)} agents.")

    # Remove file if exists, otherwise we append and that
    # breaks fmm
    if os.path.isfile(os.path.join(output_dir, 'gps.csv')):
        os.remove(os.path.join(output_dir, 'gps.csv'))

    # Process agents one at a time
    i = 0
    first = True  # Flag to put us in append mode

    def get_args():
        for agent_id in tqdm(trajectories.keys(), total=len(trajectories)):
            if agent_id not in stop_points.index.values:
                continue
            # Get stop points for agent
            stop_table = stop_points.loc[[agent_id]]
            if len(stop_table) < 2:
                continue
            yield agent_id, trajectories[agent_id], stop_table

    with Pool(42) as pool:

        for res in pool.imap_unordered(convert_traj_fmm_sub, iter(get_args())):
            for result in res:
                geom, timestamp, agent_id, start, stop = result
                # Add data to output data structure
                data['geometry'].append(geom)
                data['id'].append(i)
                data['timestamp'].append(timestamp)
                path_map[i] = {
                    'agent_id': agent_id,
                    'start_time': start,
                    'stop_time': stop}
                i += 1

                if i % 10000 == 0:
                    # Memory could blow up
                    # Save to disk
                    df = pd.DataFrame(data)
                    data = {
                        'id': [],
                        'timestamp': [],
                        'geometry': []
                    }
                    print(f"saving gps.csv to: {output_dir}")
                    df.to_csv(
                        os.path.join(output_dir, 'gps.csv'),
                        sep=';',
                        mode='a',
                        index=False,
                        header=first)
                    first = False

    # Save remaining data to disk
    if len(data['id']) > 0:
        df = pd.DataFrame(data)
        df.to_csv(
            os.path.join(output_dir, 'gps.csv'),
            sep=';',
            mode='a',
            index=False,
            header=first)
    else:
        print(f"WARNING: no gps file created ({len(data['id'])})")
    # Save off path map for converting matches back
    pickle.dump(path_map, open(os.path.join(output_dir, 'path_map.pkl'), 'wb'))


def get_used_edges(matches_file: str):
    """
    Parse matches file from FMM

    :param matches_file: path to fmm matches file
    :type matches_file: str
    :return: used_edges, valid_tracks, edge_list, pgeom_list
    :rtype: tuple
    """
    valid_tracks = []
    # Load all of the matches from FMM
    with open(matches_file, 'r') as fin:
        cpath_idx = None
        opath_idx = None
        pgeom_idx = None
        id_idx = None
        used_edges = set()
        edge_list = []
        pgeom_list = []
        for line in fin:
            line = line.split(';')
            if cpath_idx is None:
                cpath_idx = line.index('cpath')
                opath_idx = line.index('opath')
                pgeom_idx = line.index('pgeom')
                id_idx = line.index('id')
            else:
                cpath = line[cpath_idx]
                opath = line[opath_idx]
                if ',' in cpath:
                    # used_edges = used_edges | set(map(int, cpath.split(',')))
                    # valid_tracks.append(int(line[id_idx]))
                    # edge_list.append(list(map(int, opath.split(','))))
                    # pgeom_list.append(line[pgeom_idx])
                    yield int(line[id_idx]), list(map(int, opath.split(','))), pgeom_list
    # used_edges = sorted(used_edges)
    # return used_edges, valid_tracks, edge_list, pgeom_list


def convert_matches_to_event(matches_file: str, path_map_file: str, output_csv: str):
    """
    Convert FMM matches file to event format as a csv

    :param matches_file: Path to FMM matches file
    :type matches_file: str
    :param path_map_file: Path to path_map.pkl file
    :type path_map_file: str
    :param output_csv: Path to output save file
    :type output_csv: str
    """
    
    # Parse matches file
    # used_edges, valid_tracks, edge_list, pgeom_list = get_used_edges(
    #     matches_file)
    # Load mapping from matches index
    path_map = pickle.load(open(path_map_file, 'rb'))

    # Convert to event format
    dfs = []
    for res in tqdm(get_used_edges(matches_file)):
        
        track_id, edges, pgeom = res
        # Load metadata about track
        path = path_map[track_id]
        start_time = path['start_time']
        stop_time = path['stop_time']
        timestamps = list(range(int(start_time), int(stop_time - 1)))
        agent_id = [path['agent_id']] * len(timestamps)
        gdf_points = pd.DataFrame(
            {'timestamp': timestamps, 'agent_id': agent_id, 'id': edges})

        # Convert to events
        edge_keys = np.array(edges)
        arrivals = np.concatenate(
            [np.array([True]), edge_keys[1:] != edge_keys[:-1]])
        departures = np.concatenate([edge_keys[:-1] != edge_keys[1:], [True]])
        event_type = [None] * len(gdf_points)
        event_type = np.where(arrivals, 'arrival', event_type)
        event_type = np.where(departures, 'departure', event_type)
        event_type = np.where(arrivals & departures, None, event_type)
        gdf_points['EventType'] = event_type
        gdf_points = gdf_points[['agent_id',
                                 'timestamp', 'EventType', 'id']].dropna()
        dfs.append(gdf_points)
    # Save off to disk
    df = pd.concat(dfs)
    df.to_csv(output_csv)


if __name__ == "__main__":

    convert_traj_fmm('../map_matching_test/singapore_stop_points.csv', '../map_matching_test/traj_parquet', '../map_matching_test/')
    run_map_matching(
        '../map_matching_test/roads.shp',
        '../map_matching_test/gps.csv',
        '../map_matching_test/matches.txt',
        working_dir='../map_matching_test/tmp')
    convert_matches_to_event('../map_matching_test/matches.txt', '../map_matching_test/path_map.pkl', '../map_matching_test/event_log.csv')
