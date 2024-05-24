import uuid
import argparse
import json
from pathlib import Path, PurePath

from shapely import wkt, Point
import pandas as pd
import geopandas as gpd

from haystac_kw.utils.data_utils.data import SUMOData
from haystac_kw.data.schemas.hos import HOS

def has2internalhas(
        input_spec_file_folder, 
        sim_stop_points_pq_path,
        output_spec_file_folder,
        distance_threshold):
    """
    Method for converting Hide Specification to internal Hide
    Specification Format for use with TA-2 algorithms.

    Each location in the HOS is originally specified as a point or polygon.
    What we use is an internal list of stop points. So for each polygon
    this script takes the centroid and checks if it is within a threshold
    distance to a known set of stop points. If it is then the polygon
    is replaced with the UUID of the stop point in the HAS. If it is
    not then the centroid is added to the list of stop points and then
    its UUID is used. Finally the modified HOS and the new stop
    points are saved in output_spec_file_folder.

    Parameters
    ----------
    input_spec_file_folder : str
        Folder containing HAS jsons
    sim_stop_points_pq_path : str
        Path of the existing simulation stop points with .parquet extension
    output_spec_file_folder : str
        Folder to save off modified HAS
    distance_threshold : float
        Maximum distance to stop point
    """

    # Create output directory
    output_spec_file_folder = Path(output_spec_file_folder)
    output_spec_file_folder.mkdir(exist_ok=True)

    # Load stop points into memory
    poi_table = pd.read_parquet(sim_stop_points_pq_path)
    poi_table['global_stop_points'] = poi_table.LocationUUID.values 
    poi_table['point'] = [Point(x) for x in zip(poi_table['Longitude'].values, poi_table['Latitude'].values)]
    poi_table = gpd.GeoDataFrame(poi_table, geometry='point', crs='EPSG:4326')

    for input_spec_file in Path(input_spec_file_folder).glob('*.json'):
        # Load up input spec file
        spec = json.load(open(input_spec_file, 'r'))
        output_spec_file = Path(
            output_spec_file_folder,
            PurePath(input_spec_file).name)

        # Grab POI geometries
        loc_id, geometry = [], []
        for i, movement in enumerate(spec['movements']):
            for itinerary in movement["itineraries"]: 
                for action in itinerary["itinerary"]:
                    name = list(action.keys())[0]
                    if 'location' not in action[name].keys(): 
                        continue
                    geo_json = json.dumps(action[name]['location'])
                    loc_id.append(i)
                    geometry.append(
                        gpd.read_file(
                            geo_json,
                            driver='GeoJSON').geometry.values[0].centroid)

        building_gdf = gpd.GeoDataFrame({'id': loc_id, 'geometry': geometry},
                                        geometry='geometry',
                                        crs='EPSG:4326')
        
        # Convert to cartesian coordinate system
        utm = building_gdf.estimate_utm_crs()
        building_gdf = building_gdf.to_crs(utm)
        poi_table = poi_table.to_crs(utm)

        # Spatially join POI geometries with known stop points
        building_mapping = building_gdf.sjoin_nearest(
            poi_table[['point', 'global_stop_points']],
            max_distance=distance_threshold,
            how='left',
            distance_col='distance')
        while (len(building_mapping[['global_stop_points']].dropna()) < len(
                building_mapping)):
            # Here we have a case were there are POI geometries that aren't
            # within the distance threshold to be considered one of the known
            # stop points. While this is true we add the first unkown geometry's
            # stop point to the list of known stop points and run spatial join
            # again.

            # Grab first unkown POI geometry
            sub = building_mapping[building_mapping['global_stop_points'].isna()]
            row = sub.iloc[0]

            # Add its centroid to the list of known stop points
            poi_table = pd.concat([poi_table,
                                   gpd.GeoDataFrame({'point': [row.geometry],
                                                     'global_stop_points':[str(uuid.uuid4())]},
                                                    geometry='point',
                                                    crs=utm)],
                                  ignore_index=True)

            # Spatially join POI geometries with known stop points
            building_mapping = building_gdf.sjoin_nearest(
                poi_table, max_distance=37.5, how='left', distance_col='distance')

        # Replace geometries with edge/pos along edge in road network
        # building_mapping = building_mapping.set_index('id')
        # for i, event in enumerate(spec['events']):
        #     row = building_mapping.loc[i]
        #     event['location'] = row.global_stop_points

        counter = 0

        for i, movement in enumerate(spec['movements']):
            for itinerary in movement["itineraries"]: 
                for action in itinerary["itinerary"]:
                    name = list(action.keys())[0]
                    if 'location' not in action[name].keys(): 
                        continue
                    row = building_mapping.loc[counter]
                    action[name]["location"] = row.global_stop_points
                    counter += 1
                    
        # Save modified spec to disk
        json.dump(spec, open(output_spec_file, 'w'), indent=2)

    # Save table of known stop points in sharable format
    poi_table = poi_table.to_crs('EPSG:4326')
    poi_table['LocationUUID'] = poi_table.global_stop_points.values
    poi_table['Latitude'] = [x.y for x in poi_table.point.values]
    poi_table['Longitude'] = [x.x for x in poi_table.point.values]
    del poi_table['point']
    del poi_table['global_stop_points']
    poi_table.to_parquet(
        output_spec_file_folder.joinpath('StopPoints.parquet'))


def hos2internal_hos(
        input_spec_file_folder,
        input_global_poi_table,
        output_spec_file_folder,
        distance_threshold):
    """
    Script for converting Hide Specification to internal Hide
    Specification Format for use with TA-2 algorithms.

    Each location in the HOS is originally specified as a polygon.
    What we use is an internal list of stop points. So for each polygon
    this script takes the centroid and checks if it is within a threshold
    distance to a known set of stop points. If it is then the polygon
    is replaced with the UUID of the stop point in the HAS. If it is
    not then the centroid is added to the list of stop points and then
    its UUID is used. Finally the modified HOS and the new stop
    points are saved in output_spec_file_folder.

    Parameters
    ----------
    input_spec_file_folder : str
        Folder containing HOS json's
    input_global_poi_table : str
        CSV file containing Set Unique Stop Points with columns `point` and `global_stop_points`
    output_spec_file_folder : str
        Folder to save off modified HOS
    distance_threshold : float
        Maximum distance to stop point
    """

    # Create output directory
    output_spec_file_folder = Path(output_spec_file_folder)
    output_spec_file_folder.mkdir(exist_ok=True)

    # Load stop points into memory
    poi_table = pd.read_csv(input_global_poi_table)
    poi_table = poi_table[['point', 'global_stop_points']]
    poi_table = poi_table.drop_duplicates(['global_stop_points'])
    poi_table['point'] = [wkt.loads(x) for x in poi_table.point.values]
    poi_table = gpd.GeoDataFrame(poi_table, geometry='point', crs='EPSG:4326')

    for input_spec_file in Path(input_spec_file_folder).glob('*.json'):
        # Load up input spec file
        spec = HOS.from_json(input_spec_file.read_text())
        output_spec_file = Path(
            output_spec_file_folder,
            PurePath(input_spec_file).name)

        internal_spec, poi_table = spec.convert_to_internal(poi_table,
                                                            distance_threshold=distance_threshold)

        # Save modified spec to disk
        output_spec_file.write_text(json.dumps(json.loads(internal_spec.model_dump_json()), indent=2))

    # Save table of known stop points in sharable format
    poi_table = poi_table.to_crs('EPSG:4326')
    poi_table['LocationUUID'] = poi_table.global_stop_points.values
    poi_table['Latitude'] = [x.y for x in poi_table.point.values]
    poi_table['Longitude'] = [x.x for x in poi_table.point.values]
    del poi_table['point']
    del poi_table['global_stop_points']
    poi_table.to_parquet(
        output_spec_file_folder.joinpath('StopPoints.parquet'))

def simtime_to_realworld_time(seconds : int) -> str:
    """Converts simulation time (s) to a string version of real world time. Output format follows HH:MM:SS.

    Parameters
    ----------
    seconds : int
        number of seconds to convert

    Returns
    -------
    str
        the input seconds as a human-readable string
    """
    hours = (seconds // 3600) % 24
    minutes = (seconds % 3600) // 60
    seconds = (seconds % 60)
    
    time = "%02dh%02dm%02s" % (hours, minutes, seconds)
    return time 

def realworld_time_to_simtime(time : str) -> int:
    """Converts real world time string represenation to number of seconds.
    This function assumes time is represented as [HH]h[SS].

    Parameters
    ----------
    time : str
        Formatted time; [HH]h[SS]

    Returns
    -------
    int
        number of seconds the string represents
    """
    seconds = [float(x) for x in time.split("h")]
    seconds = 3600 * seconds[0] + 60 * seconds[1]
    return seconds 

def sumo2parquet(args):
    """Converts SUMO fcd output file to ULLT parquet files. This is the entrypoint to the process.

    Parameters
    ----------
    args : argparser.Namespace
        argparse arguments; run with --help to view options.
    """
    target_dir = args.tgt_dir if args.tgt_dir is not None else "outputs/parquet"

    converter = SUMOData(args.src_dir)
    converter.export(target_dir, format='parquet')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-dir', '-s',
                            required=True, 
                            help='path to read sumo simulation data directory')
    parser.add_argument('--tgt-dir', '-t',
                            default=None,
                            help='output path to write converted ULLT simulation files')
    parser.add_argument('--conversion', '-c',
                            default='sumo2ullt',
                            choices=['sumo2ullt'],
                            help='path to write converted ULLT simulation files')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    
    args = parse_args()
    sumo2parquet(args)