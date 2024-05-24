import click
from pathlib import Path, PurePath
from haystac_kw.data.schemas.hos import HOS, InternalHOS
from haystac_kw.data.schemas.has import InternalHAS
from haystac_kw.utils.s3.s3_utils import cp_from_s3, cp_to_s3
import geopandas as gpd
import pandas as pd
from shapely import Point
import time
import os
import json
from shapely.geometry import mapping, GeometryCollection, shape


@click.command()
@click.argument('hos_directory')
@click.argument('internal_hos_directory')
@click.argument('internal_has_directory')
@click.argument('road_network_file')
@click.argument('stop_point_file')
@click.argument('output_directory')
@click.option('--debug', default=False)
def main(
        hos_directory: str,
        internal_hos_directory: str,
        internal_has_directory: str,
        road_network_file: str,
        stop_point_file: str,
        output_directory: str,
        debug: bool):
    """
    Convert a directory of Internal HAS to the HAS format
    output for the government.

    Parameters
    ----------
    hos_directory : str
        Directory containing HOS files
    internal_hos_directory : str
        Directory containing InternalHOS files
    internal_has_directory : str
        Directory containing InternalHAS files
    road_network_file : str
        Shapefile describing road network
    stop_point_file : str
        Parquet file containing all POI
    output_directory : str
        Directory to save HAS files
    debug : bool
        Switch to turn on debug mode which will save a road net file
    """
    start_time = time.time()

    # if s3 bucket is passed in as the path, copt locally
    hos_directory = Path(cp_from_s3(hos_directory))
    internal_hos_directory = Path(cp_from_s3(internal_hos_directory))
    internal_has_directory = Path(cp_from_s3(internal_has_directory))
    if road_network_file.startswith('s3://'):
        # copy the folder and then point at the local file
        s3par = os.path.dirname(road_network_file)
        base = os.path.basename(road_network_file)
        local_dir = cp_from_s3(s3par)
        road_network = os.path.join(local_dir, base)
    if stop_point_file.startswith('s3://'):
        # copy local
        s3par = os.path.dirname(stop_point_file)
        base = os.path.basename(stop_point_file)
        local_dir = cp_from_s3(s3par)
        stop_point_file = os.path.join(local_dir, base)

    # handle setting up a local folder for s3 upload at end
    if output_directory.startswith('s3://'):
        s3_output_dir = output_directory
        output_directory = "/tmp/output_dir"

    # make the output folder, if needed
    os.makedirs(output_directory, exist_ok=True)

    print(f"DEBUG MODE: {debug}")

    # Load stop point file
    stop_point_table = pd.read_parquet(stop_point_file)
    stop_point_table = gpd.GeoDataFrame(
        stop_point_table, geometry=[
            Point(
                x, y) for x, y in zip(
                stop_point_table.Longitude.values, stop_point_table.Latitude.values)])
    stop_point_table = stop_point_table.set_crs('EPSG:4326')
    print(f"loaded stop points: {time.time() - start_time}")

    # check to see if we have a debug file to load
    debug_file = "../temp_saved_road_network.json"
    if os.path.exists(debug_file) and debug:
        # open this instead of the long process
        print(f"loading debug file {debug_file}")
        # geopandas dataframe created from geojson
        road_network = gpd.read_file(debug_file)
        print(f"opened geojson: {time.time() - start_time}")
    else:
        # Load Road Network and dilate the road geometries to polygons
        road_network = gpd.read_file(road_network_file)
        print(f"loaded road network: {time.time() - start_time}")
        utm_crs = road_network.iloc[:100].estimate_utm_crs()
        print(f"estimate_utm_crs: {time.time() - start_time}")
        road_network = road_network.to_crs(utm_crs)
        print(f"to_crs: {time.time() - start_time}")
        road_network['geometry'] = [x.buffer(10)
                                    for x in road_network['geometry'].values]
        print(f"create geometries: {time.time() - start_time}")
        road_network = road_network.to_crs('EPSG:4326')
        print(f"to_crs2: {time.time() - start_time}")
        road_network['id'] = road_network['id'].astype(str)

        # save the debug file
        if debug:
            print(f"saving debug file {debug_file}")
            # save the geopandas dataframe
            road_network.to_file(debug_file, driver="GeoJSON")
    print(f"finished computing road network: {time.time() - start_time}")
    #print(f"road_network type = {type(road_network)}")
    #print(f"road_network = {road_network}")

    # Ensure output directory exists
    output_directory = Path(output_directory)
    output_directory.mkdir(exist_ok=True)

    # Load HOS and InternalHOS into memory
    hos_map = {}
    for hos_file in Path(hos_directory).glob('*.json'):
        hos = HOS.from_json(hos_file.read_text())
        hos_map[hos.objective_uid] = hos
    print(f"loaded hos map: {time.time() - start_time}")
    ihos_map = {}
    for ihos_file in Path(internal_hos_directory).glob('*.json'):
        ihos = InternalHOS.from_json(ihos_file.read_text())
        ihos_map[ihos.objective_uid] = ihos
    print(f"time for loading items in memory: {time.time() - start_time}")

    # Convert the InternalHAS
    ihas_files = Path(internal_has_directory).glob('*.json')
    for ihas_file in ihas_files:
        # Iterate through each has file and convert it

        # Load Internal HAS into memory
        ihas = InternalHAS.from_json(ihas_file.read_text())

        # Grab corresponding InternalHOS/HOS
        ihos = ihos_map[ihas.objective]
        hos = hos_map[ihas.objective]

        # Convert to HAS
        has = ihas.convert_to_external(
            hos, ihos, stop_point_table, road_network)

        # Write to disk
        output_directory.joinpath(ihas_file.name).write_text(has.model_dump_json())
    print(f"conversion time: {time.time() - start_time}")

    # copy up to s3 if the output was to a bucket
    if s3_output_dir:
        cp_to_s3(output_directory, s3_output_dir)

if __name__ == "__main__":
    main()
