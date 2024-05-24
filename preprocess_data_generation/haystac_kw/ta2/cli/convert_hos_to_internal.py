import json
import geopandas as gpd
import pandas as pd
from shapely import wkt
import click
from uuid import uuid4
from pathlib import Path, PurePath
from haystac_kw.data.schemas.hos import HOS


@click.command()
@click.argument('input_spec_file_folder')
@click.argument('input_global_poi_table')
@click.argument('output_spec_file_folder')
@click.option('--distance-threshold', default=37.5)
def main(
        input_spec_file_folder,
        input_global_poi_table,
        output_spec_file_folder,
        distance_threshold):
    """
    Script for converting HOS to InternalHOS Format 
    for use with TA-2 algorithms.

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


if __name__ == "__main__":

    main()
