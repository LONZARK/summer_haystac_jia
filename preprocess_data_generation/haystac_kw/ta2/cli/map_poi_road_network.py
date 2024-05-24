from haystac_kw.utils.data_utils.road_network import map_buildings
import pandas as pd
import geopandas as gpd
from shapely import Point
import click
from tqdm import tqdm

@click.command()
@click.argument('stop_point_file')
@click.argument('road_network_file')
@click.argument('output_file')
@click.option('--maximum-distance', type=float, default=50)
@click.option('--road-id-column', type=str, default='id')
def main(
        stop_point_file: str,
        road_network_file: str,
        output_file: str,
        maximum_distance: float,
        road_id_column: str):
    """
    Create a pickle file mapping poi to edges of the road network for using the OfflineMapping
    object.

    Parameters
    ----------
    stop_point_file : str
        Parquet file containing stop points
    road_network_file : str
        Shapefile describing road network (must have an `id` column)
    output_file : str
        Location to save mapped poi to disk
    maximum_distance : float
        Maximum distance from the road network to consider a POI
    road_id_column : str
        ID column in the road network
    """
    # Load StopPoints.parquet
    poi = pd.read_parquet(stop_point_file)
    len_poi = len(poi)
    # Intialize shapely geometries
    centroid = [Point(x, y) for x, y
                in zip(poi.Longitude.values, poi.Latitude.values)]

    # Grab id_column
    id_building = poi.LocationUUID

    # build geopandas dataframe
    gdf = gpd.GeoDataFrame({'id': id_building, 'geometry': centroid},
                           geometry='geometry', crs='EPSG:4326')

    # load road network into meory
    road_network = gpd.read_file(road_network_file)
    road_network['id'] = road_network[road_id_column].values

    # Map poi to the road network
    mapping = map_buildings(
        gdf,
        road_network,
        maximum_distance=maximum_distance)

    # rename columns
    mapping['building_centroid'] = mapping['geometry_y'].values
    mapping['building_poly'] = mapping['geometry_y'].values

    # explicitly select columns to save
    mapping = mapping[['id_building',
                       'id_road',
                       'length_along_edge',
                       'building_centroid',
                       'building_poly']]
    
    # Make sure we haven't lost any poi
    assert(len(mapping) == len_poi), f'Found {len(mapping)} rows in output, expected {len_poi}, try increasing maximum-distance'
    # save poi to disk
    poi = pd.read_parquet(stop_point_file)
    assert all([x in  mapping.id_building.values for x in tqdm(poi.LocationUUID.values)]), 'Expected all POI to be present in output, try increasing maximum-distance'
    mapping.to_pickle(output_file)


if __name__ == "__main__":

    main()
