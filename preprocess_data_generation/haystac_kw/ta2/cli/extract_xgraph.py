import geopandas as gpd
from networkx import MultiDiGraph
import pickle
import click
import numpy as np
import re


@click.command()
@click.argument('input_edge_file')
@click.argument('input_node_file')
@click.argument('output_file')
@click.option('--speed-regex', type=str, default='(\\d+.?\\d*) mph')
@click.option('--road-distance-conversion', type=float,
              default=1.60934 * 1000 / 3600)
@click.option('--default-speed', type=float, default=55)
def main(
        input_edge_file: str,
        input_node_file: str,
        output_file: str,
        speed_regex: str,
        road_distance_conversion: float,
        default_speed: str):
    """
    Extract a networkx DiGraph representing a road network
    from shape files.

    Parameters
    ----------
    input_edge_file : str
        Shapefile representing road network edges
    input_node_file : str
        Shapefile representing road network nodes
    output_file : str
        Location to save networkx DiGraph as *.pkl
    speed_regex: str
        Regex for getting road speeds from input_edge_file
    road_distance_conversion: float
        Conversion factor to change speed to m/s
    default_speed: float
        Speed (pre conversion to m/s) to use when no speed is available
        in the edge file for that road
    """

    # Load edge file into memeory
    gdf = gpd.read_file(input_edge_file)

    # Estimate UTM zone for the dge file
    utm = gdf.iloc[:100].estimate_utm_crs()
    gdf = gdf.to_crs(utm)
    # Load node file into memory
    gdf_nodes = gpd.read_file(input_node_file)

    # Initialize the DiGraph
    graph = MultiDiGraph()

    # Grab node ids and their lat/long's
    ids = gdf_nodes.osmid.values
    x = gdf_nodes.x.values
    y = gdf_nodes.y.values

    # Add each node to the graph
    for i in range(len(ids)):
        graph.add_node(ids[i], coord=[x[i], y[i]])

    # Grab u,v,id arrays from roads dataframe
    u = gdf.u.values
    v = gdf.v.values
    id = gdf.id.values

    # compile regex for extracting road speeds
    pattern = re.compile(speed_regex)

    # Calculate travel time for each road edge in seconds
    gdf['length'] = [x.length for x in gdf.geometry.values]
    gdf['maxspeed'] = [pattern.findall(str(x)) for x in gdf.maxspeed.values]
    gdf['maxspeed'] = [
        x[0] if len(x) > 0 else default_speed for x in gdf.maxspeed.values]
    weight = gdf['length'].values.astype(
        float) / (gdf.maxspeed.values.astype(float) * road_distance_conversion)
    # Add road edges to the graph
    for i in range(len(u)):
        graph.add_edge(u[i], v[i], id=str(id[i]), weight=weight[i])

    # Save graph to disk
    pickle.dump(graph, open(output_file, 'wb'))


if __name__ == "__main__":

    main()
