import geopandas as gpd
from shapely.geometry import GeometryCollection
from haystac_kw.ta1.agent_behavior_sim.map import Map
from tqdm import tqdm
from collections import defaultdict
from pyrosm import OSM
from argparse import ArgumentParser
import numpy as np
import json
np.float = float

def extract_aoi(net_xml_file, osm_file):

    # Load SUMO road network
    city_map = Map(net_xml_file)
    # Create mapping of valid driving edges
    is_valid = defaultdict(lambda: False)
    for edge in city_map.edges:
        is_valid[int(edge.split('#')[0])] = True

    # Load geometry from the OSM
    osm = OSM(osm_file) 
    gdf = osm.get_network('driving').set_crs('EPSG:4326')
    # Filter out invalid roads
    gdf = gdf.loc[[is_valid[x] for x in tqdm(gdf.id.values)]]
    # Take convex hull and dilate
    hull = GeometryCollection(gdf.geometry.values).convex_hull.buffer(.01)
    # Write to file
    gpd.GeoSeries(hull).set_crs('EPSG:4326').to_file('aoi.geojson')

    # Simplify geojson by removing geopandas stuff
    aoi = json.load(open('aoi.geojson', 'r'))
    aoi = aoi['features'][0]['geometry']
    json.dump(aoi, open('aoi.geojson', 'w'))


if __name__ == "__main__":

    parser = ArgumentParser(description='Script to extract AOI from SUMO net files. Takes the Covnex hull of the road network and dilates it, writes the output to aoi.geojson in the working directory')
    parser.add_argument('net_xml_file', help='SUMO net xml file')
    parser.add_argument('osm_file', help='OSM used to make net xml')
    args = parser.parse_args()

    extract_aoi(args.net_xml_file, args.osm_file)
