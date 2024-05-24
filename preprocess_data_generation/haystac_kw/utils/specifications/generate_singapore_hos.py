import sys
import os, glob 
import json 

os.environ['USE_PYGEOS'] = '0'

import uuid
import pandas as pd 
import geopandas as gpd 
from shapely import GeometryCollection

from haystac_kw.utils.data_utils.validate_json_schema import validate_json

POI_PICKLE = os.path.join("/home/local/KHQ/laura.zheng/citydata/hos_app_files", "poi.pkl")
HOS_SCHEMA = os.path.join("/home/local/KHQ/laura.zheng/haystac", "schemas", "hos.schema.json")

def sample_random_geojson(poi_gdf : gpd.GeoDataFrame):
    """
        Sample a random polygon from the GeoJSON dataframe, and return the GeometryCollection JSON needed for the HAS Schema.

        :param poi_gdf: places-of-interest geodataframe containing all the the polygons to sample from
        :type poi_gdf: geopandas.Dataframe

        :returns: dict representing shapely.GeometryCollection/polygon information of a sampled location
        :rtype: dict
    """

    sampled_location = poi_gdf.sample(n=1)
    location_polygon = sampled_location["building_poly"]

    geojson_collection = GeometryCollection([location_polygon])
    geojson_dict = json.loads(gpd.GeoSeries(geojson_collection).set_crs('EPSG:4326').to_json())
    geojson_dict = geojson_dict['features'][0]['geometry']

    return geojson_dict

def generate_all_hos_in_dir(hos_dirpath, save_dirpath):

    hos_list = glob.glob(os.path.join(hos_dirpath, "*.json"))
    os.makedirs(save_dirpath, exist_ok=True)

    for hos in hos_list:
        
        valid, msg = validate_json(hos, HOS_SCHEMA)
        assert valid, "HOS does not follow HOS schema: %s" % msg

        generate_single_hos(hos, save_dirpath)

    print("Generated %d HOS files to path %s" % (len(hos_list), save_dirpath))

def generate_single_hos(original_hos_path, new_hos_dirpath):

    poi_df = pd.read_pickle(POI_PICKLE)
    poi_df = gpd.GeoDataFrame(poi_df, 
                            geometry="building_centroid", 
                            crs='EPSG:4326')
    
    utm = poi_df.estimate_utm_crs() 

    poi_df = gpd.GeoDataFrame(poi_df, 
                            geometry="building_poly", 
                            crs=utm).to_crs('EPSG:4326')

    with open(original_hos_path, 'r') as file:
        original_hos_data = json.load(file)

    new_hos_data = original_hos_data.copy() 
    # new_hos_obj_uid = str(uuid.uuid4())
    new_hos_obj_uid = os.path.basename(original_hos_path).replace("hos_", "").replace(".json", "")
    new_hos_data["objective_uid"] = new_hos_obj_uid

    # uid_conversions = {}
    location_to_singapore_location = {}

    new_events_data = [] 

    for event in original_hos_data["events"]:

        new_event = event.copy() 

        old_location_str = str(new_event["location"])

        if old_location_str in location_to_singapore_location:
            new_location = location_to_singapore_location[old_location_str]
        else:
            new_location = sample_random_geojson(poi_df)
            location_to_singapore_location[old_location_str] = new_location

        # replace the geojson with a singpore building 
        new_event["location"] = new_location 

        new_events_data.append(new_event) 
    
    new_hos_data["events"] = new_events_data

    new_hos_path = os.path.join(new_hos_dirpath, "hos_%s.json" % new_hos_obj_uid)
    
    assert new_hos_path.endswith(('.json')), "savepath must be appended with *.json extension"

    with open(new_hos_path, 'w') as fp:
        json.dump(new_hos_data, fp, indent=2)

    valid, msg = validate_json(new_hos_path, HOS_SCHEMA)
    assert valid, "HOS does not follow HOS schema: %s" % msg

if __name__ == "__main__":

    baseline_hos_path = "/home/local/KHQ/laura.zheng/haystac/behaviorsim/baseline_hos"
    savedir = "/home/local/KHQ/laura.zheng/haystac/behaviorsim/baseline_hos_singapore"

    generate_all_hos_in_dir(baseline_hos_path, savedir)

