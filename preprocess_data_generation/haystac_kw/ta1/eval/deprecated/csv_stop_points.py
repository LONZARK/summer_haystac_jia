import json
import geopandas
import psycopg2
import pandas as pd
import datetime
from sqlalchemy import create_engine
from rich import print

# import common setting from json
settings_config = "db_connection_settings.json"
settings = {}
with open(settings_config, 'r') as sf:
    settings = json.load(sf)
_host = settings["host"]
_dbname = settings["database"]
_user = settings["user"]
_psw = settings["psw"]
_port = settings["port"]
db_connection_url = "postgresql://maplibre:maplibre@localhost:5432/trajectories"
csv1 = "/home/local/KHQ/joseph.vanpelt/projects/haystac/data/test.csv"
csv2 = "/home/local/KHQ/joseph.vanpelt/projects/haystac/data/train.csv"
prefix1 = None
prefix2 = "agent2"
table_name1 = "stop_points_1"
table_name2 = "stop_points_2"
date_format = "%Y-%m-%d %H:%M:%S"


def new_con(host=_host, dbname=_dbname, user=_user, psw=_psw, port=_port):
    #print(f"connecting to {dbname} on {host}:{port}")
    conn = psycopg2.connect(
        host=host,
        database=dbname,
        user=user,
        password=psw,
        port=port
        )
    return conn


def change_date_format(string):
    return datetime.datetime.strptime(string, date_format)


if __name__ == "__main__":
    # create connection for the geopandas insertion
    engine = create_engine(db_connection_url)

    # drop existing tables
    conn = new_con()
    with conn.cursor() as cur:
        cur.execute(f'DROP TABLE IF EXISTS {table_name1};')
        conn.commit()
        cur.execute(f'DROP TABLE IF EXISTS {table_name2};')
        conn.commit()

    # get the csv read in
    table1 = pd.read_csv(csv1, header=0)
    # add an optional prefix to the front
    if prefix1 is not None:
        table1["agent_id"] = table1["agent_id"].str.replace("agent_",
                                                            f"{prefix1}_")
    # print(table1)
    # convert the times to datetime format so the db will get them correctly
    table1["time_start"] = table1["time_start"].apply(change_date_format)
    table1["time_stop"] = table1["time_stop"].apply(change_date_format)
    gdf = geopandas.GeoDataFrame(
        table1,
        geometry=geopandas.GeoSeries.from_wkt(table1["geometry"]),
        crs='EPSG:4326')
    gdf = gdf.to_crs("EPSG:4326").dropna()[
        ['agent_id', 'time_start', 'time_stop', 'geometry']]

    print(gdf)
    gdf.to_postgis(table_name1, engine, if_exists='append', index=False)

    table2 = pd.read_csv(csv2, header=0)
    if prefix2 is not None:
        table2["agent_id"] = table2["agent_id"].str.replace("agent_",
                                                            f"{prefix2}_")
    # print(table2)
    table2["time_start"] = table2["time_start"].apply(change_date_format)
    table2["time_stop"] = table2["time_stop"].apply(change_date_format)
    gdf = geopandas.GeoDataFrame(
        table2,
        geometry=geopandas.GeoSeries.from_wkt(table2['geometry']),
        crs='EPSG:4326')
    gdf = gdf.to_crs("EPSG:4326").dropna()[
        ['agent_id', 'time_start', 'time_stop', 'geometry']]
    print(gdf)
    gdf.to_postgis(table_name2, engine, if_exists='append', index=False)
