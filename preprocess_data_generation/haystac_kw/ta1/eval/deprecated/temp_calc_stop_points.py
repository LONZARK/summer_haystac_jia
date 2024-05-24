import psycopg2
import json
from multiprocessing import Pool
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm
from rich import print
import sys
sys.path.insert(1, '..')
#from haystac_shared.utils.stop_points import calculate_stop_points
from old_stop_points import calculate_stop_points
import datetime

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


def get_unixtime(dt64):
    """
    https://stackoverflow.com/questions/11865458/how-to-get-unix-timestamp-from-numpy-datetime64
    """
    return dt64.astype('datetime64[s]').astype('int')


def get_reg_datetime(dt):
    newdt = datetime.datetime.fromtimestamp(dt)
    return newdt


def process_stops(args):
    engine = create_engine(db_connection_url)
    agent_id, table = args
    conn = new_con()
    with conn.cursor() as cur:
        query = f"SELECT timestamp FROM {agent_id}"
        # print(f"query = {query}")
        cur.execute(query)
        res = cur.fetchall()
        cur.execute(f"SELECT latitude from {agent_id};")
        lat = cur.fetchall()
        cur.execute(f"SELECT longitude from {agent_id};")
        lon = cur.fetchall()
    df = pd.DataFrame(res, columns=['timestamp'])
    #df['timestep'] = df['timestep'] + 1686155145
    #df = df.rename(columns={"timestep": "timestamp"})
    df['timestamp'] = get_unixtime(df['timestamp'].values)
    # add the lat lon values
    df['latitude'] = np.array(lat)
    df['longitude'] = np.array(lon)
    #print(df)

    #print(f"agnet_id = {agent_id}")
    skipped = False
    try:
        gdf = calculate_stop_points(agent_id, df)
        #print(f"gdf = {gdf}")
        # convert datetimes back to easily readable format for db
        gdf['time_start'] = [get_reg_datetime(x) for x in gdf['time_start'].values]
        gdf['time_stop'] = [get_reg_datetime(x) for x in gdf['time_stop'].values]
        #print(f"agent_id = {gdf['agent_id'][0]}")
        gdf.to_postgis(table, engine, if_exists='append', index=False)
    except ValueError as ex:
        print(f"ERROR: skipped agent_id : {agent_id}\n{ex}")
        skipped = True
    except IndexError as ex:
        print(f"Index ERROR: skipped agent_id : {agent_id}\n{ex}")
        skipped = True
    finally:
        if skipped:
            with open('skipped.txt', 'a') as sf:
                sf.write(f'{agent_id}\n')


if __name__ == "__main__":
    # Get a list of all of our agent ids
    conn = new_con()
    agent_ids1 = []
    agent_ids2 = []
    with conn.cursor() as cur:
        cur.execute("SELECT agent_id FROM agent_ids;")
        res = cur.fetchall()
        for i in range(len(res)):
            agent_ids1.append(res[i][0])

    print(f"agents 1: {len(agent_ids1)}")

    # run a single agent
    single = False

    # get rid of the old stop points tables and re-create
    if single is False:
        with conn.cursor() as cur:
            cur.execute('DROP TABLE IF EXISTS stop_points_1;')
            conn.commit()
            ct = ''' CREATE TABLE stop_points_1 (
                agent_id        text,
                time_start      timestamp without time zone,
                time_stop       timestamp without time zone,
                geometry        geometry(Point,4326)
            );
            '''
            cur.execute(ct)
            conn.commit()

    # test single agent:
    if single:
        args = "agent1_78230_12878230_1283_af739e7a060911eebcf8b518e64d283b", "stop_points_1"
        process_stops(args)
    else:
        args = [(x, 'stop_points_1') for x in agent_ids1]
        with Pool(4) as p:
            r = list(tqdm(p.imap(process_stops, iter(args)),
                          total=len(agent_ids1)))
