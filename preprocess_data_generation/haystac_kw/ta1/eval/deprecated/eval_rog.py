'''
This script will interrogate a database and create a Radius of Gyration (ROG) metric output
'''
import psycopg2
import json
import numpy as np
import time
from rich import print
import math
import pymap3d
import pandas as pd
from plot_2_histograms import get_plot_and_js_divergence
from multiprocessing import Pool, Manager
import pickle
from tqdm import tqdm

# import common settings from json
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
# use the database or use the csv files?
use_database = False
# central point in knoxville
_lat0 = 35.960443
_lon0 = -83.921263
save_plotting_info = True


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


def get_agents(connection, table1, table2):
    '''
    We are expecting 2 tables with a list of
    the agent ids (one for the test subject and one for the 
    type 1 - comparison data) - see setup_fdw_tables.py
    '''
    agent_ids = []
    agent_ids2 = []
    with connection.cursor() as cur:
        cur.execute(f"SELECT agent_id FROM {table1};")
        res = cur.fetchall()
        agent_ids = np.array(res)

        cur.execute(f"SELECT agent_id FROM {table2};")
        res = cur.fetchall()
        agent_ids2 = np.array(res)
    return (agent_ids, agent_ids2)


def get_enu_from_ll(lat, lon):
    '''
    This function takes a lat/lon and returns an East/North combo
    we are assuming height of zero
    '''
    en = pymap3d.enu.geodetic2enu(lat, lon, 0, _lat0, _lon0, 0)
    return en[:2]


def get_agent_stops(connection, table, agents):
    '''
    This function grabs the stop points per day from a table of all
    stop points per agent.  The returned is a dictionary by day and by agent
    '''
    agent_stops = {}
    for agent in agents:
        agent_stops[str(agent[0])] = {}
        agent_stops[str(agent[0])]["stop_points"] = []
        query = '''SELECT ST_X(geometry) from [table]
        WHERE agent_id='[agent_id]'
        '''
        query = query.replace("[table]", table)
        query = query.replace("[agent_id]", str(agent[0]))

        # print(query)
        with connection.cursor() as cur:
            cur.execute(query)
            resX = np.array(cur.fetchall())
            query = query.replace("ST_X", "ST_Y")  # change to the Y coord
            cur.execute(query)
            resY = np.array(cur.fetchall())
        for x in range(0, len(resX)):
            en = get_enu_from_ll(resY[x][0], resX[x][0])
            agent_stops[str(agent[0])]["stop_points"].append([x/1000 for x in en])
    return agent_stops


def mp_get_agent_stops(args):
    '''
    This function grabs the stop points per day from a table of all
    stop points per agent.  The returned is a dictionary by day and by agent
    '''
    agent_stops, table, agents = args
    connection = new_con()
    for agent in agents:
        agent_stops[str(agent[0])] = {}
        agent_stops[str(agent[0])]["stop_points"] = []
        query = '''SELECT ST_X(geometry) from [table]
        WHERE agent_id='[agent_id]'
        '''
        query = query.replace("[table]", table)
        query = query.replace("[agent_id]", str(agent[0]))

        # print(query)
        with connection.cursor() as cur:
            cur.execute(query)
            resX = np.array(cur.fetchall())
            query = query.replace("ST_X", "ST_Y")  # change to the Y coord
            cur.execute(query)
            resY = np.array(cur.fetchall())
        for x in range(0, len(resX)):
            en = get_enu_from_ll(resY[x][0], resX[x][0])
            agent_stops[str(agent[0])]["stop_points"].append([x/1000 for x in en])
    return agent_stops


def center_of_gravity(stop_points):
    stop_points = np.array(stop_points)
    return np.mean(stop_points, axis=0)


def radius_of_gyration(stop_points, cog):
    # subtract lat lon from the center values
    lats = [x[0] - cog[0] for x in stop_points]
    lons = [x[1] - cog[1] for x in stop_points]
    # take the larger of the 2 (TODO: this wasn't in the document)
    dists = []
    for x in range(0, len(lats)):
        if lats[x] > lons[x]:
            dists.append(lats[x])
        else:
            dists.append(lons[x])
    sqd_d = [x ** 2 for x in dists]
    rad1 = sum(sqd_d)/len(sqd_d)
    return math.sqrt(rad1)


def type2_func(rg, r0, beta, K):
    # per the metric document, the
    # equation is a truncated power law:
    # P(rg) = (rg + r0) ^ (−beta * exp(-r_g / K))
    return (rg + r0) ** (-beta) * np.exp(-rg / K)


def get_stop_points_csv(csv_file):
    # get the time it takes to pull stops via CSV
    csvq = time.time()
    table1 = pd.read_csv(csv_file, header=0)
    dist_of_stops = {}
    for index, row in table1.iterrows():
        agent = row['agent_id']
        if agent not in dist_of_stops.keys():
            dist_of_stops[agent] = {}
            dist_of_stops[agent]['stop_points'] = []
        coords = row['geometry']
        temp = coords.split("(")[-1]
        temp = temp.split(")")[0]
        lon, lat = temp.split(" ")
        # print(f"{lat}, {lon}")
        en = get_enu_from_ll(float(lat), float(lon))
        dist_of_stops[agent]['stop_points'].append([x/1000 for x in en])
    #print(first_dist_stops)
    print(f"csv query time: {time.time() - csvq}")
    return dist_of_stops


if __name__ == "__main__":
    start_time = time.time()

    # collect all the agent stops from table 1
    agent_ids1 = []
    agent_ids2 = []
    settings = {}
    with open("common_files.json", 'r') as sf:
        settings = json.load(sf)
    csv1 = settings['test_subject_stops']
    csv2 = settings['reference_stops']
    if use_database:
        # Get a list of all of our agent ids
        conn = new_con()
        agent_ids1, agent_ids2 = get_agents(conn, "agent_ids", "agent_ids_2")

        # !!!!!!!!! reduce size for TESTING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #agent_ids1 = agent_ids1[:2]
        #agent_ids2 = agent_ids2[:2]

        st1 = time.time()
        with Manager() as manager:
            first_dist_stops = manager.dict()
            args = [(first_dist_stops, "stop_points_1", x) for x in agent_ids1]
            with Pool(10) as p:
                r = list(tqdm(p.imap(mp_get_agent_stops, iter(args)),
                              total=len(agent_ids1)))
        first_dist_stops = get_agent_stops(conn, "stop_points_1", agent_ids1)
        print(f"db query time: {time.time() - st1}")
    else:
        first_dist_stops = get_stop_points_csv(csv1)
        agent_ids1 = list(first_dist_stops.keys())
    # print(first_dist_stops)
    print(f"agents 1: {len(agent_ids1)} ({agent_ids1[:2]})")

    rog1 = []
    tic = time.time()
    for ag in agent_ids1:
        agent = ag
        if use_database:
            agent = str(ag[0])
        if agent not in first_dist_stops.keys():
            print(f"agent has no stops: {agent}")
            continue
        # print(f"agent = {agent}")
        # get the center of all stops
        if len(first_dist_stops[agent]["stop_points"]) < 1:
            print(f"agent {agent} has no stops")
            continue
        # calculate ROG1
        xy = first_dist_stops[agent]["stop_points"]
        rog1.append(math.sqrt(sum(np.std(xy, axis=0)**2)))
    print(f"np.std time: {time.time() - tic}")

    '''
    # old method for calculating the rog
    tic = time.time()
    for ag in agent_ids1:
        agent = str(ag[0])
        cog = center_of_gravity(first_dist_stops[agent]["stop_points"])
        first_dist_stops[agent]["cog"] = cog
        # print(f"cog = {cog}")
        rog1.append(radius_of_gyration(first_dist_stops[agent]["stop_points"],
                                       first_dist_stops[agent]["cog"]))
    print(f"old method time: {time.time() - tic}")
    # print(f" rog1 = {rog1}")
    '''

    # collect all the agent stops from table 2
    if use_database:
        second_dist_stops = get_agent_stops(conn, "stop_points_2", agent_ids2)
    else:
        agent_ids2 = agent_ids1  # the database has renamed ids, but csv does not
        second_dist_stops = get_stop_points_csv(csv2)
    # print(second_dist_stops)
    print(f"agents 2: {len(agent_ids2)} ({agent_ids2[:2]})")

    rog2 = []
    for ag in agent_ids2:
        agent = ag
        if use_database:
            agent = str(ag[0])
        if agent not in second_dist_stops.keys():
            print(f"agent has no stops: {agent}")
            continue
        if len(second_dist_stops[agent]["stop_points"]) < 1:
            continue  # skip any that don't have stops
        # print(f"agent = {agent}")

        # calculate the ROG 2
        xy = second_dist_stops[agent]["stop_points"]
        rog2.append(math.sqrt(sum(np.std(xy, axis=0)**2)))
    # print(f" rog2 = {rog2}")

    # get the max value across both sets
    maxv = max([max(rog1), max(rog2)])
    N = 30
    bins = np.linspace(0, maxv, N)
    print(f"maxv = {maxv}")
    minx = min([min(rog1), min(rog2)])

    density1, bins1 = np.histogram(rog1, bins, density=True)
    density2, bins2 = np.histogram(rog2, bins, density=True)
    ymax = max([max(density1), max(density2)])
    ymin = 0
    ymax = ymax*1.15
    print(f"ymax = {ymax}")

    # save this info for further use
    if save_plotting_info:
        items = {}
        items["density1"] = density1
        items["density2"] = density2
        items["bins"] = bins
        items["minx"] = minx
        items["maxv"] = maxv
        with open('rog_data.pickle', 'wb') as pf:
            pickle.dump(items, pf)

    jsd_type1 = get_plot_and_js_divergence(density1, density2, bins,
                                           minx, maxv,
                                           metric_name="Radius of Gyr. (km)",
                                           show_plot=False,
                                           save_filename='metric_rog.png',
                                           plot_log10_x=True,
                                           plot_log10_y=True)
    print(f"Jensen-Shannon Type 1: {jsd_type1:0.8f}")

    if use_database:
        conn.close()
    print(f"time: {time.time() - start_time}")
