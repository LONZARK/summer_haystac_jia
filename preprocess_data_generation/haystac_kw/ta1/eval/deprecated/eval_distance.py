'''
This script will interrigate a database and create a distance per day metric output
'''
import psycopg2
import json
import numpy as np
import geopy.distance
import time
from rich import print
from plot_2_histograms import get_plot_and_js_divergence
import pickle
import pandas as pd
import pymap3d
import datetime

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
use_database = False
# central point in knoxville
_lat0 = 35.960443
_lon0 = -83.921263
save_plotting_data = True
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


def get_dates(connection, table):
    # grabs the dates from the table of stop points
    with connection.cursor() as cur:
        # get the dates so we can look at distance per day
        query = f"SELECT date_trunc('day',time_start) from {table};"
        cur.execute(query)
        res = cur.fetchall()
        dates = np.array(res)
        query = f"SELECT date_trunc('day',time_stop) from {table};"
        cur.execute(query)
        res = cur.fetchall()
        np.append(dates, res)

    dates = np.unique(dates).tolist()
    return dates


def get_agent_stops(connection, table, dates, agents):
    '''
    This function grabs the stop points per day from a table of all
    stop points per agent.  The returned is a dictionary by day and by agent
    '''
    agent_stops = {}
    for date in dates:
        agent_stops[str(date)] = {}
        for agent in agents:
            agent_stops[str(date)][str(agent[0])] = []
            # TODO: do the items after the AND need grouping "(" and ")"?
            query = '''SELECT ST_Astext(geometry) from [table]
            WHERE agent_id='[agent_id]' AND (
            date_trunc('day',time_start)='[date]'
            OR date_trunc('day',time_stop)='[date]' );
            '''
            query = query.replace("[table]", table)
            query = query.replace("[agent_id]", str(agent[0]))
            query = query.replace("[date]", str(date))
            # print(query)
            with connection.cursor() as cur:
                cur.execute(query)
                res = np.array(cur.fetchall())
            for itm in res:
                # print(itm[0])
                temp = itm[0].split("(")[-1]
                temp = temp.split(")")[0]
                lon, lat = temp.split(" ")
                agent_stops[str(date)][str(agent[0])].append((float(lat),
                                                              float(lon)))
    return agent_stops


def get_distances(agent_stops):
    '''
    Accumulates the distances from a dictionary into a distribution (list)
    '''
    distances = []
    for date in agent_stops:
        for agent in agent_stops[date]:
            # get all the stop points
            coords = agent_stops[date][agent]
            # get the points grouped by 2s (e.g. [[0,1], [1,2], [2,3]] )
            subList = [coords[n:n+2] for n in range(0, len(coords))]
            distance = 0
            for points in subList:
                if len(points) < 2:
                    continue  # ignore the last item since it won't have a pair
                # get the distance between the 2 stop points
                if use_database:
                    dist = geopy.distance.geodesic(points[0], points[1]).km
                else:
                    # euclidean since we are in EN coordinates
                    dist = ((points[0][0] - points[1][0])**2 +
                            (points[0][1] - points[1][1])**2)**0.5
                #print(f"distance between: {points[0]}->{points[1]} = {dist:0.3f}")
                distance += dist
            # add this distance per the day
            distances.append(distance)
    return distances


def get_enu_from_ll(lat, lon):
    '''
    This function takes a lat/lon and returns an East/North combo
    we are assuming height of zero
    '''
    en = pymap3d.enu.geodetic2enu(lat, lon, 0, _lat0, _lon0, 0)
    return en[:2]


def get_stop_points_csv(csv_file):
    table1 = pd.read_csv(csv_file, header=0)
    # get the dates
    #date_series = table1['time_start']
    #dt = time.time()
    #date_series = date_series.map(lambda x: parse(x))
    #dates = set(date_series.dt.date.tolist())
    #table1.insert(4, "date", date_series)
    #print(f"dates: {dates}")
    #print(f"map query time: {time.time() - dt}")
    # get the time it takes to pull stops via CSV
    csvq = time.time()
    stops_by_date_by_agent = {}
    for index, row in table1.iterrows():
        agent = row['agent_id']
        date = str(datetime.datetime.strptime(row['time_start'],
                                              date_format).date())
        # print(f"date = {date}")
        if date not in stops_by_date_by_agent.keys():
            stops_by_date_by_agent[date] = {}
        if agent not in stops_by_date_by_agent[date].keys():
            stops_by_date_by_agent[date][agent] = []
        coords = row['geometry']
        temp = coords.split("(")[-1]
        temp = temp.split(")")[0]
        lon, lat = temp.split(" ")
        # print(f"{lat}, {lon}")
        en = get_enu_from_ll(float(lat), float(lon))
        stops_by_date_by_agent[date][agent].append([x/1000 for x in en])
    #print(first_dist_stops)
    print(f"stops aggregation time: {time.time() - csvq}")
    return stops_by_date_by_agent


if __name__ == "__main__":
    start_time = time.time()

    distances1 = []
    distances2 = []
    if use_database:
        # Get a list of all of our agent ids
        conn = new_con()
        agent_ids1, agent_ids2 = get_agents(conn, "agent_ids", "agent_ids_2")
        print(f"agents 1: {len(agent_ids1)} ({agent_ids1[:2]})")
        print(f"agents 2: {len(agent_ids2)} ({agent_ids2[:2]})")

        # !!!!!!!!! reduce size for TESTING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #agent_ids1 = agent_ids1[:10]
        #agent_ids2 = agent_ids2[:10]

        # get the dates so we can look at distance per day
        dates1 = get_dates(conn, "stop_points_1")
        print(dates1)
        # collect all the agent stops from table 1
        agent_stops1 = get_agent_stops(conn, "stop_points_1", dates1,
                                       agent_ids1)
        # print(agent_stops)
        # calculate distances from the first table
        distances1 = get_distances(agent_stops1)
        #print(distances1)

        # get the dates so we can look at distance per day
        dates2 = get_dates(conn, "stop_points_2")
        print(dates2)
        # collect all the agent stops from table 2
        agent_stops2 = get_agent_stops(conn, "stop_points_2", dates2,
                                       agent_ids2)
        # print(agent_stops)
        # calculate distances from the seconds table
        distances2 = get_distances(agent_stops2)
        #print(distances2)
    else:
        settings = {}
        with open("common_files.json", 'r') as sf:
            settings = json.load(sf)
        csv1 = settings['test_subject_stops']
        csv2 = settings['reference_stops']
        # get an ordered dictionary of stops per day per agent
        # the coordinates will be in km (ENU without the U-height)
        agent_stops1 = get_stop_points_csv(csv1)
        #print(agent_stops1)
        dates = list(agent_stops1.keys())
        print(f"dates = {dates}")
        agent = list(agent_stops1[dates[0]].keys())[0]
        print(f"sample data from agent {agent}= {agent_stops1[dates[0]][agent]}")
        # calc the distances
        distances1 = get_distances(agent_stops1)
        #print(distances1)
        agent_stops2 = get_stop_points_csv(csv2)
        dates = list(agent_stops2.keys())
        print(f"dates = {dates}")
        agent = list(agent_stops2[dates[0]].keys())[0]
        print(f"sample data from agent {agent}= {agent_stops2[dates[0]][agent]}")
        # calc the distances
        distances2 = get_distances(agent_stops2)
        #print(distances1)

    # get the max value across both sets
    maxv = max([max(distances1), max(distances2)])
    N = 30
    bins = np.linspace(0, maxv, N)
    print(f"maxv = {maxv}")
    minx = min([min(distances1), min(distances2)])

    # calculate the histogram densities from the distributions
    density1, bins1 = np.histogram(distances1, bins, density=True)
    density2, bins2 = np.histogram(distances2, bins, density=True)
    ymax = max([max(density1), max(density2)])
    ymin = 0
    ymax = ymax*1.15
    print(f"ymax = {ymax}")

    # save this info for further use
    if save_plotting_data:
        items = {}
        items["density1"] = density1
        items["density2"] = density2
        items["bins"] = bins
        items["minx"] = minx
        items["maxv"] = maxv
        with open('distance_pd_data.pickle', 'wb') as pf:
            pickle.dump(items, pf)

    # get the Jensen-Shannon distance
    jsd_type1 = get_plot_and_js_divergence(density1, density2, bins,
                                           minx, maxv,
                                           metric_name="Distance per day (km)",
                                           show_plot=False,
                                           save_filename='metric_distance.png',
                                           name_reference="Knoxville Train",
                                           name_subject="Knoxville Test",
                                           plot_log10_x=True,
                                           plot_log10_y=True)
    print(f"Jensen-Shannon Type 1: {jsd_type1:0.8f}")

    if use_database:
        conn.close()
    print(f"time: {time.time() - start_time}")
