'''
This script will interrogate a database and create a Number of locations visited metric output
'''
import json
import time
import pickle
import numpy as np
from rich import print
import pandas as pd
import datetime
from plot_2_histograms import get_plot_and_js_divergence

settings_format = "%Y-%m-%d"
csvdate_format = "%Y-%m-%d %H:%M:%S"
save_plotting_data = True


def add_uid(dist, uid, date, agent):
    if date not in dist.keys():
        dist[date] = {}
    if agent not in dist[date].keys():
        dist[date][agent] = []

    if uid not in dist[date][agent]:
        # add the uid
        dist[date][agent].append(uid)


def get_unique_stops(csv_file, rdates, date_split):
    csvq = time.time()
    table1 = pd.read_csv(csv_file, header=0)

    unique_dist1 = {}
    unique_dist_ref = {}
    for index, row in table1.iterrows():
        agent = row['agent_id']
        uid = row['unique_stop_point']
        full_date = datetime.datetime.strptime(row['time_start'],
                                               csvdate_format)
        date = str(full_date.date())
        # print(f"date = {date}")

        # get the dates that belong to the reference set
        # based on the split time and days - group them
        # separate from the test subject data
        if date in rdates and full_date < date_split:
            # add these to reference distribution
            add_uid(unique_dist_ref, uid, date, agent)
            #if date == "2023-01-16":
            #    print(f"date = {full_date}")
        else:
            add_uid(unique_dist1, uid, date, agent)

    print(f"stops aggregation time: {time.time() - csvq}")
    return (unique_dist1, unique_dist_ref)


if __name__ == "__main__":
    start_time = time.time()
    with open("common_files.json", 'r') as sf:
        settings = json.load(sf)
    csv1 = settings['unique_stops']
    reference_dates = settings['reference_unique_stop_dates']
    date_split = datetime.datetime.strptime(settings['date_split'],
                                            csvdate_format)

    # get the sets split by dates
    unique_dist1, unique_dist_ref = get_unique_stops(csv1,
                                                     reference_dates,
                                                     date_split)
    #print(f"unique_dist1 = {unique_dist1}")
    print(f"dates1 = {list(unique_dist1.keys())}")
    print(f"dates2 = {list(unique_dist_ref)}")

    # now pull the numbers from the dictionary
    distribution1 = []
    for date in unique_dist1:
        for agent in unique_dist1[date]:
            # add up the total number of unique items
            distribution1.append(len(unique_dist1[date][agent]))
    distribution1 = sorted(distribution1, reverse=True)
    #print(f"distribution1 = {distribution1}")

    distribution2 = []
    for date in unique_dist_ref:
        for agent in unique_dist_ref[date]:
            # add up the total number of unique items
            distribution2.append(len(unique_dist_ref[date][agent]))
    distribution2 = sorted(distribution2, reverse=True)

    # get the max value across both sets
    maxv = max([max(distribution1), max(distribution2)])
    N = 30
    bins = np.linspace(0, maxv, N)
    print(f"maxv = {maxv}")
    minx = min([min(distribution1), min(distribution2)])

    density1, bins1 = np.histogram(distribution1, bins, density=True)
    density2, bins2 = np.histogram(distribution2, bins, density=True)
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
        with open('locations_visited_data.pickle', 'wb') as pf:
            pickle.dump(items, pf)

    jsd_type1 = get_plot_and_js_divergence(density1, density2, bins,
                                           minx, maxv,
                                           metric_name="Locations Visited per day",
                                           show_plot=False,
                                           save_filename='metric_locations_visited.png',
                                           plot_log10_x=False,
                                           plot_log10_y=False)
    print(f"Jensen-Shannon Type 1: {jsd_type1:0.8f}")

    print(f"time: {time.time() - start_time}")
