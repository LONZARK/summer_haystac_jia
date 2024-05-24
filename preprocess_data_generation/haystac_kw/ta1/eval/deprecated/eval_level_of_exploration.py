'''
This script will interrogate a database and create a
Level of Exploration visited metric output
'''
import json
import time
import pickle
import numpy as np
from rich import print
import pandas as pd
import datetime
from scipy.spatial import distance
from plot_2_histograms import get_plot_and_js_divergence

csvdate_format = "%Y-%m-%d %H:%M:%S"
save_plotting_data = True


def add_uid(dist, uid, agent):
    if agent not in dist.keys():
        dist[agent] = {}
        dist[agent]["total_stops"] = 0

    if uid not in dist[agent]:
        # add the uid
        dist[agent][uid] = 0

    # count a visit to this uid
    dist[agent][uid] += 1
    # get the total stops per agent
    dist[agent]["total_stops"] += 1


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
            add_uid(unique_dist_ref, uid, agent)
            #if date == "2023-01-16":
            #    print(f"date = {full_date}")
        else:
            add_uid(unique_dist1, uid, agent)

    print(f"stops aggregation time: {time.time() - csvq}")
    return (unique_dist1, unique_dist_ref)


def get_topk_list_per_agent(dist, agent):
    total = float(dist[agent]["total_stops"])
    topk = [float(dist[agent][u])/total for u in dist[agent] if u != "total_stops"]
    return sorted(topk, reverse=True)


def get_js_score(distribution1, distribution2):
    # get the max value across both sets
    maxv = max([max(distribution1), max(distribution2)])
    N = 30
    bins = np.linspace(0, maxv, N)
    # print(f"maxv = {maxv}")

    density1, bins1 = np.histogram(distribution1, bins, density=True)
    density2, bins2 = np.histogram(distribution2, bins, density=True)
    # get the Jensen-Shannon distance
    jsd_type1 = distance.jensenshannon(density1, density2) ** 2
    # print(f"Jensen-Shannon Type 1: {jsd_type1:0.8f}")
    return jsd_type1


if __name__ == "__main__":
    start_time = time.time()
    with open("common_files.json", 'r') as sf:
        settings = json.load(sf)
    csv1 = settings['unique_stops']
    reference_dates = settings['reference_unique_stop_dates']
    date_split = datetime.datetime.strptime(settings['date_split'],
                                            csvdate_format)

    # get the sets split by dates
    # these dictionaries will now have the data arranged as:
    # dict[agent][uid] = (count of visits per simulation)
    # and dict[agent]["total_stops"] = (total stops for simulation)
    unique_dist1, unique_dist_ref = get_unique_stops(csv1,
                                                     reference_dates,
                                                     date_split)
    # print(f"unique_dist1 = {unique_dist1}")

    # pick an agent from the reference set to be the comparison
    # TODO: how does this get decided?  For now we are getting the first one
    agent = list(unique_dist_ref.keys())[0]
    # store this distribution for plotting
    distribution1 = get_topk_list_per_agent(unique_dist_ref, agent)
    print(f"distribution1 [agent: {agent}] = {distribution1}")

    # hold the total scores for the test subject dataset (to be averaged)
    js_scores = []
    # save the first distribution for plotting
    distribution2 = []
    d2_js_score = 0  # and save the individual score for plotting
    # now got through all the test subject scores
    for agent in unique_dist1:
        dist2 = get_topk_list_per_agent(unique_dist1, agent)
        js_score = get_js_score(distribution1, dist2)
        js_scores.append(js_score)
        # save this for plotting
        if len(distribution2) < 1:
            distribution2 = dist2
            d2_js_score = js_score

    avg_js_score = sum(js_scores)/len(js_scores)
    # TODO: add this score to the plot
    print(f"average JS score: {avg_js_score}")

    # TODO: should this plot change to match the metrics doc?

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
        with open('level_of_exploration_data.pickle', 'wb') as pf:
            pickle.dump(items, pf)

    jsd_type1 = get_plot_and_js_divergence(density1, density2, bins,
                                           minx, maxv,
                                           metric_name="Rank(k) locations visited",
                                           show_plot=False,
                                           save_filename='metric_level_of_exploration.png',
                                           plot_log10_x=False,
                                           plot_log10_y=False,
                                           y_axis_label="Visitation Frequency")
    print(f"Jensen-Shannon Type 1: {jsd_type1:0.8f}")

    print(f"time: {time.time() - start_time}")
