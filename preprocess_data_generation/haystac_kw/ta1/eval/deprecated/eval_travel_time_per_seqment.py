'''
This script will interrigate a database and create a distance per day metric output
'''
import json
import numpy as np
import time
from rich import print
from plot_2_histograms import get_plot_and_js_divergence
import pickle
import pandas as pd
import datetime

# central point in knoxville
_lat0 = 35.960443
_lon0 = -83.921263
save_plotting_data = True
date_format = "%Y-%m-%d %H:%M:%S"


def get_stop_points_csv(csv_file):
    table1 = pd.read_csv(csv_file, header=0)
    # get the time it takes to pull stops via CSV
    csvq = time.time()
    stops_per_segment = []
    for index, row in table1.iterrows():
        # get the start and stop times
        start = datetime.datetime.strptime(row['time_start'],
                                           date_format)
        stop = datetime.datetime.strptime(row['time_stop'],
                                          date_format)
        diff = stop - start
        stops_per_segment.append(diff.seconds/60)  # minutes
    print(f"stops aggregation time: {time.time() - csvq}")
    return stops_per_segment


if __name__ == "__main__":
    start_time = time.time()

    settings = {}
    with open("common_files.json", 'r') as sf:
        settings = json.load(sf)
    csv1 = settings['test_subject_stops']
    csv2 = settings['reference_stops']
    # get an ordered dictionary of stops per day per agent
    # the coordinates will be in km (ENU without the U-height)
    distribution1 = get_stop_points_csv(csv1)
    #print(distribution1)

    distribution2 = get_stop_points_csv(csv2)

    # get the max value across both sets
    maxv = max([max(distribution1), max(distribution2)])
    N = 30
    bins = np.linspace(0, maxv, N)
    print(f"maxv = {maxv}")
    minx = min([min(distribution1), min(distribution2)])

    # calculate the histogram densities from the distributions
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
        with open('travel_time_per_seg_data.pickle', 'wb') as pf:
            pickle.dump(items, pf)

    # get the Jensen-Shannon distance
    jsd_type1 = get_plot_and_js_divergence(density1, density2, bins,
                                           minx, maxv,
                                           metric_name="Travel time per segment (min.)",
                                           show_plot=False,
                                           save_filename='metric_travel_time_per_segment.png',
                                           name_reference="Knoxville Train",
                                           name_subject="Knoxville Test",
                                           plot_log10_x=True,
                                           plot_log10_y=True)
    print(f"Jensen-Shannon Type 1: {jsd_type1:0.8f}")

    print(f"time: {time.time() - start_time}")
