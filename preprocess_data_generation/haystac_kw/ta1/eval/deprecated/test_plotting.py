import pickle
import time
from plot_2_histograms import get_plot_and_js_divergence


if __name__ == "__main__":
    start_time = time.time()

    with open('distance_pd_data.pickle', 'rb') as jf:
        data = pickle.load(jf)

    density1 = data['density1']
    density2 = data['density2']
    bins = data['bins']
    minx = data['minx']
    maxv = data['maxv']

    jsd_type1 = get_plot_and_js_divergence(density1, density2, bins,
                                           minx, maxv,
                                           metric_name="Distance per day (km)",
                                           show_plot=True,
                                           save_filename='metric_distance.png',
                                           name_reference="Knoxville Train",
                                           name_subject="Knoxville Test",
                                           plot_log10_x=True,
                                           plot_log10_y=True)
    print(f"Jensen-Shannon Type 1: {jsd_type1:0.8f}")

    print(f"time: {time.time() - start_time}")
