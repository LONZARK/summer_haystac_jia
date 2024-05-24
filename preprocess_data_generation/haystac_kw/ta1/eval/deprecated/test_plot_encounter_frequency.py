import pickle
import time
from plot_2_histograms import get_plot_and_js_divergence
import numpy as np

if __name__ == "__main__":
    start_time = time.time()
    rank=2
    nbins=51

    with open('enounter_frequency_train.pkl', 'rb') as jf:
        data_train = pickle.load(jf)

    with open('enounter_frequency_test.pkl', 'rb') as jf:
        data_test = pickle.load(jf)

    data_1 = data_train['count'].values
    data_2 = data_test['count'].values
    minx = min(np.nanmin(data_1),np.nanmin(data_2))
    maxv = max(np.nanmax(data_1),np.nanmax(data_2))
    bins = np.linspace(minx,maxv,nbins)

    density1, _ = np.histogram(data_1, bins, density=False)
    density2, _ = np.histogram(data_2, bins, density=False)

    jsd_type1 = get_plot_and_js_divergence(density1, density2, bins,
                                        minx, maxv,
                                        metric_name=f"Encounter Frequency (encounters)",
                                        show_plot=True,
                                        save_filename=f'encounter_frequency.png',
                                        name_reference="Knoxville Train",
                                        name_subject="Knoxville Test",
                                        plot_log10_x=True,
                                        plot_log10_y=True,
                                        ylabel='Count')
    print(f"Jensen-Shannon Type 1: {jsd_type1:0.8f}")

    print(f"time: {time.time() - start_time}")
