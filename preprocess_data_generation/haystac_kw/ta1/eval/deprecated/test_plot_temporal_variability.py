import pickle
import time
from plot_2_histograms import get_plot_and_js_divergence
import numpy as np

if __name__ == "__main__":
    start_time = time.time()
    rank=2
    nbins=101

    with open('scripts/temporal_variability_train.pkl', 'rb') as jf:
        data_train = pickle.load(jf)

    with open('scripts/temporal_variability_test.pkl', 'rb') as jf:
        data_test = pickle.load(jf)

    for i in range(rank):
        for typ in ['arr', 'dep']:
            
            
            data_1 = list(data_train[f'std-{typ}-{i}'])
            data_2 = list(data_test[f'std-{typ}-{i}'])
            minx = min(np.nanmin(data_1),np.nanmin(data_2))
            maxv = max(np.nanmax(data_1),np.nanmax(data_2))
            bins = np.linspace(minx,maxv,nbins)

            density1, _ = np.histogram(data_1, bins, density=True)
            density2, _ = np.histogram(data_2, bins, density=True)

            jsd_type1 = get_plot_and_js_divergence(density1, density2, bins,
                                                minx, maxv,
                                                metric_name=f"Temporal Variability std-{typ}-{i}(min)",
                                                show_plot=True,
                                                save_filename=f'temporal_variability std-{typ}-{i}.png',
                                                name_reference="Knoxville Train",
                                                name_subject="Knoxville Test",
                                                plot_log10_x=False,
                                                plot_log10_y=False)
            print(f"Jensen-Shannon Type 1: {jsd_type1:0.8f}")

            print(f"time: {time.time() - start_time}")
