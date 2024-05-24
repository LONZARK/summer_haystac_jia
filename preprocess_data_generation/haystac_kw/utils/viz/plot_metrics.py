import matplotlib.pyplot as plt
from scipy.spatial import distance
import numpy as np
import seaborn as sns
import os


def set_matplotlib_params():
    params = {'legend.fontsize': 25,
              'figure.figsize': (32, 14),
              'axes.labelsize': 45,
              'axes.titlesize': 45,
              'axes.linewidth': 3,
              'xtick.labelsize': 30,
              'ytick.labelsize': 30,
              'font.size': 30}

    plt.rcParams.update(params)
    plt.grid(alpha=0.3)


def plot_seaborn_js_divergence(density1, density2,
                               hist_data1, hist_data2,
                               bins,
                               minx, maxx,
                               metric_name="Radius of Gyr. (km)",
                               show_plot=False,
                               save_filename='metric_rog.png',
                               name_reference="Reference",
                               name_subject="Test Subject",
                               plot_log10_x=False,
                               plot_log10_y=False):

    set_matplotlib_params()

    # ymax = max([max(density1), max(density2)]) * 1.5
    # ymin = 0

    jsd_type1 = distance.jensenshannon(density1, density2) ** 2

    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(1, 2)
        sns.histplot(hist_data1, bins=bins, kde=True, ax=axs[0])
        axs[0].title.set_text(name_subject)
        axs[0].set_xlabel(metric_name)
        axs[0].set_ylabel('Count')
        axs[0].set_xlim([minx, maxx])
        # axs[0].set_ylim([ymin, ymax])

        sns.histplot(hist_data2, bins=bins, kde=True, ax=axs[1])
        axs[1].title.set_text(name_reference)
        axs[1].set_xlabel(metric_name)
        axs[1].set_ylabel('Count')
        axs[1].set_xlim([minx, maxx])
        # axs[1].set_ylim([ymin, ymax])

        axs[0].text(0.99, 0.05, f"Bins = {bins}",
                    verticalalignment='top',
                    horizontalalignment='right',
                    transform=axs[0].transAxes,
                    color='black', fontsize=35)
        axs[1].text(0.99, 0.05, f"Bins = {bins}",
                    verticalalignment='top',
                    horizontalalignment='right',
                    transform=axs[1].transAxes,
                    color='black', fontsize=35)
        # put the divergence in the top
        axs[1].text(0.98, 0.98, f"JS Divergence = {jsd_type1:0.8f}",
                    verticalalignment='top',
                    horizontalalignment='right',
                    transform=axs[1].transAxes,
                    color='black', fontsize=45)

        plt.tight_layout()
        plt.savefig("%s.pdf" % (save_filename))
        plt.close()

    return jsd_type1


def get_plot_and_js_divergence(density1, density2, bins,
                               minx=None, maxx=None,
                               metric_name="Radius of Gyr. (km)",
                               show_plot=False,
                               save_folder='',
                               save_filename='metric_rog.png',
                               name_reference="Reference",
                               name_subject="Test Subject",
                               plot_log10_x=False,
                               plot_log10_y=False,
                               y_axis_label="Density", **kwargs
                               ):
    ymax = max([max(density1), max(density2)])
    ymin = 0
    ymax = ymax * 1.15

    # get the Jensen-Shannon distance
    jsd_type1 = distance.jensenshannon(density1, density2) ** 2

    # plot the distributions
    fig = plt.figure(figsize=(32, 14), dpi=80)
    # kwargs = dict(alpha=1.0, bins=100, density=True, stacked=False)
    plt.rc('font', **{'size': 30})
    plt.rc('axes', linewidth=3)
    fig.tight_layout()
    plt.minorticks_on()

    ax = plt.subplot(1, 2, 2)
    if plot_log10_y:
        ax.set_yscale('log')
    if plot_log10_x:
        ax.set_xscale('log')
    plt.title(name_subject, fontsize=45)
    ax.set_xlabel(metric_name, fontsize=45)
    ax.set_ylabel(y_axis_label, fontsize=45)
    plt.bar((bins[1:] + bins[:-1]) / 2,
            height=density1,
            width=np.diff(bins),
            label=f'JS Divergence = {jsd_type1:0.8f}')
    ax.set_xlim([minx, maxx])
    ax.set_ylim([ymin, ymax])
    # plt.legend()
    ax.text(0.99, 0.05, f"Bins = {len(bins)}",
            verticalalignment='top',
            horizontalalignment='right',
            transform=ax.transAxes,
            color='black', fontsize=35)
    # put the divergence in the top
    ax.text(0.98, 0.98, f"JS Divergence = {jsd_type1:0.8f}",
            verticalalignment='top',
            horizontalalignment='right',
            transform=ax.transAxes,
            color='black', fontsize=45)

    ax = plt.subplot(1, 2, 1)
    if plot_log10_y:
        ax.set_yscale('log')
    if plot_log10_x:
        ax.set_xscale('log')
    plt.title(name_reference, fontsize=45)
    ax.set_xlabel(metric_name,
                  fontsize=45)
    ax.set_ylabel(y_axis_label, fontsize=45)
    plt.bar((bins[1:] + bins[:-1]) / 2,
            height=density2,
            width=np.diff(bins), label=f"Bins={len(bins)}")
    ax.set_xlim([minx, maxx])
    ax.set_ylim([ymin, ymax])

    # plt.legend()
    ax.text(0.99, 0.05, f"Bins = {len(bins)}",
            verticalalignment='top',
            horizontalalignment='right',
            transform=ax.transAxes,
            color='black', fontsize=35)

    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(os.path.join(save_folder, save_filename))
    if show_plot:
        print(f"ymax = {ymax}")
        print(f"Jensen-Shannon Type 1: {jsd_type1:0.8f}")
        plt.show()
    return jsd_type1
