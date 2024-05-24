import glob
import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import warnings

from datetime import datetime

from .plot_2_histograms import get_plot_and_js_divergence, plot_seaborn_js_divergence

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

NUM_BINS = 30

EARTH_RADIUS_METERS = 6.3781370e+6

MAX_SPEED_THRESHOLD = 45. # m/s


class ParquetData(object):

    def __init__(self, pq_paths):

        self.data = pq_paths
        self.length = len(pq_paths)

    def __iter__(self):

        self.n = 0
        return self

    def __len__(self):

        return self.length

    def __next__(self):

        if self.n < self.length:

            data_path = self.data[self.n]
            data = pd.read_parquet(data_path)
            data = data.sort_values(by='timestamp', ascending=True)
            
            self.n += 1

            return data

        else:

            raise StopIteration


def haversine(theta):
    theta = np.radians(theta)
    return 0.5 * (1 - np.cos(theta))


class KinematicEvaluator(object):

    def __init__(self, data_path: str, max_agents : int = None, fig_path : str ="figs"):
        """
        Constructor for Evaluator object.

        arguments:
        - data_path: where the data is stored, the parent directory of parquet files
        - fig_path: optional, what the directory for figure output should be named. default: "figs"
        """

        assert os.path.exists(data_path), "data path provided does not exist"
        self.data_path = data_path
        self.data_pq = glob.glob(os.path.join(self.data_path, "*.parquet"))

        # truncate the amt of data if maximum is specified for speed
        if max_agents is not None and max_agents < len(self.data_pq): 
            warnings.warn("truncating kinematics evaluation to %d agents" % max_agents)
            self.data_pq = self.data_pq[:max_agents]
            
        self.data = ParquetData(self.data_pq)
        self.fig_folder = fig_path

        os.makedirs(fig_path, exist_ok=True)

        print("loaded data from %s" % (data_path))
        print("total agents: %d" % (len(self.data)))

    @staticmethod
    def plot_histogram(hist_data: np.array,
                       title: str,
                       x_label: str,
                       out_name: str):

        """
        Plots a histogram based on the provided histogram data.
        Uses seaborn for rendering.

        input:
        - hist_data: histogram data
        - title: title of plot
        - x_label: metric for x-axis labeling
        - out_name: where to save the figure to
        """

        with sns.axes_style("darkgrid"):
            hist_plot = sns.displot(hist_data, bins=NUM_BINS, kde=True)
            plt.title(title)
            plt.xlabel(x_label)
            plt.tight_layout()
            plt.savefig("%s.pdf" % (out_name))

    @staticmethod
    def plot_2_histograms(hist_data1: np.array,
                          hist_data2: np.array,
                          x_label: str,
                          reference: str,
                          subject: str,
                          out_name: str, 
                          plot_log10_y: bool = False):
        """
        Plots two histograms side-by-side, with one being the reference and one being the subject.
        Uses the plot_2_histograms module.

        input:
        - hist_data1: histogram data for reference
        - hist_data2: histogram data for subject
        - x_label: metric for x-axis labeling
        - reference: name of reference distribution
        - subject: name of subject distribution
        - out_name: where to save the figure to

        output:
        - None
        - side effect: saves figure to specified out_name in parameters
        """

        minx = min(hist_data1.min(), hist_data2.min())
        maxx = max(hist_data1.max(), hist_data2.max())

        bins_arr = np.linspace(minx, maxx, NUM_BINS)
        density1, _ = np.histogram(hist_data1, bins_arr, density=True)
        density2, _ = np.histogram(hist_data2, bins_arr, density=True)

        plot_seaborn_js_divergence(density2, density1,
                                   hist_data1, hist_data2,
                                   NUM_BINS,
                                   minx, maxx,
                                   metric_name=x_label,
                                   save_filename=out_name,
                                   name_reference=reference,
                                   name_subject=subject,
                                   plot_log10_x=False,
                                   plot_log10_y=plot_log10_y)

    @staticmethod
    def compute_distance_m_from_latlon(trajectory_latlon: np.array) -> np.array:
        """
        Computes the distance in meters between points on a trajectory using the Haversine formula.

        input:
        - trajectory_latlon: array of N trajectory points in lat and lon format, with shape [N, 2]

        output:
        - distances: array of N-1 distances between the N trajectory points, in meters
        """

        traj = trajectory_latlon
        traj_roll = np.roll(traj, -1, axis=0)

        traj = traj[:-1]
        traj_roll = traj_roll[:-1]

        traj = traj.T
        traj_roll = traj_roll.T

        lat1 = traj[0]
        lat2 = traj_roll[0]

        lon1 = traj[1]
        lon2 = traj_roll[1]

        term1 = haversine(lat2-lat1)
        term2 = (1 + haversine(lat1-lat2) - haversine(lat1+lat2)) * haversine(lon2-lon1)
        distances = 2 * EARTH_RADIUS_METERS * np.arcsin(np.sqrt(term1 + term2))

        return distances

    def compute_speed_dist(self, graph=False) -> np.array:
        """
        Computes the distribution of speeds (m/s) in the ULLT data.

        input:
        - graph: optional, whether or not to graph a histogram for the computed speed distribution

        output:
        - agent_speeds: array of agent speeds, averaged over their entire trajectory
        """

        agent_speeds = np.array([])
        total_teleports = 0

        for i, traj in enumerate(tqdm(self.data)):
            
            traj.columns = map(str.lower, traj.columns)

            columns = ['latitude', 'longitude']
            traj = traj[columns].to_numpy()

            if len(traj) > 600: 
                traj = traj[:600] # only keep the first 5 mins of simulation

            norms = KinematicEvaluator.compute_distance_m_from_latlon(traj)

            agent_speeds_i = norms 
            num_teleports = (agent_speeds_i > MAX_SPEED_THRESHOLD).sum() # teleporting if above 112 mph threshold
            agent_speeds_i = agent_speeds_i[agent_speeds_i <= MAX_SPEED_THRESHOLD] # only keep non-teleports
            agent_speeds = np.concatenate((agent_speeds, agent_speeds_i), axis=None)

            total_teleports += num_teleports

        if graph:
            save_path: str = os.path.join(self.fig_folder, "agent_speeds_pq1")
            KinematicEvaluator.plot_histogram(agent_speeds, "Speed Distribution: Parquet 1",
                                              "Agent Speed (m/s)", save_path)
        
        print("Total teleports: %d" % total_teleports)
        return np.array(agent_speeds)

    def compute_accel_dist(self, graph=False):
        """
        Computes the distribution of accelerations (m/s^2) in the ULLT data.

        input:
        - graph: optional, whether or not to graph a histogram for the computed acceleration distribution

        output:
        - agent_accels: array of agent accelerations, averaged over their entire trajectory
        """

        agent_accels = []

        for i, traj in enumerate(tqdm(self.data)):
            
            traj.columns = map(str.lower, traj.columns)
            columns = ['longitude', 'latitude']
            traj = traj[columns].to_numpy()

            norms = KinematicEvaluator.compute_distance_m_from_latlon(traj)
            norms1 = np.append(norms[1:], -1)
            norms = norms[norms != norms1]

            accels = np.diff(norms)

            agent_avg_accel = np.mean(accels)
            agent_accels.append(agent_avg_accel)

        if graph:
            save_path = os.path.join(self.fig_folder, "agent_accels_pq1")
            KinematicEvaluator.plot_histogram(agent_accels, "Accel Distribution: Parquet 1",
                                              "Agent Accel (m/s^2)", save_path)

        return np.array(agent_accels)


if __name__ == "__main__":
    path_test = "/home/local/KHQ/laura.zheng/haystac/data/parquet_1"
    evaluator_test = KinematicEvaluator(path_test)

    path_train = "/home/local/KHQ/laura.zheng/haystac/data/parquet_2"
    evaluator_train = KinematicEvaluator(path_train)

    start_time_speed = time.time()
    speed_dist_test = evaluator_test.compute_speed_dist(graph=False)
    speed_dist_train = evaluator_train.compute_speed_dist(graph=False)
    out_path = os.path.join(evaluator_train.fig_folder, "agent_speed_dist")
    KinematicEvaluator.plot_2_histograms(speed_dist_test, speed_dist_train,
                                         x_label="Speed (m/s)",
                                         reference="HumoNet Test",
                                         subject="HumoNet Train",
                                         out_name=out_path)
    speed_time = time.time() - start_time_speed

    start_time_accel = time.time()
    accel_dist_test = evaluator_test.compute_accel_dist(graph=False)
    accel_dist_train = evaluator_train.compute_accel_dist(graph=False)
    out_path = os.path.join(evaluator_train.fig_folder, "agent_accel_dist")
    KinematicEvaluator.plot_2_histograms(accel_dist_test, accel_dist_train,
                                         x_label="Accel (m/s^2)",
                                         reference="HumoNet Test",
                                         subject="HumoNet Train",
                                         out_name=out_path)
    accel_time = time.time() - start_time_accel

    print("speed distribution computation time: %f" % speed_time)
    print("accel distribution computation time: %f" % accel_time)
