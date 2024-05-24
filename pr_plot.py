import os, sys, time
# from frechetdist import frdist
from discrete_frechet.distances.discrete import DiscreteFrechet, LinearDiscreteFrechet, VectorizedDiscreteFrechet, FastDiscreteFrechetSparse, FastDiscreteFrechetMatrix, DirectedHausdorff, DTW
from discrete_frechet.distances.discrete import euclidean, haversine, earth_haversine
import numpy as np
# import math
# import matplotlib
from matplotlib import pyplot as plt
# from typing import Callable, Dict
# from numba import jit, types, prange, int32, int64
# from numba import typed
# from timeit import default_timer as timer
import pickle
import multiprocessing
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve, average_precision_score, auc
import pandas as pd
import argparse


def chunk_file_paths(file_path_list, K):
    """
    It is designed to take a list of file paths and an integer K, and return a list of lists (chunks)
    where each sublist contains up to K file paths from the original list. This function is useful for
    processing or handling large lists of file paths in smaller, more manageable batches.

    INPUT
    file_path_list: A list of strings, where each string represents a path to a file. This list contains
    all the file paths that you want to divide into chunks.
    K: An integer representing the maximum number of file paths that should be included in each chunk (sublist).


    OUTPUT
    chunked_list: A list of lists (sublists). Each sublist contains up to K file paths from the original file_path_list.
    The last chunk may contain fewer than K file paths if the total number of file paths in file_path_list is not exactly divisible by K.
    """
    # Initialize the list to hold chunks of file names
    chunked_list = []

    # Iterate over the list and create sublists of size K or less
    for i in range(0, len(file_path_list), K):
        # Slice the list and append the chunk to the chunked_list
        chunked_list.append(file_path_list[i:i + K])

    return chunked_list

class Consumer(multiprocessing.Process):

    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        # proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means we should exit
                # print
                # '%s: Exiting' % proc_name
                break
            # print
            # '%s: %s' % (proc_name, next_task)
            answer = next_task()
            self.result_queue.put(answer)
        return

def get_dist_mat(traj_list_1, traj_list_2, dist = haversine, method =""):
    '''
    Input: 
    :traj_list_1:           Input list of 2-d sequence of stop points, each trajectory should have shape of [(n, 2), ...]
    :traj_list_2:           Another input list of 2-d sequence of stop points, each trajectory should have shape of [(n', 2), ...]
    :dist:                  Distance measure, pick from euclidean, haversine, earth_haversine
                            Haversine distance is calculated on either a unit sphere or the eath's surface
    :method:                Approach achieved in discrete_frechet.distances file

    Output:
    :frechet_dist_mat:      The frechet distance matrix that measures the distance between every pair of daily trajectorys
                            Shape: (n_days_1, n_days_2)
    
    TODO: add multi-thread processing
    '''
    # assert len(traj_list_1) > 1 and len(traj_list_2) > 1
    frechet_dist_mat = np.zeros((len(traj_list_1), len(traj_list_2)))
    # create solver instance
    if method == "":
        solver = DiscreteFrechet(dist)
    elif method == "Linear":
        solver = LinearDiscreteFrechet(dist)
    # elif method == "Fast":
    #     solver = FastDiscreteFrechet(dist)
    elif method == "Vectorized":
        solver = VectorizedDiscreteFrechet(dist)
    elif method == "FastSparse":
        solver = FastDiscreteFrechetSparse(dist)
    elif method == "FastMatrix": # fastest
        solver = FastDiscreteFrechetMatrix(dist)
    elif method == "DirectedHausdorff":
        solver = DirectedHausdorff(dist)
    elif method == "DTW":
        solver = DTW(dist)
    else:
        raise NotImplementedError

    # compute frechet distance matrix (14,14)
    # diagonals are 0


    if all(np.array_equal(a, b) for a, b in zip(traj_list_1, traj_list_2)):

    # if (traj_list_1 == traj_list_2):
        if len(traj_list_1) > 1:
            for i in range(len(traj_list_1)):
                for j in range(i + 1, len(traj_list_1)): # symmetric
                    curr_dist = solver.distance(np.array(traj_list_1[i]), np.array(traj_list_2[j]))
                    frechet_dist_mat[i, j], frechet_dist_mat[j, i] = curr_dist, curr_dist
        else:
            return np.array([[0]])
    else:
        for i in range(len(traj_list_1)):
            for j in range(len(traj_list_2)): # symmetric
                curr_dist = solver.distance(np.array(traj_list_1[i]), np.array(traj_list_2[j]))
                frechet_dist_mat[i, j] = curr_dist

    return frechet_dist_mat

def get_agents_dis_dist(agent_date_coords_dict_1, agent_date_coords_dict_2, args, dist = haversine, method =""):
    '''
    Trip-level frechet distance distribution vector

    Idea1: Calculate trip-level frechet distribution and calculate mahalanobis distance with a flat distribution

    Idea2: KL-div
    Idea3: entropy / margin / internal variation

    Input: 
    :agent_date_coords_dict_1:          Subset of the dictionary with chunked agents
    :agent_date_coords_dict_2:          Subset of the second dictionary with same chunked agents

    Output:
    :agent_frechet_dist_dict:           A dictionary of the structure: 
                                        {agent_id: [train_within_dist_matrix, test_within_dist_matrix, test_train_cross_dist_matrix]}
    '''
    agent_frechet_dist_dict = {}

    # start_time = time.time()
    # count = 0
    for agent_id, date_time_dict_train in agent_date_coords_dict_1.items():
        dict_list = []
        date_time_dict_test = agent_date_coords_dict_2[agent_id]

        # for time, tuple_traj_hour in date_time_dict_train.items():
        #     print(time, tuple_traj_hour)
        #     exit()
        if args.hour_of_day_flag:
            traj_list_train = [tuple_traj_hour for time, tuple_traj_hour in date_time_dict_train.items()] # all traj
            traj_list_test = [tuple_traj_hour for time, tuple_traj_hour in date_time_dict_test.items()] # all traj
        else:
            traj_list_train = [traj for time, traj in date_time_dict_train.items()] # all traj
            traj_list_test = [traj for time, traj in date_time_dict_test.items()] # all traj
        # # within train
        # dict_list.append(get_dist_mat(traj_list_train, traj_list_train, dist, method))
        # within test
        dict_list.append(get_dist_mat(traj_list_test, traj_list_test, dist, method))
        # cross
        dict_list.append(get_dist_mat(traj_list_train, traj_list_test, dist, method))
        agent_frechet_dist_dict[agent_id] = dict_list
        # count += 1
        # if count == 1000:
        #     print(count, time.time()-start_time)
        #     exit()

    return agent_frechet_dist_dict




def multi_process_frechet_dist(train_dataset_folder, test_dataset_folder, dist = haversine, method = "", num_consumers = 32):
    '''
    Input:
    :agent_date_stopp_coords_dict:      Dictionary of dictionary of ndarray, with agent_id, date time, and trajectory
    :dir:                               Output directory
    :num_consumers:                     Predefined consumer number

    Output:
    :agent_2_frechet_dist_matrix.pkl:   The dictionary with key of agent id and value of frechet_dist_mat
    '''

    with open(train_dataset_folder + "/preprocess/50k_train_agent_2_date_2_stopp_coords_dict.pkl", 'rb') as f:
        agent_date_coords_dict_train = pickle.load(f)

    with open(test_dataset_folder + "/preprocess/50k_test_agent_2_date_2_stopp_coords_dict.pkl", 'rb') as f:
        agent_date_coords_dict_test = pickle.load(f)

    agent_id_list_train = list(agent_date_coords_dict_train.keys())
    agent_id_list_test = list(agent_date_coords_dict_test.keys())
    agent_id_list_common = [id for id in agent_date_coords_dict_train.keys() if id in agent_date_coords_dict_test.keys()]


    tasks           = multiprocessing.Queue()
    results         = multiprocessing.Queue()
    n_agents        = len(agent_date_coords_dict_train.keys())
    # num_consumers   = 32
    if num_consumers > n_agents:
        num_consumers = int(np.ceil(n_agents * 0.5))
    
    K               = int(np.ceil(n_agents / num_consumers))
    consumers       = [Consumer(tasks, results) for _ in range(num_consumers)]
    for w in consumers:
        w.start()

    chunk_agent_id_list_train = chunk_file_paths(agent_id_list_train, K)
    chunk_agent_id_list_test = chunk_file_paths(agent_id_list_test, K)
    chunk_agent_id_list_common = chunk_file_paths(agent_id_list_common, K)

    frechet_dist_mat_dict = {}

    num_jobs                 = len(chunk_agent_id_list_train)
    count                    = 0
    debug_time_start         = time.time()
    for i_agent_id_list in chunk_agent_id_list_train:
        # slicing the agent dictionary
        chunk_agent_date_coords_dict_train = {agent: date_dict for agent, date_dict in agent_date_coords_dict_train.items() if agent in i_agent_id_list}
        chunk_agent_date_coords_dict_test = {agent: date_dict for agent, date_dict in agent_date_coords_dict_test.items() if agent in i_agent_id_list} # query by the training agent id list
        tasks.put(FeatureExtractionTask(chunk_agent_date_coords_dict_train, chunk_agent_date_coords_dict_test, dist = haversine, method = method)) # change this!
        count += 1
        if count % 10 == 0:
            print(count, 'tasks generated')

    # Add a poison pill for each consumer
    for _ in range(num_consumers):
        tasks.put(None)
    while num_jobs:
        chunk_agent_frechet_dist_dict = results.get() # catch the results
        # merge results
        frechet_dist_mat_dict = {**frechet_dist_mat_dict, **chunk_agent_frechet_dist_dict}
        # agent_id_list.extend(chunk_agents_id)

        num_jobs -= 1
        if num_jobs % 100 == 0:
            print(num_jobs, 'agents left')
    
    with open(test_dataset_folder + "/preprocess/agent_2_frechet_dist_mat_dict.pkl", 'wb') as f:
        pickle.dump(frechet_dist_mat_dict, f)
    print(f"Frechet distance dict saved to: {test_dataset_folder}")


class FeatureExtractionTask(object):
    def __init__(self, chunk_agent_date_coords_dict_train, chunk_agent_date_coords_dict_test, dist = haversine, method = "FastMatrix"):
        self.chunk_agent_date_coords_dict_train = chunk_agent_date_coords_dict_train
        self.chunk_agent_date_coords_dict_test = chunk_agent_date_coords_dict_test
        self.dist = dist
        self.method = method

    def __call__(self):
        agent_frechet_dist_dict = {}

        for agent_id, date_time_dict_train in self.chunk_agent_date_coords_dict_train.items():
            dict_list = []
            date_time_dict_test = self.chunk_agent_date_coords_dict_test[agent_id]
            traj_list_train = [traj for time, traj in date_time_dict_train.items()]
            traj_list_test = [traj for time, traj in date_time_dict_test.items()]
            # within train
            dict_list.append(get_dist_mat(traj_list_train, traj_list_train, self.dist, self.method))
            # within test
            dict_list.append(get_dist_mat(traj_list_test, traj_list_test, self.dist, self.method))
            # cross
            dict_list.append(get_dist_mat(traj_list_train, traj_list_test, self.dist, self.method))
            agent_frechet_dist_dict[agent_id] = dict_list

        return agent_frechet_dist_dict

def retrieve_abnormal_agent_id_list(test_dataset_folder):

    test_dataset_folder_gts_dict = {
        '/home/jxl220096/data/hay/new_format/trial2/sanfrancisco/test_stops/' : '/home/jxl220096/data/hay/new_format/trial2/gts/sf_gts',
        '/home/jxl220096/data/hay/new_format/trial2/knoxville/test_stops/' : '/home/jxl220096/data/hay/new_format/trial2/gts/kx_gts',
        '/home/jxl220096/data/hay/new_format/trial2/losangeles/test_stops/' : '/home/jxl220096/data/hay/new_format/trial2/gts/la_gts',
    }
    agent_anomaly_days_dict = np.load(os.path.join(train_dataset_folder, "preprocess/agent_anomaly_days_dict.npy"), allow_pickle=True).item()
    abnormal_agent_id_list = list(agent_anomaly_days_dict.keys())

    # abnormal_agent_id_list = []
    # for filename in test_dataset_folder_gts_dict[test_dataset_folder]:
    #     if filename.startswith('agent='):
    #         abnormal_agent_id_list.append(np.int64(filename.replace('agent=', '')))
    return abnormal_agent_id_list


def frechet_dist_score(train_dataset_folder, test_dataset_folder, method = "agent", n_agents = "5k"):
    '''
    Calculate the anomaly score based on each agent's frechet distance matrix. 
    Input:
    :test_dataset_folder:           Testing dataset folder that includes the saved /preprocess/agent_2_frechet_dist_mat_dict.pkl
    :train_dataset_folder:          Used to extract ground-truth of each agent


    Output:
    :anomaly_score:                 Transformed anomaly score
    :PRplot:                        PR curve showing the detection result of the anonaly score
    '''
    abnormal_agent_id_list = retrieve_abnormal_agent_id_list(test_dataset_folder)
    y_dict = {}

    with open(test_dataset_folder + "/preprocess/agent_2_frechet_dist_mat_dict.pkl", 'rb') as f:
        frechet_dist_mat_dict = pickle.load(f)
    
    # generate new anomaly score
    anomaly_score_dict = {}
    n_agents = len(frechet_dist_mat_dict.keys())
    for agent_id, dist_mat_list in frechet_dist_mat_dict.items():
        if method == "agent":
            anomaly_score_dict[agent_id] = np.max(np.min(dist_mat_list[-1], axis = 1))
        elif method == "l2norm":
            anomaly_score_dict[agent_id] = np.linalg.norm(dist_mat_list[-1], ord = 2)
        elif method == "l1norm":
            anomaly_score_dict[agent_id] = np.linalg.norm(dist_mat_list[-1], ord = 1)
        elif method == "max":
            anomaly_score_dict[agent_id] = np.max(dist_mat_list[-1])
        elif method == "mean_quantile": # better
            anomaly_score_dict[agent_id] = np.mean(np.percentile(dist_mat_list[-1], 0.75, axis = 1))
        elif method == "sum_quantile": # better
            anomaly_score_dict[agent_id] = np.sum(np.percentile(dist_mat_list[-1], 0.75, axis = 1))
        elif method == "mean":
            anomaly_score_dict[agent_id] = np.quantile(np.mean(dist_mat_list[-1], axis = 1), 0.5)
        elif method == "quantile_quantile":
            anomaly_score_dict[agent_id] = np.percentile(np.percentile(dist_mat_list[-1], 0.25, axis = 1), 0.75)
        elif method == "ratio_l2norm":
            dist_ratio_mat = np.abs(dist_mat_list[-1] / np.mean(np.percentile(dist_mat_list[0], 0.75, axis = 1)))
            anomaly_score_dict[agent_id] = np.linalg.norm(dist_ratio_mat, ord = 2)
        elif method == "ratio_l1norm":
            dist_ratio_mat = np.abs(dist_mat_list[-1] / np.mean(np.percentile(dist_mat_list[0], 0.75, axis = 1)))
            anomaly_score_dict[agent_id] = np.linalg.norm(dist_ratio_mat, ord = 1)
        elif method == "quantile_ratio": # better ~0.28
            anomaly_score_dict[agent_id] = np.abs(np.mean(np.percentile(dist_mat_list[-1], 0.75, axis = 1)) / np.mean(dist_mat_list[0]))
        elif method == "quantile_ratio_v2": # better ~0.3
            anomaly_score_dict[agent_id] = np.abs(np.mean(np.percentile(dist_mat_list[-1], 0.75, axis = 1)) / np.max(dist_mat_list[0]))
        elif method == "quantile_diff":
            anomaly_score_dict[agent_id] = np.abs(np.mean(np.percentile(dist_mat_list[-1], 0.75, axis = 1)) - np.mean(dist_mat_list[0]))
        elif method == "quantile_diff_v2":
            anomaly_score_dict[agent_id] = np.abs(np.mean(np.percentile(dist_mat_list[-1], 0.75, axis = 1)) - np.max(dist_mat_list[0]))
        elif method == "Mahalanobis":
            cov = np.cov(dist_mat_list[-1], rowvar = False)
            inv_cov = inv(cov)
            m_dist_mat = np.zeros()
            ...
        elif method == "ROCKET":
            ...
        elif method == "DTW":
            from tslearn.metrics import dtw, cdist_dtw
            dtw_mat_cross = cdist_dtw(dist_mat_list[-1].T, dist_mat_list[0].T)
            dtw_mat_train = cdist_dtw(dist_mat_list[0].T)
            # anomaly_score_dict[agent_id] = np.quantile(dtw_mat_cross @ dtw_mat_train, 0.75) # ~0.0308
            anomaly_score_dict[agent_id] = np.quantile(np.sum(dtw_mat_cross, axis = 0), 0.75) # ~0.0360
            
        if agent_id in abnormal_agent_id_list:
            y_dict[agent_id] = 1
        else:
            y_dict[agent_id] = 0

    anomaly_score_list = [float(score) for score in anomaly_score_dict.values()]
    y_list = [float(y) for y in y_dict.values()]

    # visualize PR plot
    fig, ax = plt.subplots(1, 1, figsize = (6, 6), dpi = 320)
    prec, recall, _ = precision_recall_curve(y_list, anomaly_score_list, pos_label = None)
    aupr = average_precision_score(y_list, anomaly_score_list)
    print(f"AUPR is: {aupr}")
    ax.set_title(f"{n_agents}_agents_{method}_AUPR_{aupr}")
    pr_display = PrecisionRecallDisplay(precision = prec, recall = recall).plot(ax = ax)
    if not os.path.exists(test_dataset_folder + "/preprocess/plots/"):
        os.makedirs(test_dataset_folder + "/preprocess/plots/")
    fig.savefig(test_dataset_folder + f'/preprocess/plots/{n_agents}_frechet_dist_PR_plot_{method}.png')

    pass

def get_agents_dist_score(frechet_dist_mat_dict):
    num_agents = len(list(frechet_dist_mat_dict.keys()))
    global_max = -float('inf')
    global_min = float('inf')
    agents_id_ndarray = np.zeros(num_agents)
    agents_dist_score_ndarray = np.zeros(num_agents)
    count = 0
    for key, value in frechet_dist_mat_dict.items():
        temp_agent_max = -float('inf')
        for temp_matrix in value:
            temp_matrix_max = temp_matrix.max()
            if temp_matrix_max > temp_agent_max:
                temp_agent_max = temp_matrix_max
        if temp_agent_max > global_max:
            global_max = temp_agent_max
        if temp_agent_max < global_min:
            global_min = temp_agent_max
        agents_id_ndarray[count] = key
        agents_dist_score_ndarray[count] = temp_agent_max
        count += 1
    agents_dist_score_ndarray = (agents_dist_score_ndarray - global_min) / (global_max - global_min)
    agents_dist_score_df = pd.DataFrame({'agent_id': agents_id_ndarray, 'anomaly_score': agents_dist_score_ndarray})
    return agents_dist_score_df

def get_dist_score_plots(test_dataset_folder, agents_dist_score_df, args):
    abnormal_agent_id_list = retrieve_abnormal_agent_id_list(test_dataset_folder)

    agents_dist_score_df['label'] = 0
    for temp_agent in abnormal_agent_id_list:
        agents_dist_score_df.loc[agents_dist_score_df['agent_id'] == temp_agent, 'label'] = 1
    anomaly_score = agents_dist_score_df['anomaly_score'].values
    labels = agents_dist_score_df['label'].values
    lr_precision, lr_recall, thresholds = precision_recall_curve(labels, anomaly_score)
    aupr = auc(lr_recall, lr_precision)
    fontsize = 14
    plt.plot(lr_recall, lr_precision, marker='.', lw=2, label=aupr)
    plt.xlabel('Recall', fontsize=fontsize)
    plt.ylabel('Precision', fontsize=fontsize)
    plt.title(f'PR Curve of {args.method} Distance between Daily Trajectory')
    plt.legend()
    plt.tight_layout()

    directory = os.path.join(test_dataset_folder, "plots")
    if not os.path.exists(directory):
        os.makedirs(directory)

    if args.hour_of_day_flag:
        plt.savefig(os.path.join(test_dataset_folder, "plots/pr_trial2" + args.method + "_distance_daily_trajectory_addhour.png"))
    else:
        plt.savefig(os.path.join(test_dataset_folder, "plots/pr_trial2" + args.method + "_distance_daily_trajectory.png"))
    plt.show()
    plt.close('all')














if __name__ == "__main__":

    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_dir', type=str,
                        default='/data/jxl220096/hay/trial2_oldformat/losangeles_eventlogs_train/')
    parser.add_argument('--test_dataset_dir', type=str,
                        default='/data/jxl220096/hay/trial2_oldformat/losangeles_test_event_logs/')
    parser.add_argument('--method', type=str, default='Linear', help='distance metric: Linear, DirectedHausdorff, DTW')
    parser.add_argument('--n_agents', type=str, default="50k")
    parser.add_argument('--hour_of_day_flag', type=bool, default=True)
    args = parser.parse_args()
    train_dataset_folder = args.train_dataset_dir
    test_dataset_folder = args.test_dataset_dir






    method = args.method
    n_agents = args.n_agents


    if args.hour_of_day_flag:
        with open(os.path.join(train_dataset_folder, f"preprocess/train_{args.n_agents}_agent_date_time_sp_coords_dict.pkl"), 'rb') as f:
            agent_date_coords_dict_train = pickle.load(f)

        with open(os.path.join(test_dataset_folder + f"preprocess/test_{args.n_agents}_agent_date_time_sp_coords_dict.pkl"), 'rb') as f:
            agent_date_coords_dict_test = pickle.load(f)

    else:
        # single-thread
        with open(os.path.join(train_dataset_folder, f"preprocess/bw-0.1-K-10-25-rawX-0-agg-0-STKDE-0-comb-0-aggX-1/{n_agents}_train_agent_2_date_2_stopp_coords_dict.pkl"), 'rb') as f:
            agent_date_coords_dict_train = pickle.load(f)

        with open(os.path.join(test_dataset_folder + f"preprocess/bw-0.1-K-10-25-rawX-0-agg-0-STKDE-0-comb-0-aggX-1/{n_agents}_test_agent_2_date_2_stopp_coords_dict.pkl"), 'rb') as f:
            agent_date_coords_dict_test = pickle.load(f)


    time1 = time.time()
    frechet_dist_mat_dict = get_agents_dis_dist(agent_date_coords_dict_train, agent_date_coords_dict_test, args, dist = haversine, method =method)

    with open(os.path.join(test_dataset_folder, "preprocess/agent_2_" + method + "_dist_mat_dict.pkl"), 'wb') as f:
        pickle.dump(frechet_dist_mat_dict, f)
    print(f"Frechet distance dict saved to: {test_dataset_folder}")

    # saved at /home/jxl220096/data/hay/haystac_trial1/fix_LosAngeles_Test_dataset/preprocess/agent_2_frechet_dist_mat_dict.pkl

    # PR curve visualization
    with open(os.path.join(test_dataset_folder, "preprocess/agent_2_" + method + "_dist_mat_dict.pkl"), 'rb') as f:
        frechet_dist_mat_dict = pickle.load(f)
    agents_dist_score_df = get_agents_dist_score(frechet_dist_mat_dict)
    get_dist_score_plots(test_dataset_folder, agents_dist_score_df, args)
    time2 = time.time()
    print("time:", time2 - time1)
    # frechet_dist_score(train_dataset_folder, test_dataset_folder, method = "DTW", n_agents = n_agents) # agent level or trip-level anomaly detection

    pass



# python pr_plot.py --train_dataset_dir '/home/jxl220096/data/hay/new_format/trial2/sanfrancisco/train_stops/' --test_dataset_dir '/home/jxl220096/data/hay/new_format/trial2/sanfrancisco/test_stops/' --method Linear
# python pr_plot.py --train_dataset_dir '/home/jxl220096/data/hay/new_format/trial2/sanfrancisco/train_stops/' --test_dataset_dir '/home/jxl220096/data/hay/new_format/trial2/sanfrancisco/test_stops/' --method DirectedHausdorff
# python pr_plot.py --train_dataset_dir '/home/jxl220096/data/hay/new_format/trial2/sanfrancisco/train_stops/' --test_dataset_dir '/home/jxl220096/data/hay/new_format/trial2/sanfrancisco/test_stops/' --method DTW 

# python pr_plot.py --train_dataset_dir '/home/jxl220096/data/hay/new_format/trial2/knoxville/train_stops/' --test_dataset_dir '/home/jxl220096/data/hay/new_format/trial2/knoxville/test_stops/' --method Linear
# python pr_plot.py --train_dataset_dir '/home/jxl220096/data/hay/new_format/trial2/knoxville/train_stops/' --test_dataset_dir '/home/jxl220096/data/hay/new_format/trial2/knoxville/test_stops/' --method DirectedHausdorff
# python pr_plot.py --train_dataset_dir '/home/jxl220096/data/hay/new_format/trial2/knoxville/train_stops/' --test_dataset_dir '/home/jxl220096/data/hay/new_format/trial2/knoxville/test_stops/' --method DTW 

# python pr_plot.py --train_dataset_dir '/home/jxl220096/data/hay/new_format/trial2/losangeles/train_stops/' --test_dataset_dir '/home/jxl220096/data/hay/new_format/trial2/losangeles/test_stops/' --method Linear
# python pr_plot.py --train_dataset_dir '/home/jxl220096/data/hay/new_format/trial2/losangeles/train_stops/' --test_dataset_dir '/home/jxl220096/data/hay/new_format/trial2/losangeles/test_stops/' --method DirectedHausdorff
# python pr_plot.py --train_dataset_dir '/home/jxl220096/data/hay/new_format/trial2/losangeles/train_stops/' --test_dataset_dir '/home/jxl220096/data/hay/new_format/trial2/losangeles/test_stops/' --method DTW 
