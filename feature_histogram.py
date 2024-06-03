import pickle
import os
import json
import traceback
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import sys


def plot_feature_histograms(X, y, feature_index, feature_name, save_path):
    # Separate normal and abnormal agents
    normal = X[y == 0, feature_index]
    abnormal = X[y == 1, feature_index]
    
    # Create histogram
    plt.figure(figsize=(8, 5))
    plt.hist(normal, bins=40, alpha=0.6, label='Normal', color='blue', density=True)
    plt.hist(abnormal, bins=40, alpha=0.6, label='Abnormal', color='red', density=True)
    plt.title(f'Histogram for Feature {feature_index + 1} {feature_name}')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    plt.savefig( os.path.join( save_path, f'feature_{feature_index + 1}_{feature_name}.png'))
    plt.close()

def PCA_analysis(X, y, save_path):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)   
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pc1 = X_pca[:, 0]
    pc2 = X_pca[:, 1]

    plt.figure(figsize=(8, 6))
    for label, color in zip([0, 1], ['blue', 'red']):
        plt.scatter(pc1[y == label], pc2[y == label], label=f'{"Normal" if label == 0 else "Abnormal"}', color=color, alpha=0.7)

    plt.title('PCA of Agent Features')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.savefig(os.path.join( save_path, 'PCA of Agent Features.png'))



if __name__ == '__main__':

    start = time.time()
    args = sys.argv
    if len(args) == 5:
        train_dataset_folder = args[1]
        fis_folder = args[2]
        dataset = args[3]
        std_flag = args[4]
    else:
        raise Exception("no dataset")

    if std_flag =='wo':
        with open(train_dataset_folder+'/preprocess/val_wo_std_X_y.pkl', 'rb') as f:
            X1, y1 = pickle.load(f)
        with open(train_dataset_folder+'/preprocess/train_wo_std_X_y.pkl', 'rb') as f:
            X2, y2 = pickle.load(f)
    elif std_flag == 'top_low' or std_flag == 'low':
        with open(train_dataset_folder+'/preprocess/val_wo_std_X_y_top3_low3.pkl', 'rb') as f:
            X1, y1 = pickle.load(f)
        with open(train_dataset_folder+'/preprocess/train_wo_std_X_y_top3_low3.pkl', 'rb') as f:
            X2, y2 = pickle.load(f)
    else:
        with open(train_dataset_folder+'/preprocess/val_w_std_X_y.pkl', 'rb') as f:
            X1, y1 = pickle.load(f)
        with open(train_dataset_folder+'/preprocess/train_w_std_X_y.pkl', 'rb') as f:
            X2, y2 = pickle.load(f)
        

    X1_clean = []
    X2_clean = []
    y1_clean = []
    y2_clean = []

    for item1, label in zip(X1, y1):
        if std_flag =='wo' or std_flag =='norm' :
            if len(item1) == 12 and not np.isnan(item1).any():
                X1_clean.append(item1)
                y1_clean.append(label)
        elif std_flag == 'top_low' or std_flag == 'low':
            if len(item1) == 24 and not np.isnan(item1).any():
                X1_clean.append(item1)
                y1_clean.append(label)
    for item2, label in zip(X2, y2):
        if std_flag =='wo' or std_flag =='norm' :
            if len(item2) == 12 and not np.isnan(item2).any():
                X2_clean.append(item2)
                y2_clean.append(label)
        elif std_flag == 'top_low' or std_flag == 'low':
            if len(item2) == 24 and not np.isnan(item2).any():
                X2_clean.append(item2)
                y2_clean.append(label)  

    # X2_clean = X2_clean[:100]
    # y2_clean = y2_clean[:100]

    print('X2_clean', len(X2_clean))

    # X2_clean = X2_clean[:10000]
    # y2_clean = y2_clean[:10000]

    X = X1_clean + X2_clean
    y = y1_clean + y2_clean

    X_array = np.array(X)
    y_array = np.array(y)

    X_array = X_array[:, -12:]

    feature_name_dict = {
        '0':'top1_travel_distance_anomaly',
        '1':'top2_travel_distance_anomaly',
        '2':'top3_travel_distance_anomaly',
        '3':'top1_travel_time_anomaly',
        '4':'top2_travel_time_anomaly',
        '5':'top3_travel_time_anomaly',
        '6':'top1_unique_locations_anomaly',
        '7':'top2_unique_locations_anomaly',
        '8':'top3_unique_locations_anomaly',
        '9':'top1_duration_anomalies',
        '10':'top2_duration_anomalies',
        '11':'top3_duration_anomalies',
        '12':'lowest1_travel_distance_anomaly',
        '13':'lowest2_travel_distance_anomaly',
        '14':'lowest3_travel_distance_anomaly',
        '15':'lowest1_travel_time_anomaly',
        '16':'lowest2_travel_time_anomaly',
        '17':'lowest3_travel_time_anomaly',
        '18':'lowest1_unique_locations_anomaly',
        '19':'lowest2_unique_locations_anomaly',
        '20':'lowest3_unique_locations_anomaly',
        '21':'lowest1_duration_anomalies',
        '22':'lowest2_duration_anomalies',
        '23':'lowest3_duration_anomalies',
    }

    # Plot histograms for each feature
    save_path = os.path.join('agent_feature_hist/', dataset)
    os.makedirs(save_path, exist_ok=True)


    # for i in range(X_array.shape[1]):
    #     plot_feature_histograms(X_array, y_array, i, feature_name_dict[str(i)], save_path)

    PCA_analysis(X_array, y_array, save_path)

    

# python feature_histogram.py /home/jxl220096/data/hay/new_format/trial2/losangeles/train_stops /home/jxl220096/data/hay/new_format/trial2/losangeles/test_stops_valsplit losangeles_wo_std wo
# python feature_histogram.py /home/jxl220096/data/hay/new_format/trial2/losangeles/train_stops /home/jxl220096/data/hay/new_format/trial2/losangeles/test_stops_valsplit losangeles_std norm
# python feature_histogram.py /home/jxl220096/data/hay/new_format/trial2/sanfrancisco/train_stops /home/jxl220096/data/hay/new_format/trial2/sanfrancisco/test_stops_valsplit sanfrancisco_wo_std wo
# python feature_histogram.py /home/jxl220096/data/hay/new_format/trial2/sanfrancisco/train_stops /home/jxl220096/data/hay/new_format/trial2/sanfrancisco/test_stops_valsplit sanfrancisco_std norm
# python feature_histogram.py /home/jxl220096/data/hay/new_format/trial2/singapore/train_stops /home/jxl220096/data/hay/new_format/trial2/singapore/test_stops_valsplit singapore_wo_std wo
# python feature_histogram.py /home/jxl220096/data/hay/new_format/trial2/singapore/train_stops /home/jxl220096/data/hay/new_format/trial2/singapore/test_stops_valsplit singapore_std norm

# python feature_histogram.py /home/jxl220096/data/hay/new_format/trial2/losangeles/train_stops /home/jxl220096/data/hay/new_format/trial2/losangeles/test_stops_valsplit losangeles_wo_std_top_low top_low
# python feature_histogram.py /home/jxl220096/data/hay/new_format/trial2/sanfrancisco/train_stops /home/jxl220096/data/hay/new_format/trial2/sanfrancisco/test_stops_valsplit sanfrancisco_wo_std_top_low top_low
# python feature_histogram.py /home/jxl220096/data/hay/new_format/trial2/singapore/train_stops /home/jxl220096/data/hay/new_format/trial2/singapore/test_stops_valsplit singapore_wo_std_top_low top_low

# python feature_histogram.py /home/jxl220096/data/hay/new_format/trial2/losangeles/train_stops /home/jxl220096/data/hay/new_format/trial2/losangeles/test_stops_valsplit losangeles_wo_std_low low
# python feature_histogram.py /home/jxl220096/data/hay/new_format/trial2/sanfrancisco/train_stops /home/jxl220096/data/hay/new_format/trial2/sanfrancisco/test_stops_valsplit sanfrancisco_wo_std_low low
# python feature_histogram.py /home/jxl220096/data/hay/new_format/trial2/singapore/train_stops /home/jxl220096/data/hay/new_format/trial2/singapore/test_stops_valsplit singapore_wo_std_low low
