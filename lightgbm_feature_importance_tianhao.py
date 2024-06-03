import pickle
import os
import json
import traceback
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, roc_curve, auc, average_precision_score
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
#import lightgbm as lgb
import optuna.integration.lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import matplotlib.pyplot as plt

def plot_auroc_aupr(y_true, y_score, verbose=True):
    y_pred = np.rint(y_score)
    acc = accuracy_score(y_true, y_pred)

    if verbose:
        print(f'Accuracy: {acc}')

    cm = confusion_matrix(y_true, y_pred)
    if verbose:
        print('Confusion Matrix:')
        print(cm)

    # ROC, PR
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    # aupr = average_precision_score(y_true, y_score)
    try:
        aupr = auc(recall, precision)
    except ValueError as e:
        traceback.print_exc()
        aupr = 0.0

    if verbose:
        print(f'AUPR: {aupr}')

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    try:
        auroc = auc(fpr, tpr)
    except ValueError as e:
        traceback.print_exc()
        auroc = 0.0
    
    if verbose:
        print(f'AUROC: {auroc}')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)
    roc_display.plot(ax=ax1, label='roc')
    pr_display.plot(ax=ax2, label='prc')

    metrics = {
        'accuracy': acc,
        'confusion_matrix': cm.tolist(),
        'aupr': aupr,
        'auroc': auroc,
    }
    return metrics, fig


def lightgbm_feature_importance(X, y, test_size=0.2, random_state=None, save_path=None):
    if save_path is None:
        save_path = 'lightgbm_feature_importance'
    os.makedirs(save_path, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    num_train_pos = np.sum(y_train==1)
    num_train_neg = np.sum(y_train==0)
    num_test_pos = np.sum(y_test==1)
    num_test_neg = np.sum(y_test==0)
    print(f'Training set number of positive samples: {num_train_pos}, negative samples: {num_train_neg} (ratio: {num_train_pos/(num_train_pos+num_train_neg)})')
    print(f'Testing set number of positive samples: {num_test_pos}, negative samples: {num_test_neg} (ratio: {num_test_pos/(num_test_pos+num_test_neg)})')

    # model = lgb.LGBMClassifier(
    #     num_leaves=31,
    #     max_depth=-1,
    #     learning_rate=0.1,
    #     n_estimators=100,
    #     is_unbalance=True,
    #     #scale_pos_weight=(num_train_pos+num_train_neg)/num_train_pos,
    #     random_state=random_state,
    #     # default='split'. If ‘split’, result contains numbers of times the feature is used in a model. If ‘gain’, result contains total gains of splits which use the feature.
    #     importance_type='split', 
    #     verbosity=-1,
    # )

    # model.fit(X_train, y_train)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'sample_pos_weight': num_train_neg/num_train_pos,
    }

    # tuned from '/home/jxl220096/data/hay/haystac_trial1/fix_Baseline_TA1_Trial_Train_Submission/preprocess/full_train_raw_X_y.pkl'
    # params = {
    #     'objective': 'binary',
    #     'metric': 'binary_logloss',
    #     'verbosity': -1,
    #     'boosting_type': 'gbdt',
    #     'is_unbalanced': True,
    #     'random_state': None,
    #     'feature_pre_filter': False,
    #     'lambda_l1': 1.0404930253233646e-06,
    #     'lambda_l2': 0.002242148126928162,
    #     'num_leaves': 3,
    #     'feature_fraction': 0.9799999999999999,
    #     'bagging_fraction': 0.7531541955265642,
    #     'bagging_freq': 1,
    #     'min_child_samples': 20,
    #     'num_iterations': 1000
    # }

    dataset_train = lgb.Dataset(X_train, label=y_train)
    dataset_test = lgb.Dataset(X_test, label=y_test)

    model = lgb.train(
        params,
        dataset_train,
        valid_sets=[dataset_train, dataset_test],
        callbacks=[early_stopping(100), log_evaluation(100)],
    )

    best_params = model.params
    print('Best parameters:')
    print(best_params)

    y_score = model.predict(X_train)
    print('===  Training  ===')
    train_metrics, train_fig = plot_auroc_aupr(y_train, y_score)

    y_score = model.predict(X_test)
    print('=== Testing ===')
    test_metrics, test_fig = plot_auroc_aupr(y_test, y_score)

    importance = model.feature_importance('split')

    print('===            ===')
    print(f'Feature importance: {importance}')

    importance_gain = model.feature_importance('gain')

    all_metrics = {
        'train': train_metrics,
        'test': test_metrics,
        'feature_importance': importance.tolist(),
        'feature_importance_gain': importance_gain.tolist(),
    }
    if save_path is not None:
        model.save_model(os.path.join(save_path, 'model.lgb'))
        with open(os.path.join(save_path, 'params.json'), 'w') as f:
            json.dump(best_params, f, indent=4)

        with open(os.path.join(save_path, 'metrics.json'), 'w') as f:
            json.dump(all_metrics, f, indent=4)

        train_fig.savefig(os.path.join(save_path, 'train_roc_pr.png'))
        test_fig.savefig(os.path.join(save_path, 'test_roc_pr.png'))

    return importance, model, all_metrics


if __name__ == '__main__':

    for dataset in os.listdir('/home/jxl220096/data/hay/haystac_trial1/'):
        # if dataset != 'fix_L3Harris_TA1_Trial_Train_Submission':
        #     # Temp: just do L3Harris now
        #     continue
        dataset_path = os.path.join('/home/jxl220096/data/hay/haystac_trial1/', dataset)
        if os.path.islink(dataset_path):
            dataset_path = os.path.realpath(dataset_path)
        pickle_file_location = os.path.join(dataset_path, 'preprocess/full_train_raw_X_y.pkl')
        save_path = os.path.join('lightgbm_feature_importance/', dataset)

        if not os.path.exists(pickle_file_location):
            continue

        print(f'Processing {dataset}')
        with open(pickle_file_location, 'rb') as f:
            X, y = pickle.load(f)

        lightgbm_feature_importance(X, y, save_path=save_path)
