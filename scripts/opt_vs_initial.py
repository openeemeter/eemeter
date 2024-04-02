import os
from pathlib import Path

#auto load the eemeter module
import warnings
warnings.filterwarnings("ignore")

from matplotlib import pyplot as plt
import numpy as np
import time
import pickle

from hourly_test_utils import *
from applied_data_science.bigquery.data import Meter_Data
from eemeter import eemeter as em
from eemeter.common.metrics import BaselineTestingMetrics as Metrics

import multiprocessing as mp



separate_folder = "/app/.recurve_cache/mce_3_yr_precovid/Hyperparameter_opt/separateV1"
subsamples = [1, 2, 3, 4, 5, 6]
has_solars = [True, False]
results = {}
for sub in subsamples:
    for has_solar in has_solars:
        path = os.path.join(separate_folder, f"subsample_{sub}_solar_meter_{has_solar}.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
            loss = data["loss"]
            x_opt = data["x_opt"]
            results[(sub, has_solar)] = (loss, x_opt)

print(results)

#load the whole populaiton best
best_res = {}
alltogether_file = "/app/.recurve_cache/mce_3_yr_precovid/Hyperparameter_opt/alltogetherV1/subsample_all_solar_meter_all.pkl"
with open(alltogether_file, "rb") as f:
    data = pickle.load(f)
    loss = data[0]
    x_opt = data[1]
    best_res["all"] = (loss, x_opt)


    
optimal_errors = {
    'train': [],
    'test': []
}
initial_errors = {
    'train': [],
    'test': []
}
subsamples = [1, 2, 3, 4, 5, 6]
has_solars = [True, False]
all_results = {}
# modes = ['initial', 'optimal', 'l1', 'l2', 'l12', 'overfit']
modes = ['global_optimal']
for sub in subsamples:
    for has_solar in has_solars:
        for mode in modes:
            print(sub, has_solar, mode)

            if mode == 'initial':
                settings = em.HourlySettings(
                    TRAIN_FEATURES=['ghi'],
                    ALPHA=0.1,
                    L1_RATIO=0.1,
                    SEED=42
                )
            elif mode == 'optimal':
                settings = em.HourlySettings(
                    TRAIN_FEATURES=['ghi'],
                    ALPHA=results[(sub, has_solar)][1][0],
                    L1_RATIO=results[(sub, has_solar)][1][1],
                    SEED=42
                )
            elif mode == 'l1':
                settings = em.HourlySettings(
                    TRAIN_FEATURES=['ghi'],
                    ALPHA=1,
                    L1_RATIO=1,
                    SEED=42
                )
            elif mode == 'l2':
                settings = em.HourlySettings(
                    TRAIN_FEATURES=['ghi'],
                    ALPHA=1,
                    L1_RATIO=0,
                    SEED=42
                )
            elif mode == 'l12':
                settings = em.HourlySettings(
                    TRAIN_FEATURES=['ghi'],
                    ALPHA=1,
                    L1_RATIO=0.5,
                    SEED=42
                )
            elif mode == 'overfit':
                settings = em.HourlySettings(
                    TRAIN_FEATURES=['ghi'],
                    ALPHA=0,
                    L1_RATIO=0,
                    SEED=42
                )
            elif mode == 'global_optimal':
                settings = em.HourlySettings(
                    TRAIN_FEATURES=['ghi'],
                    ALPHA=best_res["all"][1][0],
                    L1_RATIO=best_res["all"][1][1],
                    SEED=42
                )                                         

            solar_meters = [has_solar]
            subsamples = [sub]
            kwargs = {
                'settings': settings,
                'subsamples': subsamples,
                'solar_meters': solar_meters,
                'mp' : True,
                'max_id': -1
            }
            prf = Population_Run_Features(**kwargs)
            prf._load_data()
            prf.run()
            calc_id_kf_num = 0
            train_errors = []
            test_errors = []
            for res in prf.results:
                _, _, errors = res
                if (errors!=None) and (errors['train'][0]!= np.inf) and (errors['test'][0]!= np.inf):

                    calc_id_kf_num += 1
                    train_errors.append(errors['train'])
                    test_errors.append(errors['test'])
            print(len(prf.results))
            print(f"Calculated {calc_id_kf_num} id/kfold pairs")
            print('avg. train pnrmse:', np.mean(train_errors))
            print('avg. test pnrmse:', np.mean(test_errors))
            if mode == 'initial':
                initial_errors['train'].append(np.mean(train_errors))
                initial_errors['test'].append(np.mean(test_errors))
            else:
                optimal_errors['train'].append(np.mean(train_errors))
                optimal_errors['test'].append(np.mean(test_errors))
            all_results[(sub, has_solar, mode)] = (train_errors, test_errors)

# save the results
new_path = "/app/.recurve_cache/mce_3_yr_precovid/Hyperparameter_opt/best_global_results.pkl"
with open(new_path, "wb") as f:
    pickle.dump(all_results, f)