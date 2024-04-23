#!/usr/bin/env python3

import os
os.environ["OMP_NUM_THREADS"] = "1" # take that multithreading
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from copy import deepcopy as copy
from timeit import default_timer as timer
from pathlib import Path
import pandas as pd
import numpy as np
import random
import warnings
warnings.simplefilter("ignore")

import multiprocessing as mp

from applied_data_science.bigquery.data import Meter_Data
from applied_data_science.tools.optimization.optimize import Optimize


datasets = {'socalgas': [1, 2, 3, 4],
            'mce': [1, 31]}

obj_lambda = 1.0
k_fold_number = 10
n_max = 1000

seed = 1576

try_except = True
use_test_train = True
mp_meter = True

df_cache_dir = Path("/app/.recurve_cache/caltrack_2_1").resolve()
data_cache_dir = Path("/app/.recurve_cache/data").resolve()

x0 = [0.1, 0.1]

bnds = [
    [  -10,   10 ],    # alpha:   regularization alpha
    [   1E-4,   1 ],    # l1_ratio:    regularization l1_ratio
]

# Optimization settings
opt_options = {
    "global": {"algorithm": "RBFOpt", 
               "stop_criteria_type": 'Iteration Maximum', 
               "stop_criteria_val": 1000, 
               "initial_step": 0.01, # percentage},
               "xtol_rel": 1E-5,
               "ftol_rel": 1E-5,
               "initial_pop_multiplier": 2,
    },
    # "local":  {"algorithm": "nlopt_SBPLX", 
    #            "stop_criteria_type": 'Iteration Maximum', 
    #            "stop_criteria_val": 1000, 
    #            "initial_step": 0.15, # percentage},
    #            "xtol_rel": 1E-5,
    #            "ftol_rel": 1E-5,
    # },
}

# CalTrack 2.1 SSE
settings_dict = {
    "alpha_selection": 2.0,
    "alpha_final": 2.0,
    "regularization_alpha": 0.01,
    "regularization_percent_lasso": 0.75,
    "iterative_model_initial_guesses": True,
    "fit_tidd_model": True,
    "fit_c_hdd_model": True,
    "fit_c_hdd_tidd_model": True,
    "fit_c_hdd_tidd_smooth_model": True,
    "fit_hdd_tidd_cdd_model": True,
    "fit_hdd_tidd_cdd_smooth_model": True,
    "allow_separate_summer": True,
    "allow_separate_shoulder": True,
    "allow_separate_winter": True,
    "allow_separate_weekday_weekend": True,
    "reduce_splits_by_gaussian": False,
    "reduce_splits_num_std": [1.4, 0.9],
    "split_selection_criteria": "BIC",
    "split_selection_penalty_multiplier": 0.8,
    "split_selection_penalty_power": 2.1,
}

# Correct x0 and bounds
x0 = np.array(x0)
x0[0] = np.log10(x0[0])

bnds = np.array(bnds)
bnds[0, :] = np.log10(bnds[0, :])

# Path/cwd setup
os.chdir(Path(__file__).parent.absolute())
main_path = Path(__file__).absolute().parent

df_cache_dir.mkdir(parents=True, exist_ok=True)
df_caltrack_2_0_csv_path = df_cache_dir / "df_caltrack_2_0.csv"
df_cache_path = df_cache_dir / "df_cache.csv"


def fit_meter(args_list):
    meter_id_data, i, settings, use_test_train, mp_k_fold, try_except = args_list

    start_time = timer()
    if try_except:
        try:
            fit = fit_model(meter_id_data, settings, use_test_train=use_test_train, multiprocessing=mp_k_fold, print_res=False)     
        except:
            fit = None
            fit_time = timer() - start_time
    
    else:
        fit = fit_model(meter_id_data, settings, use_test_train=use_test_train, multiprocessing=mp_k_fold, print_res=False)
    
    fit_time = timer() - start_time

    dataset = meter_id_data["dataset"][0]
    subsample = meter_id_data["subsample"][0]

    return fit, dataset, subsample, i, fit_time


def worker_function(item, q):
    res = fit_meter(item)
    q.put(res)

    return res


def listener(q, verbose=False):
    """
    continue to listen for messages on the queue and writes to file when receive one
    if it receives a '#done#' message it will exit
    """
    while True:
        res = q.get()

        if res == '#done#':
            break

        if verbose:
            fit, dataset, subsample, i, fit_time = res
            print(f"{dataset:^10s} {subsample:>4} {i:>4} {fit_time:.1f} s")
        
        # f.flush()


def load_data(datasets, n_max):
    id_used = []
    data_all = []
    for [dataset, subsample] in datasets:
        data = Meter_Data(dataset, subsample, "daily", cache_dir=data_cache_dir, verbose=False)

        data.set_test_train(
            force_test_train_reload=False,
            period='baseline',
            weekend_weekday_rnd=True,
            season_rnd=True,
            kfold_number=k_fold_number,
            test_size=0.2,
            bins_type='doane',
            multiprocessing=True,
            debug=False,
        )

        dataset_ids = data.df["meta"].index.unique().to_list()
        random.seed(seed)
        random.shuffle(dataset_ids)

        ids = []
        for id in dataset_ids:
            if id not in id_used:
                ids.append(id)
                id_used.append(id)

            if len(ids) >= n_max:
                break

        data = data.df['meter'].loc[ids]
        data["dataset"] = dataset
        data["subsample"] = subsample

        data_all.append(data)

    return pd.concat(data_all)


def obj_fcn_dec(data, obj_lambda=0.6, df_caltrack_2_0=None, settings_dict={}, mp_meter=True):    
    def get_args_list(data, settings):
        args_list = []
        for i, id in enumerate(data.index.unique().to_list()):
            data_meter = data.loc[id]
            
            mp_k_fold = False
            args_list.append([data_meter, i, settings, use_test_train, mp_k_fold, try_except])

        return args_list
    
    def fit_all_datasets(settings):
        args_list = get_args_list(data, settings)
        
        if mp_meter:
            manager = mp.Manager()
            q = manager.Queue()
            file_pool = mp.Pool(1)
            file_pool.apply_async(listener, (q, ))

            pool = mp.Pool(processes=mp.cpu_count())
            jobs = []
            for item in args_list:
                job = pool.apply_async(worker_function, (item, q))
                jobs.append(job)

            res = []
            for job in jobs:
                res.append(job.get())

            q.put('#done#')  # all workers are done, we close the output file
            pool.close()
            pool.join()
            file_pool.close()

        else:
            res = []
            for args_list_i in args_list:
                res.append(fit_meter(args_list_i))

        res = [item for item in res if ((item is not None) and (item[0] is not None))]
        res = [[item[1], item[2], item[0].id, item[0].error['RMSE_test'], item[0].error['MBE_test']] for item in res]
        # res = [[item[1], item[2], item[0].id, item[0].error['RMSE_test'] + item[0].error['RMSE_test_PI'], item[0].error['MBE_test'] + item[0].error['MBE_test_PI']] for item in res]

        df_res = pd.DataFrame(res, columns=["dataset", "subsample", "id", "RMSE", "MBE"])

        # print(f"total time: {(timer() - start)/60:.2f} min\n")

        return df_res

    def obj_fcn(X, gradient=np.array([])):
        timer_start = timer()

        X = np.array(X)
        X[0] = 10**X[0] # alpha comes as log(alpha)
        alpha, beta, omega, eta = X

        # Reads from cache, if it already exists, then pass that
        df_res_cache = None
        if df_cache_path.is_file():
            df_res_cache = pd.read_csv(df_cache_path)

            cache_values = df_res_cache[["alpha", "beta", "omega", "eta"]].values

            idx_cache = np.argwhere(np.isclose(cache_values, X, rtol=1E-5, atol=1E-8).all(axis=1)).flatten()
            
            if len(idx_cache) > 0:
                return float(df_res_cache.iloc[idx_cache]["obj_fcn"].values[0])

        settings_dict.update({
            "regularization_alpha": alpha,
            "regularization_percent_lasso": beta,
            "split_selection_penalty_multiplier": omega,
            "split_selection_penalty_power": eta,
        })
        settings = Daily_Settings(**settings_dict)

        df_res = fit_all_datasets(settings)
        
        df_caltrack_base = copy(df_caltrack_2_0)
        df_res = df_res.merge(df_caltrack_base, on=["dataset", "subsample", "id"], suffixes=[None, "_base"])

        obj = np.mean(obj_lambda*df_res["RMSE"]/df_res["RMSE_base"] + (1 - obj_lambda)*df_res["MBE"]/df_res["MBE_base"])

        # Make dataframe for saving
        df_res = pd.DataFrame({
            "alpha": alpha,
            "beta": beta,
            "omega": omega,
            "eta": eta,
            "obj_fcn": obj,
            "eval_time": timer() - timer_start
        }, index=[0])

        if df_res_cache is None:
            df_res.to_csv(df_cache_path, mode="w", header=True, index=False)
        else:
            df_res.to_csv(df_cache_path, mode="a", header=None, index=False)
        
        return float(obj)
    
    # if caltrack has not been calculated then output it
    if df_caltrack_2_0 is None:
        return fit_all_datasets(caltrack_legacy_settings())
    
    else:
        return obj_fcn


if __name__ == "__main__":
    datasets = [[dataset, subsample] for dataset, subsamples in datasets.items() for subsample in subsamples]
    data = load_data(datasets, n_max)

    # if caltrack 2.0 hasn't been calculated (as baseline), then do it and save it
    if not df_caltrack_2_0_csv_path.is_file():
        df_caltrack_2_0 = obj_fcn_dec(data, obj_lambda=obj_lambda, mp_meter=mp_meter)
        df_caltrack_2_0.to_csv(df_caltrack_2_0_csv_path, mode="w", header=True, index=False)
    else:
        df_caltrack_2_0 = pd.read_csv(df_caltrack_2_0_csv_path)

    # add everything in cache to x0 if it exists
    if df_cache_path.is_file() and ("global" in opt_options) and (opt_options["global"]["algorithm"].lower() == "rbfopt"):
        df_res_cache = pd.read_csv(df_cache_path)

        res_cache = df_res_cache[["alpha", "beta", "omega", "eta"]].values
        res_cache[:, 0] = np.log10(res_cache[:, 0])

        x0 = np.vstack([res_cache, x0])

        # only unique guesses
        idx = np.unique(x0, axis=0, return_index=True)[1]
        x0 = x0[np.sort(idx)]

        # only values within bounds
        x0_mask = np.array([(bnds[n, 0] <= y) & (y <= bnds[n, 1]) for n, y in enumerate(x0.T)]).T
        x0 = x0[np.all(x0_mask, axis=1)]

    args = [data, obj_lambda, df_caltrack_2_0, settings_dict, mp_meter]
    opt = Optimize(obj_fcn_dec, x0, bnds, opt_options, *args)
    res = opt.run()

    print("\n\n")
    print("X:  ", res.x)
    print("fun:", res.fun)