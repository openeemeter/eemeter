import os
from pathlib import Path

#auto load the eemeter module
import warnings
warnings.filterwarnings("ignore")
import rbfopt
from matplotlib import pyplot as plt
import numpy as np
import time
import pickle

from hourly_test_utils import *
from applied_data_science.bigquery.data import Meter_Data
from eemeter import eemeter as em
from eemeter.common.metrics import BaselineTestingMetrics as Metrics

import multiprocessing as mp

settings = em.HourlySettings(
    TRAIN_FEATURES=['ghi'],
    ALPHA=0.1,
    L1_RATIO=0.1,
    SEED=42
)

def obj_fcn(X, gradient=np.array([])):
    timer_start = time.time()

    # X = np.array(X)
    # X[0] = 10**X[0] # alpha comes as log(alpha)
    alpha, l1_ratio = X

    
    prf.settings.ALPHA = alpha
    prf.settings.L1_RATIO = l1_ratio
    
    prf.run()
    
    obj = 0
    count = 0
    for res in prf.results:
        _, _, errors = res
        if (errors!=None) and (errors['train'][0]!= np.inf) and (errors['test'][0]!= np.inf):
            count += 1
            obj += errors['test'][0]
    obj /= count
    print(f"alpha: {alpha}, l1_ratio: {l1_ratio}, obj: {obj}, count: {count}")
    print(f"Time taken: {time.time()-timer_start}")
    return float(obj)
    

# Optimization settings
opt_options = {
    "global": {"algorithm": "RBFOpt", 
               "stop_criteria_type": 'Iteration Maximum', 
               "stop_criteria_val": 2, 
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
subsamples = [1, 2, 3]
solar_meters = [True, False]
kwargs = {
    'settings': settings,
    'subsamples': subsamples,
    'solar_meters': solar_meters,
    'mp' : True,
    'max_id': -1
}
print('start loading data')
prf = Population_Run_Features(**kwargs)
prf._load_data()
print('data loaded')
x0 = [0.02, 0.003]
bnds = [(0.0001, 1), (0.0001, 1)]

bnds = np.array(bnds).T
n_dim = np.size(bnds[0])

var_type = ['R']*n_dim  
max_eval = 400
max_time = 1E30
rbfopt_initialize = True
bb = rbfopt.RbfoptUserBlackBox(n_dim, np.array(bnds[0]), np.array(bnds[1]),
                                np.array(var_type), obj_fcn)


bonmin_path = "/app/applied_data_science/tools/optimization/coin-or/bonmin"
ipopt_path = "/app/applied_data_science/tools/optimization/coin-or/ipopt"
rbfsettings = rbfopt.RbfoptSettings(max_iterations=max_eval,
                                    max_evaluations=max_eval,
                                    max_cycles=1E30,
                                    max_clock_time=max_time,
                                    minlp_solver_path=bonmin_path, 
                                    nlp_solver_path=ipopt_path,)
                                    
algo = rbfopt.RbfoptAlgorithm(rbfsettings, bb, init_node_pos=x0, do_init_strategy=rbfopt_initialize)

print('start optimization')
loss, x_opt, itercount, evalcount, fast_evalcount = algo.optimize()

# Save the results
save_path = "/app/.recurve_cache/mce_3_yr_precovid/Hyperparameter_opt/alltogetherV1"
if not os.path.exists(save_path):
    os.makedirs(save_path)
save_path = save_path + f"/subsample_all_solar_meter_all.pkl"
with open(save_path, 'wb') as f:
    pickle.dump([loss, x_opt, itercount, evalcount, fast_evalcount], f)
