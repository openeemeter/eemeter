#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2014-2024 OpenEEmeter contributors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""
from timeit import default_timer as timer

import nlopt
import numpy as np
from scipy.optimize import minimize as scipy_minimize
from scipy.optimize import minimize_scalar as scipy_minimize_scalar

from eemeter.eemeter.models.daily.optimize_results import OptimizedResult

nlopt_algorithms = {
    "nlopt_DIRECT": nlopt.GN_DIRECT,
    "nlopt_DIRECT_NOSCAL": nlopt.GN_DIRECT_NOSCAL,
    "nlopt_DIRECT_L": nlopt.GN_DIRECT_L,
    "nlopt_DIRECT_L_RAND": nlopt.GN_DIRECT_L_RAND,
    "nlopt_DIRECT_L_NOSCAL": nlopt.GN_DIRECT_L_NOSCAL,
    "nlopt_DIRECT_L_RAND_NOSCAL": nlopt.GN_DIRECT_L_RAND_NOSCAL,
    "nlopt_ORIG_DIRECT": nlopt.GN_ORIG_DIRECT,
    "nlopt_ORIG_DIRECT_L": nlopt.GN_ORIG_DIRECT_L,
    "nlopt_CRS2_LM": nlopt.GN_CRS2_LM,
    "nlopt_MLSL_LDS": nlopt.G_MLSL_LDS,
    "nlopt_MLSL": nlopt.G_MLSL,
    "nlopt_STOGO": nlopt.GD_STOGO,
    "nlopt_STOGO_RAND": nlopt.GD_STOGO_RAND,
    "nlopt_AGS": nlopt.GN_AGS,
    "nlopt_ISRES": nlopt.GN_ISRES,
    "nlopt_ESCH": nlopt.GN_ESCH,
    "nlopt_COBYLA": nlopt.LN_COBYLA,
    "nlopt_BOBYQA": nlopt.LN_BOBYQA,
    "nlopt_NEWUOA": nlopt.LN_NEWUOA,
    "nlopt_NEWUOA_BOUND": nlopt.LN_NEWUOA_BOUND,
    "nlopt_PRAXIS": nlopt.LN_PRAXIS,
    "nlopt_NELDERMEAD": nlopt.LN_NELDERMEAD,
    "nlopt_SBPLX": nlopt.LN_SBPLX,
    "nlopt_MMA": nlopt.LD_MMA,
    "nlopt_CCSAQ": nlopt.LD_CCSAQ,
    "nlopt_SLSQP": nlopt.LD_SLSQP,
    "nlopt_LBFGS": nlopt.LD_LBFGS,
    "nlopt_TNEWTON": nlopt.LD_TNEWTON,
    "nlopt_TNEWTON_PRECOND": nlopt.LD_TNEWTON_PRECOND,
    "nlopt_TNEWTON_RESTART": nlopt.LD_TNEWTON_RESTART,
    "nlopt_TNEWTON_PRECOND_RESTART": nlopt.LD_TNEWTON_PRECOND_RESTART,
    "nlopt_VAR1": nlopt.LD_VAR1,
    "nlopt_VAR2": nlopt.LD_VAR2,
}

nlopt_algorithms = {k.lower(): v for k, v in nlopt_algorithms.items()}

pos_msg = [
    "Optimization terminated successfully.",
    "Optimization terminated: Stop Value was reached.",
    "Optimization terminated: Function tolerance was reached.",
    "Optimization terminated: X tolerance was reached.",
    "Optimization terminated: Max number of evaluations was reached.",
    "Optimization terminated: Max time was reached.",
]
neg_msg = [
    "Optimization failed",
    "Optimization failed: Invalid arguments given",
    "Optimization failed: Out of memory",
    "Optimization failed: Roundoff errors limited progress",
    "Optimization failed: Forced termination",
]


def obj_fcn_dec(obj_fcn, x0, bnds):
    """
    Returns a function that evaluates the objective function with the given bounds.

    Args:
    - obj_fcn: the objective function to be evaluated
    - x0: the initial guess for the optimization
    - bnds: the bounds for the optimization

    Returns:
    - obj_fcn_eval: a function that evaluates the objective function with the given bounds
    - idx_opt: the indices of the variables with non-equal bounds
    """

    idx_opt = [n for n in range(np.shape(bnds)[0]) if (bnds[n, 0] < bnds[n, 1])]

    def obj_fcn_eval(
        x, *args, **kwargs
    ):  # only modify x0 where it has bounds which are not the same
        x0[idx_opt] = x

        return obj_fcn(x0, *args, **kwargs)

    return obj_fcn_eval, idx_opt


class Optimizer:
    # opt_options = {"global": {"algorithm": "scipy_COBYLA",
    #                           "stop_criteria_type": 'Iteration Maximum',
    #                           "stop_criteria_val": 2000,
    #                           "initial_step": 0.1 # percentage},
    #                           "xtol_rel": 1E-5
    #                           "ftol_rel": 1E-5
    #                           "initial_pop_multiplier": 2
    #                "local": {} # same}
    """
    This class is used to perform optimization on a given objective function using either the SciPy or NLopt library.
    The optimization can be performed globally or locally based on the options provided.

    Attributes:
        bnds (np.array): The bounds for the optimization.
        x0 (np.array): The initial guess for the optimization.
        obj_fcn (function): The objective function to be optimized.
        idx_opt (int): The index of the optimal solution.
        coef_id (str): The identifier for the coefficient.
        settings (dict): The settings for the optimization.
        opt_options (dict): The options for the optimization.
    """

    def __init__(self, obj_fcn, x0, bnds, coef_id, settings, opt_options):
        """
        The constructor for the Optimizer class.

        Parameters:
            obj_fcn (function): The objective function to be optimized.
            x0 (np.array): The initial guess for the optimization.
            bnds (list): The bounds for the optimization.
            coef_id (str): The identifier for the coefficient.
            settings (dict): The settings for the optimization.
            opt_options (dict): The options for the optimization.
        """
        self.bnds = np.array(bnds)
        self.x0 = np.clip(
            x0, bnds[:, 0], bnds[:, 1]
        )  # clip x0 to the bnds, just in case

        self.obj_fcn, self.idx_opt = obj_fcn_dec(obj_fcn, x0, bnds)

        self.coef_id = coef_id

        self.settings = settings
        self.opt_options = opt_options

    def run(self):
        """
        This method runs the optimization process.

        Returns:
            OptimizedResult: An object containing the results of the optimization.
        """
        bnds = self.bnds

        res = {}
        for opt_type in ["global", "local"]:
            options = self.opt_options[opt_type]

            if len(options) == 0:
                continue

            if len(res) == 0:
                x0 = self.x0
            else:
                x0 = res[list(res.keys())[-1]].x

            if options["algorithm"][:5] == "scipy":
                res[opt_type] = self.scipy(x0, bnds, options)
            elif options["algorithm"][:5] == "nlopt":
                res[opt_type] = self.nlopt(x0, bnds, options)

            if (
                options["algorithm"] == "nlopt_MLSL_LDS"
            ):  # if using multistart algorithm, break upon finishing loop
                break

        return res[list(res.keys())[-1]]

    def scipy(self, x0, bnds, options):
        """
        Optimize the objective function using the SciPy library. Different optimization options are available,
        such as scipy_COBYLA, scipy_SLSQP, scipy_L_BFGS_B, scipy_TNC, scipy_BFGS, scipy_Powell, scipy_Nelder-Mead.
        options argument needs to have the algorithm specified.

        Args:
            x0 (list): Initial guess for the optimization.
            bnds (tuple): Bounds for the optimization.
            options (dict): Options for the optimization.

        Returns:
            res_out (OptimizedResult): An object containing the results of the optimization.
        """

        timer_start = timer()

        algorithm = options["algorithm"][6:]

        if algorithm.lower() in ["brent", "golden", "bounded"]:
            scipy_obj_fcn = lambda x: self.obj_fcn([x])

            if algorithm.lower() in ["brent", "golden"]:
                res = scipy_minimize_scalar(
                    scipy_obj_fcn, bracket=bnds, method=algorithm.lower()
                )

            elif algorithm.lower() == "bounded":
                res = scipy_minimize_scalar(
                    scipy_obj_fcn, bounds=bnds[0], method="bounded"
                )

            res.x = [res.x]

        else:
            x0_opt = x0[self.idx_opt]
            bnds_opt = bnds[self.idx_opt, :]

            scipy_obj_fcn = lambda x: self.obj_fcn(x)

            res = scipy_minimize(
                scipy_obj_fcn, x0_opt, method=algorithm, bounds=bnds_opt
            )

        x = res.x
        x, mean_loss, TSS, T, model, weight, resid, jac, alpha, C = self.obj_fcn(
            x, optimize_flag=False
        )
        success = res.success
        message = res.message
        nfev = res.nfev
        time_elapsed = timer() - timer_start

        res_out = OptimizedResult(
            x,
            bnds,
            self.coef_id,
            alpha,
            C,
            T,
            model,
            weight,
            resid,
            jac,
            mean_loss,
            TSS,
            success,
            message,
            nfev,
            time_elapsed,
            self.settings,
        )
        # res_out.jac = res.jac
        # res_out.hess = res.hess
        # res_out.hess_inv = res.hess_inv
        # res_out.njev = res.njev
        # res_out.nhev = res.nhev

        return res_out

    def nlopt(self, x0, bnds, options):
        """
        Optimize the objective function using the NLopt library.

        Args:
            x0 (ndarray): Initial guess for the optimization.
            bnds (ndarray): Bounds on the variables.
            options (dict): Dictionary of options for the optimization.

        Returns:
            res_out (OptimizedResult): Object containing the results of the optimization.
        """
        timer_start = timer()

        obj_fcn = self.obj_fcn
        idx_opt = self.idx_opt

        x0_opt = x0[idx_opt]
        bnds_opt = bnds[idx_opt, :].T
        coef_id_opt = [id for n, id in enumerate(self.coef_id) if n in idx_opt]

        algorithm = nlopt_algorithms[options["algorithm"]]

        opt = nlopt.opt(algorithm, np.size(x0_opt))
        opt.set_min_objective(obj_fcn)
        if options["stop_criteria_type"] == "Iteration Maximum":
            opt.set_maxeval(int(options["stop_criteria_val"]) - 1)
        elif options["stop_criteria_type"] == "Maximum Time [min]":
            opt.set_maxtime(options["stop_criteria_val"] * 60)

        opt.set_xtol_rel(options["xtol_rel"])
        opt.set_ftol_rel(options["ftol_rel"])
        opt.set_lower_bounds(bnds_opt[0])
        opt.set_upper_bounds(bnds_opt[1])

        # initial_step
        max_initial_step = np.max(np.abs(bnds_opt - x0_opt), axis=0)

        initial_step = (bnds_opt[1] - bnds_opt[0]) * options["initial_step"]

        for n, coef_name in enumerate(coef_id_opt):
            if "dd_bp" in coef_name:
                initial_step[n] *= 2

            if coef_name == "hdd_bp":
                initial_step[n] *= -1

        initial_step = np.clip(initial_step, -max_initial_step, max_initial_step)

        x1 = x0_opt + initial_step
        np.putmask(
            initial_step, (x1 < bnds_opt[0]) | (x1 > bnds_opt[1]), -initial_step
        )  # first step in direction of more variable space

        opt.set_initial_step(initial_step)

        # alter default size of population in relevant algorithms
        if options["algorithm"] == "nlopt_CRS2_LM":
            default_pop_size = 10 * (len(x0_opt) + 1)
        elif options["algorithm"] in ["nlopt_MLSL_LDS", "nlopt_MLSL"]:
            default_pop_size = 4
        elif options["algorithm"] == "nlopt_ISRES":
            default_pop_size = 20 * (len(x0_opt) + 1)

            opt.set_population(
                int(np.rint(default_pop_size * options["initial_pop_multiplier"]))
            )

        if (
            options["algorithm"] == "nlopt_MLSL_LDS"
        ):  # if using multistart algorithm as global, set subopt
            local_algorithm = nlopt_algorithms[self.opt_options["local"]["algorithm"]]
            sub_opt = nlopt.opt(local_algorithm, np.size(x0_opt))
            sub_opt.set_initial_step(initial_step)
            sub_opt.set_xtol_rel(options["xtol_rel"])
            sub_opt.set_ftol_rel(options["ftol_rel"])
            opt.set_local_optimizer(sub_opt)

        x_opt = opt.optimize(x0_opt)  # optimize!

        if nlopt.SUCCESS > 0:
            success = True
            msg = pos_msg[nlopt.SUCCESS - 1]
        else:
            success = False
            msg = neg_msg[nlopt.SUCCESS - 1]

        x = x_opt
        x, mean_loss, TSS, T, model, weight, resid, jac, alpha, C = obj_fcn(
            x, optimize_flag=False
        )
        success = success
        message = msg
        nfev = opt.get_numevals()
        time_elapsed = timer() - timer_start

        res_out = OptimizedResult(
            x,
            bnds,
            self.coef_id,
            alpha,
            C,
            T,
            model,
            weight,
            resid,
            jac,
            mean_loss,
            TSS,
            success,
            message,
            nfev,
            time_elapsed,
            self.settings,
        )

        return res_out
