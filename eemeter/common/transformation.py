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

import numpy as np

from eemeter.common.utils import (
    OoM,
    RoundToSigFigs,
)


C_base = 1/np.log(10)

class Bisymlog:
    def __init__(self, C=C_base, heuristic_scaling_factor=0.5, base=10, rescale_quantile=None):
        self.C = C
        self._heuristic_scaling_factor = heuristic_scaling_factor
        self._base = base
        self.rescale_quantile = rescale_quantile
        self.inv_rescale_fcn = None

        self.scaling_factor_bnds = [-1.0, 6.0] # Hardcoded, but not necessary to do so

        if rescale_quantile is not None and (rescale_quantile < 0.0 or rescale_quantile > 0.5):
            raise ValueError("Bisymlog 'rescale_quantile' must be 0 < x < 0.5")

    def set_C_heuristically(self, y, scaling_factor=None): # scaling factor: 0 looks loglike, 1 linear like
        if scaling_factor is None:
            scaling_factor = self._heuristic_scaling_factor
        else:
            self._heuristic_scaling_factor = scaling_factor

        min_y = y.min()
        max_y = y.max()

        if min_y == max_y:
            self.C = None
            return 1/np.log(1000)

        elif np.sign(max_y) != np.sign(min_y): # if zero is within total range, find largest pos or neg range
            processed_data = [y[y >= 0], y[y <= 0]]
            C = 0
            for data in processed_data:
                range = np.abs(data.max() - data.min())
                if range > C:
                    C = range
                    max_y = data.max()

        else:
            C = np.abs(max_y-min_y)

        s_fcn = lambda x: np.power(10, np.power(x, 2))
        s_fcn_range = s_fcn([0, 1])
        scaling_factor = s_fcn(self._heuristic_scaling_factor)

        s_bnds = self.scaling_factor_bnds

        s = (scaling_factor - s_fcn_range[0])/np.diff(s_fcn_range)*np.diff(s_bnds) + s_bnds[0]

        C *= 10**(OoM(max_y) + s[0])
        # TODO: round or not?
        # C = RoundToSigFigs(C, 1)    # round to 1 significant figure

        self.C = C

        return C

    def transform(self, y):
        if self.C is None:
            self.C = self.set_C_heuristically(y)

        if self.C is None:
            return y

        else:
            idx = np.isfinite(y)   # only perform transformation on finite values
            res = np.empty_like(y)
            res[~idx] = np.nan

            if self.rescale_quantile is None:
                res[idx] = np.sign(y[idx])*np.log10(np.abs(y[idx]/self.C) + 1)/np.log10(self._base)

            else:
                # get prior quantiles for rescaling
                pq = np.quantile(y[idx], [self.rescale_quantile, 1 - self.rescale_quantile])

                res[idx] = np.sign(y[idx])*np.log10(np.abs(y[idx]/self.C) + 1)/np.log10(self._base)

                # get current quantiles for rescaling and set rescaling functions
                cq = np.quantile(res[idx], [self.rescale_quantile, 1 - self.rescale_quantile])
                rescale_fcn = lambda x: (x - cq[0])/np.diff(cq)*np.diff(pq) + pq[0]
                self.inv_rescale_fcn = lambda x: (x - pq[0])/np.diff(pq)*np.diff(cq) + cq[0]

                # rescale to prior quantiles
                res[idx] = rescale_fcn(res[idx])

            return res

    def invTransform(self, y):
        if self.C is None:
            raise Exception('C is unspecified in Bisymlog')

        idx = np.isfinite(y)   # only perform transformation on finite values

        if self.inv_rescale_fcn is not None:
            y[idx] = self.inv_rescale_fcn(y[idx])

        res = np.empty_like(y)
        res[~idx] = np.nan
        res[idx] = np.sign(y[idx])*self.C*(np.power(self._base, np.abs(y[idx])) - 1)

        return res