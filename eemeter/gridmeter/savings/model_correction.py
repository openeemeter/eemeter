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

from eemeter.common.adaptive_loss import (
    weighted_quantile,
    IQR_outlier,
)




def unit_correction(oT, mT, oCG, mCG, settings):

    if settings.method is None:
        scale = 0
    
    elif settings.method == "ordinary_difference_in_differences":
        scale = 1

    elif settings.method == "percent_difference_in_differences":
        # simplified
        # savings = mT*oCG/mCG - oT 

        scale = mT/mCG

    elif settings.method == "absolute_percent_difference_in_differences":
        # simplified
        # savings = mT(1 - np.sign(mT)*np.sign(mCG) + oCG/mCG) - oT

        scale = np.abs(mT/mCG)

    correction = scale*(mCG - oCG)

    if settings.agg == "mean":
        correction_agg = np.mean(correction)

    elif settings.agg == "median":
        correction_agg = np.median(correction)

    # uncertainty

    return correction


def bisymlog(x, C=None):
    if C is None:
        C = 1/np.log(10)

    return np.sign(x)*(np.log10(1 + np.abs(x/C)))


def unit_cluster(oCG, mCG, settings):
    # how do we want to remove outliers?
    
    if settings.weight_by is None:
        weights = None
    else:
        weights = np.abs(mCG) / np.sum(np.abs(mCG))

    if settings.remove_outliers:
        # model_error = 1 - oCG/mCG # relative error
        # model_error = (oCG - mCG)/(np.abs(oCG) + np.abs(mCG)) # relative percent difference between -1 and 1
        model_error = (bisymlog(mCG) - bisymlog(oCG))*np.log(10) # log difference
        outlier_idx = remove_outliers(model_error, weights, settings)
        oCG = np.delete(oCG, outlier_idx)
        mCG = np.delete(mCG, outlier_idx)
        if weights is not None:
            weights = np.delete(weights, outlier_idx)

            # renormalize weights
            weights = weights / np.sum(weights)

    # TODO: the rest should be the model correction fcn

    if weights is None:
        if settings.agg == "mean":
            return np.mean(oCG), np.mean(mCG)
        else:
            return np.median(oCG), np.median(mCG)
        
    else:
        if settings.agg == "mean":
            return np.average(oCG, weights=weights), np.average(mCG, weights=weights)
        else:
            return weighted_quantile(oCG, 0.5, weights=weights), weighted_quantile(mCG, 0.5, weights=weights)




    

    
