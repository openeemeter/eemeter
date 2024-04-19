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
import pandas as pd

import pydantic
from typing import Optional
from enum import Enum

from functools import cached_property # TODO: This requires Python 3.8

from eemeter.common.utils import median_absolute_deviation, t_stat
from eemeter.common.pydantic_utils import PydanticDf, PydanticFromDict


def computed_field_cached_property():
    decs = [pydantic.computed_field, cached_property]
    def deco(f):
        for dec in reversed(decs):
            f = dec(f)
        return f
    return deco


class ColumnMetrics(pydantic.BaseModel):
    class Config:
        arbitrary_types_allowed = True # required for dataframe / series

    series: pd.Series = pydantic.Field(
        exclude=True
    )

    @computed_field_cached_property()
    def sum(self) -> float:
        return self.series.sum()

    @computed_field_cached_property()
    def mean(self) -> float:
        return self.sum / len(self.series)
    
    @computed_field_cached_property()
    def variance(self) -> float:
        return self.series.var(ddof=0)

    @computed_field_cached_property()
    def std(self) -> float:
        return self.variance ** 0.5

    @computed_field_cached_property()
    def cvstd(self) -> float:
        return self.std / self.mean
    
    @computed_field_cached_property()
    def sum_squared(self) -> float:
        return (self.series ** 2).sum()

    @computed_field_cached_property()
    def median(self) -> float:
        return self.series.median()
    
    @computed_field_cached_property()
    def MAD_scaled(self) -> float:
        return median_absolute_deviation(self.series, self.median)

    @computed_field_cached_property()
    def iqr(self) -> float:
        return np.diff(np.quantile(self.series, [0.25, 0.75]))[0]
    
    @computed_field_cached_property()
    def skew(self) -> float:
        return self.series.skew()
    
    @computed_field_cached_property()
    def kurtosis(self) -> float:
        return self.series.kurtosis()


def _safe_divide(numerator, denominator, min_denominator=1E-3):
    if denominator <= min_denominator and numerator > 10*min_denominator:
        return None
    
    return numerator / denominator


class BaselineMetrics(pydantic.BaseModel):
    # TODO: Update the doc string
    """Contains measures of model fit and summary statistics on the input dataframe.

    Parameters
    ----------
    observed_input : :any:`pandas.Series`
        Series with :any:`pandas.DatetimeIndex` with a set of electricity or
        gas meter values.
    predicted_input : :any:`pandas.Series`
        Series with :any:`pandas.DatetimeIndex` with a set of electricity or
        gas meter values.
    num_parameters : :any:`int`, optional
        The number of parameters (excluding the intercept) used in the
        regression from which the predictions were derived.
    autocorr_lags : :any:`int`, optional
        The number of lags to use when calculating the autocorrelation of the
        residuals.
    confidence_level : :any:`int`, optional
        Confidence level used in fractional savings uncertainty computations.

    Attributes
    ----------
    observed_length : :any:`int`
        The length of the observed.
    predicted_length : :any:`int`
        The length of the predicted.
    merged_length : :any:`int`
        The length of the dataframe resulting from the inner join of the
        observed and the predicted.
    observed_mean : :any:`float`
        The mean of the observed.
    predicted_mean : :any:`float`
        The mean of the predicted.
    observed_skew : :any:`float`
        The skew of the observed.
    predicted_skew : :any:`float`
        The skew of the predicted.
    observed_kurtosis : :any:`float`
        The excess kurtosis of the observed.
    predicted_kurtosis : :any:`float`
        The excess kurtosis of the predicted.
    observed_cvstd : :any:`float`
        The coefficient of standard deviation of the observed.
    predicted_cvstd : :any:`float`
        The coefficient of standard deviation of the predicted.
    r_squared : :any:`float`
        The r-squared of the model from which the predicted was produced.
    r_squared_adj : :any:`float`
        The r-squared of the predicted relative to the
        observed, adjusted by the number of parameters in the model.
    cvrmse : :any:`float`
        The coefficient of variation (root-mean-squared error) of the
        predicted relative to the observed.
    cvrmse_adj : :any:`float`
        The coefficient of variation (root-mean-squared error) of the
        predicted relative to the observed, adjusted
        by the number of parameters in the model.
    mape : :any:`float`
        The mean absolute percent error of the predicted relative
        to the observed.
    mape_no_zeros : :any:`float`
        The mean absolute percent error of the predicted relative
        to the observed, with all time periods dropped where the
        observed was not greater than zero.
    num_meter_zeros : :any:`int`
        The number of time periods for which the observed was not
        greater than zero.
    nmae : :any:`float`
        The normalized mean absolute error of the predicted
        relative to the observed.
    nmbe : :any:`float`
        The normalized mean bias error of the predicted relative
        to the observed.
    autocorr_resid : :any:`float`
        The autocorrelation of the residuals (where the residuals equal the
        predicted minus the observed), measured
        using a number of lags equal to autocorr_lags.
    n_prime: :any:`float`
        The number of baseline inputs corrected for autocorrelation -- used
        in fractional savings uncertainty computation.
    single_tailed_confidence_level: :any:`float`
        The adjusted confidence level for use in single-sided tests.
    degrees_of_freedom: :any:`float
        Maxmimum number of independent variables which have the freedom to vary
    t_stat: :any:`float
        t-statistic, used for hypothesis testing
    cvrmse_auto_corr_correction: :any:`float
        Correctoin factor the apply to cvrmse to account for autocorrelation of inputs.
    approx_factor_auto_corr_correction: :any:`float
        Approximation factor used in ashrae 14 guideline for uncertainty computation.
    fsu_base_term: :any:`float
        Base term used in fractional savings uncertainty computation.

    """

    class Config:
        arbitrary_types_allowed = True # required for dataframe / series

    """Input dataframe to be used for metrics calculations"""
    df: pd.DataFrame = pydantic.Field(
        exclude=True
    )

    _min_denominator: float = 1E-3

    """Number of model parameters"""
    num_model_params: int = pydantic.Field(
        ge=1,
        validate_default=True,
    )

    @cached_property
    def _df(self) -> pd.DataFrame:
        _df = self.df[["observed", "predicted"]].copy()

        if len(_df) < 1:
            raise ValueError("Input dataframe must have at least one row")

        # Check dataframe
        expected_columns = {"observed": "float", "predicted": "float"}
        _df = PydanticDf(df=_df, column_types=expected_columns).df

        # drop non finite values from df
        _df = _df[np.isfinite(_df["observed"]) & np.isfinite(_df["predicted"])]

        # get residuals
        _df["residuals"] = _df["observed"] - _df["predicted"]

        return _df
    
    @computed_field_cached_property()
    def n(self) -> float:
        return len(self._df)
        
    @computed_field_cached_property()
    def n_prime(self) -> float:
        # lag should be 1 according to https://www.osti.gov/servlets/purl/1366449
        autocorr = self._df["residuals"].autocorr(lag=1)

        _n_prime = float(self.n * (1 - autocorr) / (1 + autocorr))

        if not np.isfinite(_n_prime):
            # TODO: Create warning
            _n_prime = 1

        return _n_prime
    
    @computed_field_cached_property()
    def ddof(self) -> float:
        _ddof = self.n - self.num_model_params

        if _ddof < 1:
            # TODO: Create warning
            _ddof = 1

        return _ddof
    
    @computed_field_cached_property()
    def ddof_autocorr(self) -> float:
        # TODO: should this be rounded?
        _ddof_autocorr = self.n_prime - self.num_model_params

        # TODO: what to do if less than 1?
        if _ddof_autocorr < 1:
            # TODO: Create warning
            _ddof_autocorr = 1

        return _ddof_autocorr

    @computed_field_cached_property()
    def observed(self) -> ColumnMetrics:
        return ColumnMetrics(series=self._df["observed"])
    
    @computed_field_cached_property()
    def predicted(self) -> ColumnMetrics:
        return ColumnMetrics(series=self._df["predicted"])
    
    @computed_field_cached_property()
    def residuals(self) -> ColumnMetrics:
        return ColumnMetrics(series=self._df["residuals"])
    
    @computed_field_cached_property()
    def mae(self) -> float:
        return self._df["residuals"].abs().mean()

    @computed_field_cached_property()
    def nmae(self) -> Optional[float]:
        return _safe_divide(self.mae, self.observed.mean, self._min_denominator)
    
    @computed_field_cached_property()
    def pnmae(self) -> Optional[float]:
        return _safe_divide(self.mae, self.observed.iqr, self._min_denominator)
    
    @computed_field_cached_property()
    def mbe(self) -> float:
        return self.residuals.mean

    @computed_field_cached_property()
    def nmbe(self) -> Optional[float]:
        return _safe_divide(self.mbe, self.observed.mean, self._min_denominator)
    
    @computed_field_cached_property()
    def pnmbe(self) -> Optional[float]:
        return _safe_divide(self.mbe, self.observed.iqr, self._min_denominator)
        
    @computed_field_cached_property()
    def sse(self) -> float:
        return self.residuals.sum_squared

    @computed_field_cached_property()
    def mse(self) -> float:
        return self.sse / self.n

    @computed_field_cached_property()
    def rmse(self) -> float:
        return self.mse ** 0.5

    @computed_field_cached_property()
    def rmse_adj(self) -> float:       
        return (self.sse / self.ddof) ** 0.5
    
    @computed_field_cached_property()
    def rmse_autocorr_adj(self) -> float:
        return (self.sse / self.ddof_autocorr) ** 0.5

    @computed_field_cached_property()
    def cvrmse(self) -> Optional[float]:
        return _safe_divide(self.rmse, self.observed.mean, self._min_denominator)

    @computed_field_cached_property()
    def cvrmse_adj(self) -> Optional[float]:
        return _safe_divide(self.rmse_adj, self.observed.mean, self._min_denominator)

    @computed_field_cached_property()
    def cvrmse_autocorr_adj(self) -> Optional[float]:
        return _safe_divide(self.rmse_autocorr_adj, self.observed.mean, self._min_denominator)

    @computed_field_cached_property()
    def pnrmse(self) -> Optional[float]:
        return _safe_divide(self.rmse, self.observed.iqr, self._min_denominator)
    
    @computed_field_cached_property()
    def pnrmse_adj(self) -> Optional[float]:
        return _safe_divide(self.rmse_adj, self.observed.iqr, self._min_denominator)
    
    @computed_field_cached_property()
    def pnrmse_autocorr_adj(self) -> Optional[float]:
        return _safe_divide(self.rmse_autocorr_adj, self.observed.iqr, self._min_denominator)

    @computed_field_cached_property()
    def r_squared(self) -> float:
        return self._df[["predicted", "observed"]].corr().iloc[0, 1] ** 2

    @computed_field_cached_property()
    def r_squared_adj(self) -> Optional[float]:       
        n = self.n
        n_adj = self.ddof

        num = (1 - self.r_squared) * (n - 1)
        den =(n_adj - 1)

        return 1 - _safe_divide(num, den, self._min_denominator)
    
    @computed_field_cached_property()
    def mape(self) -> Optional[float]:
        df = self._df
        df_no_zeros = df[np.abs(df["observed"]) >= self._min_denominator]

        if len(df_no_zeros) == 0:
            return None

        return (df_no_zeros["residuals"] / df_no_zeros["observed"]).abs().mean()
    

# Delete these later
class ColumnTestingMetrics(ColumnMetrics):
    @cached_property
    def values(self) -> np.ndarray:
        return self.series.values


class BaselineTestingMetrics(BaselineMetrics):
    @cached_property
    def _df(self) -> pd.DataFrame:
        #TODO: remove any caltrack value for the actual implementation, this is for testing
        _df = self.df[["observed", "predicted", "caltrack"]].copy()

        if len(_df) < 1:
            raise ValueError("Input dataframe must have at least one row")

        # Check dataframe
        expected_columns = {"observed": "float", "predicted": "float",  "caltrack": "float"}
        _df = PydanticDf(df=_df, column_types=expected_columns).df

        # drop non finite values from df
        _df = _df[np.isfinite(_df["observed"]) & np.isfinite(_df["predicted"]) & np.isfinite(_df["caltrack"])]

        # get residuals
        _df["residuals"] = _df["observed"] - _df["predicted"]

        return _df

    @computed_field_cached_property()
    def observed(self) -> ColumnTestingMetrics:
        return ColumnTestingMetrics(series=self._df["observed"])
    
    @computed_field_cached_property()
    def predicted(self) -> ColumnTestingMetrics:
        return ColumnTestingMetrics(series=self._df["predicted"])

    @computed_field_cached_property()
    def caltrack(self) -> ColumnTestingMetrics:
        return ColumnTestingMetrics(series=self._df["caltrack"])


def BaselineMetricsFromDict(input_dict):
    for k in ["observed", "predicted", "residuals"]:
        input_dict[k] = PydanticFromDict(input_dict[k], name="ColumnMetrics")

    return PydanticFromDict(input_dict, name="BaselineMetrics")


class ModelChoice(str, Enum):
    HOURLY = "hourly"
    HOURLYSOLAR = "hourly"
    DAILY = "daily"
    BILLING = "billing"


class ReportingMetrics(pydantic.BaseModel):
    class Config:
        arbitrary_types_allowed = True # required for dataframe / series

    baseline_metrics: BaselineMetrics = pydantic.Field(
        exclude=True
    )

    """Reporting dataframe to be used for metrics calculations"""
    reporting_df: pd.DataFrame = pydantic.Field(
        exclude=True
    )

    """Data frequency of the model for use in uncertainty calculations"""
    data_frequency: ModelChoice = pydantic.Field(
        exclude=False
    )

    """Confidence level used in uncertainty calculations"""
    confidence_level: float = pydantic.Field(
        ge=0.0,
        le=1.0,
        default=0.90,
        validate_default=True,
    )

    """Number of tails to use in uncertainty calculations"""
    t_tail: int = pydantic.Field(
        ge=1,
        le=2,
        default=2, # ASHRAE 14 uses 2 tail
        validate_default=True,
    )

    @property
    def _baseline(self) -> BaselineMetrics:
        return self.baseline_metrics

    @cached_property
    def _df(self) -> pd.DataFrame:
        _df = self.reporting_df[["observed", "predicted"]].copy()

        if len(_df) < 1:
            raise ValueError("Input dataframe must have at least one row")

        # Check dataframe
        expected_columns = {"observed": "float", "predicted": "float"}
        _df = PydanticDf(df=_df, column_types=expected_columns).df

        # drop non finite values from df
        _df = _df[np.isfinite(_df["observed"]) & np.isfinite(_df["predicted"])]

        # # get residuals
        # _df["residuals"] = _df["observed"] - _df["predicted"]

        return _df
    
    @computed_field_cached_property()
    def n(self) -> float:
        return len(self._df)

    @computed_field_cached_property()
    def observed_sum(self) -> float:
        return self._df["observed"].sum()
    
    @computed_field_cached_property()
    def predicted_sum(self) -> float:
        return self._df["predicted"].sum()
    
    @computed_field_cached_property()
    def t_stat(self) -> float:
        return t_stat(1 - self.confidence_level, self._baseline.ddof, tail=self.t_tail)
    
    @computed_field_cached_property()
    def savings(self) -> float:
        return self.predicted_sum - self.observed_sum

    @computed_field_cached_property()
    def total_savings_uncertainty(self) -> float:
        E_reporting = self.predicted_sum
        n = self._baseline.n
        n_prime = self._baseline.n_prime
        m = self.n
        t = self.t_stat
        cvrmse_autocorr_adj = self._baseline.cvrmse_autocorr_adj

        # part of approximation factor used in ashrae 14 guideline
        approx_factor = np.sqrt(n/(m*n_prime)*(1 + (2/n_prime)))

        s_unc_base = E_reporting*(t*cvrmse_autocorr_adj*approx_factor)

        if self.data_frequency == "hourly":
            s_unc = 1.26*s_unc_base

        elif self.data_frequency in ["daily", "billing"]:
            M = len(self._df.index.month.unique())

            if self.data_frequency == "daily":
                coefs = [-0.00024, 0.03535, 1.00286]
            else:
                coefs = [-0.00022, 0.03306, 0.94054]

            s_unc = np.polyval(coefs, M)*s_unc_base

        else:
            raise ValueError("model_type must be 'hourly', 'daily', or 'billing'")
        
        return s_unc

    @computed_field_cached_property()
    def fsu(self) -> float:
        return self.total_savings_uncertainty / self.savings
    
    @computed_field_cached_property()
    def predicted_data_point_unc(self) -> float:
        return self.total_savings_uncertainty / np.sqrt(self.n)