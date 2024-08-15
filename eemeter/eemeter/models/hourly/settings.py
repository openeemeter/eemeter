from __future__ import annotations

import numpy as np
import pandas as pd

import pydantic

from enum import Enum
from typing import Any, Dict, Optional

from eemeter.common.base_settings import BaseSettings
from eemeter.common.metrics import BaselineMetrics
# from eemeter.common.const import CountryCode


class SelectionChoice(str, Enum):
    CYCLIC = "cyclic"
    RANDOM = "random"


class BinningChoice(str, Enum):
    EQUAL_BIN_WIDTH = "equal_bin_width"
    EQUAL_SAMPLE_COUNT = "equal_sample_count"


class ClusteringMetric(str, Enum):
    EUCLIDEAN = "euclidean"
    DTW = "dtw"
    SOFTDTW = "softdtw"


# analytic_features = ['GHI', 'Temperature', 'DHI', 'DNI', 'Relative Humidity', 'Wind Speed', 'Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI', 'Cloud Type']
class HourlySettings(BaseSettings):
    class Config:
        frozen = False # freeze the settings

    """train features used within the model"""
    TRAIN_FEATURES: list[str] = pydantic.Field(
        default=['temperature'], 
        validate_default=True,
    )

    """minimum number of training hours per day below which a day is excluded"""
    MIN_DAILY_TRAINING_HOURS: int = pydantic.Field(
        default=12,
        ge=0,
        le=24,
        validate_default=True,
    )

    """include temperature bins"""
    INCLUDE_TEMPERATURE_BINS: bool = pydantic.Field(
        default=True,
        validate_default=True,
    )

    """how to bin temperature data"""
    TEMPERATURE_BINNING_METHOD: BinningChoice | None = pydantic.Field(
        default=BinningChoice.EQUAL_BIN_WIDTH,
        validate_default=True,
    )

    """number of temperature bins"""
    TEMPERATURE_BIN_COUNT: int | None = pydantic.Field(
        default=6,
        ge=1,                       
        validate_default=True,
    )

    """number of clusters to use for temporal clustering (day, month)"""
    MAX_TEMPORAL_CLUSTER_COUNT: int | None = pydantic.Field(
        default=6,
        ge=2,
        validate_default=True,
    )

    """metric to use for temporal clustering"""
    TEMPORAL_CLUSTER_METRIC: ClusteringMetric = pydantic.Field(
        default=ClusteringMetric.EUCLIDEAN,
        validate_default=True,
    )

    """number of times to run k-means clustering"""
    TEMPORAL_CLUSTER_N_INIT: int = pydantic.Field(
        default=5,
        ge=1,
        validate_default=True,
    )

    """supplemental data"""
    SUPPLEMENTAL_DATA: dict | None = pydantic.Field(
        default=None,
        validate_default=True,
    )

    """ElasticNet alpha parameter"""
    ALPHA: float = pydantic.Field(
        default=0.012896,
        ge=0,
        validate_default=True,
    )

    """ElasticNet l1_ratio parameter"""
    L1_RATIO: float = pydantic.Field(
        default=0.032904,
        ge=0,
        le=1,
        validate_default=True,
    )

    """ElasticNet selection parameter"""
    SELECTION: SelectionChoice = pydantic.Field(
        default=SelectionChoice.CYCLIC,
        validate_default=True,
    )

    """ElasticNet max_iter parameter"""
    MAX_ITER: int = pydantic.Field(
        default=1000,
        ge=1,
        le=2**32 - 1,
        validate_default=True,
    )

    """seed for any random state assignment (ElasticNet, Clustering)"""
    SEED: int | None = pydantic.Field(
        default=None,
        ge=0,
        validate_default=True,
    )

    @pydantic.model_validator(mode="after")
    def _lowercase_features(self):
        self.TRAIN_FEATURES = [s.lower() for s in self.TRAIN_FEATURES]

        return self
    

    @pydantic.model_validator(mode="after")
    def _check_temperature_bins(self):
        # check that temperature binning method is set based on include temperature bins
        if self.INCLUDE_TEMPERATURE_BINS:
            if self.TEMPERATURE_BINNING_METHOD is None:
                raise ValueError("'TEMPERATURE_BINNING_METHOD' must be specified if 'INCLUDE_TEMPERATURE_BINS' is True.")
        else:
            if self.TEMPERATURE_BINNING_METHOD is not None:
                raise ValueError("'TEMPERATURE_BINNING_METHOD' must be None if 'INCLUDE_TEMPERATURE_BINS' is False.")

        # check that temperature bin count is set based on binning method
        if self.TEMPERATURE_BINNING_METHOD is None:
            if self.TEMPERATURE_BIN_COUNT is not None:
                raise ValueError("'TEMPERATURE_BIN_COUNT' must be None if 'TEMPERATURE_BINNING_METHOD' is None.")
        else:
            if self.TEMPERATURE_BIN_COUNT is None:
                raise ValueError("'TEMPERATURE_BIN_COUNT' must be specified if 'TEMPERATURE_BINNING_METHOD' is not None.")

        return self


    @pydantic.model_validator(mode="after")
    def _check_features(self):
        if "temperature" not in self.TRAIN_FEATURES:
            self.TRAIN_FEATURES.insert(0, "temperature")

        return self

    @pydantic.model_validator(mode="after")
    def _check_seed(self):
        if self.SEED is None:
            self._SEED = np.random.randint(0, 2**32 - 1)
        else:
            self._SEED = self.SEED
        
        return self

    # @pydantic.model_validator(mode="after")
    # def _freeze_settings(self):
    #     self.model_config["frozen"] = True

    #     return self


class HourlySolarSettings(HourlySettings):
    """train features used within the model"""
    TRAIN_FEATURES: list[str] = pydantic.Field(
        default=['temperature', 'ghi'], 
        validate_default=True,
    )


class HourlyNonSolarSettings(HourlySettings):
    """how to bin temperature data"""
    TEMPERATURE_BINNING_METHOD: BinningChoice | None = pydantic.Field(
        default=BinningChoice.EQUAL_BIN_WIDTH,
        validate_default=True,
    )

    """number of temperature bins"""
    TEMPERATURE_BIN_COUNT: int | None = pydantic.Field(
        default=10,
        ge=1,                       
        validate_default=True,
    )

    """ElasticNet alpha parameter"""
    ALPHA: float = pydantic.Field(
        default=0.002825,
        ge=0,
        validate_default=True,
    )

    """ElasticNet l1_ratio parameter"""
    L1_RATIO: float = pydantic.Field(
        default=0.983888,
        ge=0,
        le=1,
        validate_default=True,
    )


class SerializeModel(BaseSettings):
    class Config:
        arbitrary_types_allowed = True

    SETTINGS: HourlySettings | None = None
    TEMPORAL_CLUSTERS: list[list[int]] | None = None
    TEMPERATURE_BIN_EDGES: list[float] | None = None
    TS_FEATURES: list[str] | None = None
    CATEGORICAL_FEATURES: list[str] | None = None
    FEATURE_SCALER: Dict[str, list[float]] | None = None
    CATAGORICAL_SCALER: Dict[str, list[float]] | None = None
    Y_SCALER: list[float] | None = None
    COEFFICIENTS: list[list[float]] | None = None
    INTERCEPT: list[float] | None = None
    BASELINE_METRICS: BaselineMetrics | None = None