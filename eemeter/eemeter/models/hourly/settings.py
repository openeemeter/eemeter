from __future__ import annotations

import numpy as np
import pandas as pd

import pydantic

from enum import Enum
from typing import Optional

from eemeter.common.base_settings import BaseSettings
from eemeter.common.metrics import BaselineMetrics

# from eemeter.common.const import CountryCode


class SelectionChoice(str, Enum):
    CYCLIC = "cyclic"
    RANDOM = "random"


class BinningChoice(str, Enum):
    EQUAL_SAMPLE_COUNT = "equal_sample_count"
    EQUAL_BIN_WIDTH = "equal_bin_width"
    SET_BIN_WIDTH = "set_bin_width"


class ClusteringMetric(str, Enum):
    EUCLIDEAN = "euclidean"
    DTW = "dtw"
    SOFTDTW = "softdtw"


class BaseElasticNetSettings(BaseSettings):
    """ElasticNet alpha parameter"""
    ALPHA: float = pydantic.Field(
        default=0.01,
        ge=0,
    )

    """ElasticNet l1_ratio parameter"""
    L1_RATIO: float = pydantic.Field(
        default=0.5,
        ge=0,
        le=1,
    )

    """ElasticNet fit_intercept parameter"""
    FIT_INTERCEPT: bool = pydantic.Field(
        default=True,
    )

    """ElasticNet parameter to precompute Gram matrix"""
    PRECOMPUTE: bool = pydantic.Field(
        default=False,
    )

    """ElasticNet max_iter parameter"""
    MAX_ITER: int = pydantic.Field(
        default=1000,
        ge=1,
        le=2**32 - 1,
    )

    """ElasticNet copy_X parameter"""
    COPY_X: bool = pydantic.Field(
        default=True,
    )

    """ElasticNet tol parameter"""
    TOL: float = pydantic.Field(
        default=1e-4,
        gt=0,
    )

    """ElasticNet selection parameter"""
    SELECTION: SelectionChoice = pydantic.Field(
        default=SelectionChoice.CYCLIC,
    )


class SolarElasticNetSettings(BaseElasticNetSettings):
    """ElasticNet alpha parameter"""
    ALPHA: float = pydantic.Field(
        default=0.011572,
        ge=0,
    )

    """ElasticNet l1_ratio parameter"""
    L1_RATIO: float = pydantic.Field(
        default=0.139316,
        ge=0,
        le=1,
    )


class NonSolarElasticNetSettings(BaseElasticNetSettings):
    """ElasticNet alpha parameter"""
    ALPHA: float = pydantic.Field(
        default=0.002800,
        ge=0,
    )

    """ElasticNet l1_ratio parameter"""
    L1_RATIO: float = pydantic.Field(
        default=0.983800,
        ge=0,
        le=1,
    )


# analytic_features = ['GHI', 'Temperature', 'DHI', 'DNI', 'Relative Humidity', 'Wind Speed', 'Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI', 'Cloud Type']
class BaseHourlySettings(BaseSettings):
    """train features used within the model"""

    TRAIN_FEATURES: list[str] = pydantic.Field(
        default=["temperature"],
        frozen=True,
    )

    """minimum number of training hours per day below which a day is excluded"""
    MIN_DAILY_TRAINING_HOURS: int = pydantic.Field(
        default=12,
        ge=0,
        le=24,
    )

    """include temperature bins"""
    INCLUDE_TEMPERATURE_BINS: bool = pydantic.Field(
        default=True,
    )

    """how to bin temperature data"""
    TEMPERATURE_BINNING_METHOD: Optional[BinningChoice] = pydantic.Field(
        default=BinningChoice.EQUAL_BIN_WIDTH,
    )

    """number of temperature bins"""
    TEMPERATURE_BIN_COUNT: Optional[int] = pydantic.Field(
        default=6,
        ge=1,
    )

    """temperature bin width in fahrenheit"""
    TEMPERATURE_BIN_WIDTH: Optional[float] = pydantic.Field(
        default=None,
        ge=1,
    )

    
    EDGE_TEMPERATURE_BIN_RATE: Optional[float] = pydantic.Field(
        default=1.5,
        gt=1,
    )

    """number of clusters to use for temporal clustering (day, month)"""
    MAX_TEMPORAL_CLUSTER_COUNT: Optional[int] = pydantic.Field(
        default=6,
        ge=2,
    )

    """metric to use for temporal clustering"""
    TEMPORAL_CLUSTER_METRIC: ClusteringMetric = pydantic.Field(
        default=ClusteringMetric.EUCLIDEAN,
    )

    """number of times to run k-means clustering"""
    TEMPORAL_CLUSTER_N_INIT: int = pydantic.Field(
        default=5,
        ge=1,
    )

    """supplemental data"""
    SUPPLEMENTAL_DATA: Optional[dict] = pydantic.Field(
        default=None,
    )

    """ElasticNet settings"""
    ELASTICNET: BaseElasticNetSettings = pydantic.Field(
        default_factory=BaseElasticNetSettings,
    )

    """seed for any random state assignment (ElasticNet, Clustering)"""
    SEED: Optional[int] = pydantic.Field(
        default=None,
        ge=0,
    )


    @pydantic.model_validator(mode="after")
    def _check_temperature_bins(self):
        # check that temperature binning method is set based on include temperature bins
        if self.INCLUDE_TEMPERATURE_BINS:
            if self.TEMPERATURE_BINNING_METHOD is None:
                raise ValueError(
                    "'TEMPERATURE_BINNING_METHOD' must be specified if 'INCLUDE_TEMPERATURE_BINS' is True."
                )
        else:
            if self.TEMPERATURE_BINNING_METHOD is not None:
                raise ValueError(
                    "'TEMPERATURE_BINNING_METHOD' must be None if 'INCLUDE_TEMPERATURE_BINS' is False."
                )

        # check that temperature bin count is set based on binning method
        if self.TEMPERATURE_BINNING_METHOD is None:
            if self.TEMPERATURE_BIN_COUNT is not None:
                raise ValueError(
                    "'TEMPERATURE_BIN_COUNT' must be None if 'TEMPERATURE_BINNING_METHOD' is None."
                )
            if self.TEMPERATURE_BIN_WIDTH is not None:
                raise ValueError(
                    "'TEMPERATURE_BIN_WIDTH' must be None if 'TEMPERATURE_BINNING_METHOD' is None."
                )
        else:
            if self.TEMPERATURE_BINNING_METHOD == BinningChoice.SET_BIN_WIDTH:
                if self.TEMPERATURE_BIN_WIDTH is None:
                    raise ValueError(
                        "'TEMPERATURE_BIN_WIDTH' must be specified if 'TEMPERATURE_BINNING_METHOD' is 'SET_BIN_WIDTH'."
                    )
                if self.TEMPERATURE_BIN_COUNT is not None:
                    raise ValueError(
                        "'TEMPERATURE_BIN_COUNT' must be None if 'TEMPERATURE_BINNING_METHOD' is 'SET_BIN_WIDTH'."
                    )
            else:
                if self.TEMPERATURE_BIN_COUNT is None:
                    raise ValueError(
                        "'TEMPERATURE_BIN_COUNT' must be specified if 'TEMPERATURE_BINNING_METHOD' is not None."
                    )
                if self.TEMPERATURE_BIN_WIDTH is not None:
                    raise ValueError(
                        "'TEMPERATURE_BIN_WIDTH' must be None if 'TEMPERATURE_BINNING_METHOD' is not None."
                    )

        return self

    @pydantic.model_validator(mode="after")
    def _check_seed(self):
        if self.SEED is None:
            self._SEED = np.random.randint(0, 2**32 - 1)
        else:
            self._SEED = self.SEED

        self.ELASTICNET._SEED = self._SEED

        return self


class HourlySolarSettings(BaseHourlySettings):
    """train features used within the model"""

    TRAIN_FEATURES: list[str] = pydantic.Field(
        default=["temperature", "ghi"],
    )

    """number of temperature bins"""
    TEMPERATURE_BIN_COUNT: Optional[int] = pydantic.Field(
        default=6,
        ge=1,
    )

    """ElasticNet settings"""
    ELASTICNET: SolarElasticNetSettings = pydantic.Field(
        default_factory=SolarElasticNetSettings,
    )

    @pydantic.model_validator(mode="after")
    def _check_features(self):
        # make all features lowercase
        self._TRAIN_FEATURES = [s.lower() for s in self.TRAIN_FEATURES]

        for feature in ["temperature", "ghi"]:
            if feature not in self._TRAIN_FEATURES:
                self._TRAIN_FEATURES.insert(0, feature)

        self._TRAIN_FEATURES = sorted(
            self._TRAIN_FEATURES, key=lambda x: x not in ["temperature", "ghi"]
        )

        return self


class HourlyNonSolarSettings(BaseHourlySettings):
    """number of temperature bins"""

    TEMPERATURE_BIN_COUNT: Optional[int] = pydantic.Field(
        default=10,
        ge=1,
    )

    """ElasticNet settings"""
    ELASTICNET: NonSolarElasticNetSettings = pydantic.Field(
        default_factory=NonSolarElasticNetSettings,
    )

    @pydantic.model_validator(mode="after")
    def _check_features(self):
        # make all features lowercase
        self._TRAIN_FEATURES = [s.lower() for s in self.TRAIN_FEATURES]

        if "temperature" not in self._TRAIN_FEATURES:
            self._TRAIN_FEATURES.insert(0, "temperature")

        return self


def HourlyNonSolarSettingsV2(**kwargs):
    input_kwargs = {"TEMPERATURE_BIN_COUNT": 6, "ALPHA": 0.010206, "L1_RATIO": 0.241955}
    input_kwargs.update(kwargs)

    return HourlyNonSolarSettings(**input_kwargs)


class SerializeModel(BaseSettings):
    class Config:
        arbitrary_types_allowed = True

    SETTINGS: Optional[BaseHourlySettings] = None
    TEMPORAL_CLUSTERS: Optional[list[list[int]]] = None
    TEMPERATURE_BIN_EDGES: Optional[list] = None
    TS_FEATURES: Optional[list] = None
    CATEGORICAL_FEATURES: Optional[list] = None
    FEATURE_SCALER: Optional[dict[str, list[float]]] = None
    CATAGORICAL_SCALER: Optional[dict[str, list[float]]] = None
    Y_SCALER: Optional[list[float]] = None
    COEFFICIENTS: Optional[list[list[float]]] = None
    INTERCEPT: Optional[list[float]] = None
    BASELINE_METRICS: Optional[BaselineMetrics] = None
