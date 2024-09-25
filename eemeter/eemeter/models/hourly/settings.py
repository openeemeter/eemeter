from __future__ import annotations

import numpy as np
import pandas as pd

import pydantic

from enum import Enum
from typing import Optional, TypeVar

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


class TemperatureBinSettings(BaseSettings):
    """how to bin temperature data"""
    METHOD: BinningChoice = pydantic.Field(
        default=BinningChoice.SET_BIN_WIDTH,
    )

    """number of temperature bins"""
    N_BINS: Optional[int] = pydantic.Field(
        default=None,
        ge=1,
    )

    """temperature bin width in fahrenheit"""
    BIN_WIDTH: Optional[float] = pydantic.Field(
        default=15,
        ge=1,
    )

    """rate for edge temperature bins"""
    EDGE_BIN_RATE: Optional[float] = pydantic.Field(
        default=None,
        gt=1,
    )

    @pydantic.model_validator(mode="after")
    def _check_temperature_bins(self):
        # check that temperature bin count is set based on binning method
        if self.METHOD is None:
            if self.N_BINS is not None:
                raise ValueError(
                    "'N_BINS' must be None if 'METHOD' is None."
                )
            if self.BIN_WIDTH is not None:
                raise ValueError(
                    "'N_BINS' must be None if 'METHOD' is None."
                )
        else:
            if self.METHOD == BinningChoice.SET_BIN_WIDTH:
                if self.BIN_WIDTH is None:
                    raise ValueError(
                        "'N_BINS' must be specified if 'METHOD' is 'SET_BIN_WIDTH'."
                    )
                if self.N_BINS is not None:
                    raise ValueError(
                        "'N_BINS' must be None if 'METHOD' is 'SET_BIN_WIDTH'."
                    )
            else:
                if self.N_BINS is None:
                    raise ValueError(
                        "'N_BINS' must be specified if 'METHOD' is not None."
                    )
                if self.BIN_WIDTH is not None:
                    raise ValueError(
                        "'N_BINS' must be None if 'METHOD' is not None."
                    )

        return self


class TimeSeriesKMeansSettings(BaseSettings):
    """maximum number of clusters to use for temporal clustering"""
    MAX_CLUSTER_COUNT: int = pydantic.Field(
        default=24,
        ge=2,
        le=7*12, # 7 days * 12 months for year of data
    )

    """maximum number of iterations for k-means clustering for a single run"""
    MAX_ITER: int = pydantic.Field(
        default=50,
        ge=1,
    )

    """inertia variation threshold"""
    TOL: float = pydantic.Field(
        default=1e-3,
        gt=0,
    )

    """number of times to run k-means clustering"""
    N_INIT: int = pydantic.Field(
        default=5,
        ge=1,
    )

    """metric to use for cluster assignment and barycenter computation"""
    METRIC: ClusteringMetric = pydantic.Field(
        default=ClusteringMetric.EUCLIDEAN,
    )

    """maximum number of iterations for barycenter computation"""
    MAX_ITER_BARYCENTER: int = pydantic.Field(
        default=100,
        ge=1,
    )

    """initialization method for k-means clustering"""
    INIT_METHOD: str = pydantic.Field(
        default="k-means++",
    )

    """sample size for calculating silhouette score of clustering"""
    SCORE_SAMPLE_SIZE: Optional[int] = pydantic.Field(
        default=None,
        gt=1,
    )


class ElasticNetSettings(BaseSettings):
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


class SolarElasticNetSettings(ElasticNetSettings):
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


class NonSolarElasticNetSettings(ElasticNetSettings):
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

    """temperature bin settings"""
    TEMPERATURE_BIN: Optional[TemperatureBinSettings] = pydantic.Field(
        default_factory=TemperatureBinSettings,
    )

    """settings for temporal clustering"""
    TEMPORAL_CLUSTER: TimeSeriesKMeansSettings = pydantic.Field(
        default_factory=TimeSeriesKMeansSettings,
    )

    """supplemental time series column names"""
    SUPPLEMENTAL_TIME_SERIES_COLUMNS: Optional[list] = pydantic.Field(
        default=None,
    )

    """supplemental categorical column names"""
    SUPPLEMENTAL_CATEGORICAL_COLUMNS: Optional[list] = pydantic.Field(
        default=None,
    )

    """ElasticNet settings"""
    ELASTICNET: ElasticNetSettings = pydantic.Field(
        default_factory=ElasticNetSettings,
    )

    """seed for any random state assignment (ElasticNet, Clustering)"""
    SEED: Optional[int] = pydantic.Field(
        default=None,
        ge=0,
    )

    @pydantic.model_validator(mode="after")
    def _check_seed(self):
        if self.SEED is None:
            self._SEED = np.random.randint(0, 2**32 - 1)
        else:
            self._SEED = self.SEED

        self.ELASTICNET._SEED = self._SEED
        self.TEMPORAL_CLUSTER._SEED = self._SEED

        return self


class HourlySolarSettings(BaseHourlySettings):
    """train features used within the model"""

    TRAIN_FEATURES: list[str] = pydantic.Field(
        default=["temperature", "ghi"],
    )

    """number of temperature bins"""
    # TEMPERATURE_BIN_COUNT: Optional[int] = pydantic.Field(
    #     default=6,
    #     ge=1,
    # )

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
    # TEMPERATURE_BIN_COUNT: Optional[int] = pydantic.Field(
    #     default=10,
    #     ge=1,
    # )

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
    input_kwargs = {"ALPHA": 0.010206, "L1_RATIO": 0.241955}
    input_kwargs.update(kwargs)

    return HourlyNonSolarSettings(**input_kwargs)


HourlySettings = TypeVar('HourlySettings', bound=BaseHourlySettings)
class SerializeModel(BaseSettings):
    class Config:
        arbitrary_types_allowed = True

    SETTINGS: Optional[HourlySettings] = None
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
