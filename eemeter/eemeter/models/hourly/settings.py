from __future__ import annotations

import numpy as np
import pandas as pd

import pydantic

from enum import Enum
from typing import Optional, Literal, Union, TypeVar, Dict

import pywt

from eemeter.common.base_settings import BaseSettings
from eemeter.common.metrics import BaselineMetrics

# from eemeter.common.const import CountryCode


class SelectionChoice(str, Enum):
    CYCLIC = "cyclic"
    RANDOM = "random"


class ScalingChoice(str, Enum):
    ROBUSTSCALER = "robustscaler"
    STANDARDSCALER = "standardscaler"


class BinningChoice(str, Enum):
    EQUAL_SAMPLE_COUNT = "equal_sample_count"
    EQUAL_BIN_WIDTH = "equal_bin_width"
    SET_BIN_WIDTH = "set_bin_width"


class ClusteringMetric(str, Enum):
    EUCLIDEAN = "euclidean"
    DTW = "dtw"
    SOFTDTW = "softdtw"

class ClusterScoringMetric(str, Enum):
    SILHOUETTE = "silhouette"
    SILHOUETTE_MEDIAN = "silhouette_median"
    VARIANCE_RATIO = "variance_ratio"
    DAVIES_BOULDIN = "davies-bouldin"

class DistanceMetric(str, Enum):
    """
    what distance method to use
    """

    EUCLIDEAN = "euclidean"
    SEUCLIDEAN = "seuclidean"
    MANHATTAN = "manhattan"
    COSINE = "cosine"


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
        default=12,
        ge=1,
    )

    """use edge bins bool"""
    INCLUDE_EDGE_BINS: bool = pydantic.Field(
        default=True,
    )

    """rate for edge temperature bins"""
    EDGE_BIN_RATE: Optional[Union[float, Literal["heuristic"]]] = pydantic.Field(
        default="heuristic",
    )

    """number of days in edge bins"""
    EDGE_BIN_HOURS: Optional[int] = pydantic.Field(
        default=400, # better than 75?
        ge=5,
    )

    """offset normalized temperature range for edge bins (keeps exp from blowing up)"""
    EDGE_BIN_TEMPERATURE_RANGE_OFFSET: Optional[float] = pydantic.Field(
        default=0.5,
        ge=0,
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
                elif isinstance(self.BIN_WIDTH, float):
                    if self.BIN_WIDTH <= 0:
                        raise ValueError(
                            "'BIN_WIDTH' must be greater than 0."
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

    @pydantic.model_validator(mode="after")
    def _check_edge_bins(self):
        if self.METHOD != BinningChoice.SET_BIN_WIDTH:
            if self.INCLUDE_EDGE_BINS:
                raise ValueError(
                    "'INCLUDE_EDGE_BINS' must be False if 'METHOD' is not 'SET_BIN_WIDTH'."
                )
            
        if self.INCLUDE_EDGE_BINS:
            if self.EDGE_BIN_RATE is None:
                raise ValueError(
                    "'EDGE_BIN_RATE' must be specified if 'INCLUDE_EDGE_BINS' is True."
                )
            if self.EDGE_BIN_HOURS is None:
                raise ValueError(
                    "'EDGE_BIN_DAYS' must be specified if 'INCLUDE_EDGE_BINS' is True."
                )

        else:
            if self.EDGE_BIN_RATE is not None:
                raise ValueError(
                    "'EDGE_BIN_RATE' must be None if 'INCLUDE_EDGE_BINS' is False."
                )
            if self.EDGE_BIN_HOURS is not None:
                raise ValueError(
                    "'EDGE_BIN_DAYS' must be None if 'INCLUDE_EDGE_BINS' is False."
                )

        return self


class TemporalClusteringSettings(BaseSettings):
    """wavelet decomposition level"""
    WAVELET_N_LEVELS: int = pydantic.Field(
        default=5,
        ge=1,
    )
    
    """wavelet choice for wavelet decomposition"""
    WAVELET_NAME: str = pydantic.Field(
        default="haar", # maybe db3?
    )

    """signal extension mode for wavelet decomposition"""
    WAVELET_MODE: str = pydantic.Field(
        default="periodization",
    )

    """minimum variance ratio for PCA clustering"""
    PCA_MIN_VARIANCE_RATIO_EXPLAINED: float = pydantic.Field(
        default=0.8,
        ge=0.5,
        le=1,
    )

    """number of times to recluster"""
    RECLUSTER_COUNT: int = pydantic.Field(
        default=3,
        ge=1,
    )

    """lower bound for number of clusters"""
    N_CLUSTER_LOWER: int = pydantic.Field(
        default=2,
        ge=2,
    )

    """upper bound for number of clusters"""
    N_CLUSTER_UPPER: int = pydantic.Field(
        default=24,
        ge=2,
    )

    """minimum cluster size"""
    MIN_CLUSTER_SIZE: int = pydantic.Field(
        default=1,
        ge=1,
    )

    """scoring method for clustering"""
    SCORE_METRIC: ClusterScoringMetric = pydantic.Field(
        default=ClusterScoringMetric.VARIANCE_RATIO,
    )

    """distance metric for clustering"""
    DISTANCE_METRIC: DistanceMetric = pydantic.Field(
        default=DistanceMetric.EUCLIDEAN,
    )

    @pydantic.model_validator(mode="after")
    def _check_wavelet(self):
        all_wavelets = pywt.wavelist(kind='discrete')
        if self.WAVELET_NAME not in all_wavelets:
            raise ValueError(f"'WAVELET_NAME' must be a valid wavelet in PyWavelets: \n{all_wavelets}")

        all_modes = pywt.Modes.modes
        if self.WAVELET_MODE not in all_modes:
            raise ValueError(f"'WAVELET_MODE' must be a valid mode in PyWavelets: \n{all_modes}")

        return self


class ElasticNetSettings(BaseSettings):
    """ElasticNet alpha parameter"""
    ALPHA: float = pydantic.Field(
        default=0.04,
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
    TEMPORAL_CLUSTER: TemporalClusteringSettings = pydantic.Field(
        default_factory=TemporalClusteringSettings,
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

    """Feature scaling method"""
    SCALING_METHOD: ScalingChoice = pydantic.Field(
        default=ScalingChoice.STANDARDSCALER,
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

    @pydantic.model_validator(mode="after")
    def _check_features(self):
        # make all features lowercase
        self._TRAIN_FEATURES = [s.lower() for s in self.TRAIN_FEATURES]

        if "temperature" not in self._TRAIN_FEATURES:
            self._TRAIN_FEATURES.insert(0, "temperature")

        return self


HourlySettings = TypeVar('HourlySettings', bound=BaseHourlySettings)
class SerializeModel(BaseSettings):
    class Config:
        arbitrary_types_allowed = True

    SETTINGS: Optional[HourlySettings] = None
    TEMPORAL_CLUSTERS: Optional[list[list[int]]] = None
    TEMPERATURE_BIN_EDGES: Optional[list] = None
    TEMPERATURE_EDGE_BIN_COEFFICIENTS: Optional[Dict[int, Dict[str, float]]] = None
    TS_FEATURES: Optional[list] = None
    CATEGORICAL_FEATURES: Optional[list] = None
    FEATURE_SCALER: Optional[Dict[str, list[float]]] = None
    CATAGORICAL_SCALER: Optional[Dict[str, list[float]]] = None
    Y_SCALER: Optional[list[float]] = None
    COEFFICIENTS: Optional[list[list[float]]] = None
    INTERCEPT: Optional[list[float]] = None
    BASELINE_METRICS: Optional[BaselineMetrics] = None
