from __future__ import annotations

import numpy as np

import pydantic

from enum import Enum
from typing import Any, Dict, Optional

from eemeter.common.base_settings import BaseSettings
from eemeter.common.metrics import BaselineMetrics
# from eemeter.common.const import CountryCode


class SelectionChoice(str, Enum):
    CYCLIC = "cyclic"
    RANDOM = "random"


# analytic_features = ['GHI', 'Temperature', 'DHI', 'DNI', 'Relative Humidity', 'Wind Speed', 'Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI', 'Cloud Type']
class HourlySettings(BaseSettings):
    """train features used within the model"""
    TRAIN_FEATURES: list[str] = pydantic.Field(
        default=['temperature'], 
        validate_default=True,
    )

    TEMPERATURE_BIN_COUNT: Optional[int] = pydantic.Field(
        default=5,
        ge=1,                       
        validate_default=True,
    )

    INCLUDE_TEMPERATURE_BINS: bool = pydantic.Field(
        default=True,
        validate_default=True,
    )

    TEMPERATURE_BIN_SAME_SIZE: bool = pydantic.Field(
        default=True,
        validate_default=True,
    )

    MAX_CLUSTER_NUMBER: Optional[int] = pydantic.Field(
        default=10,
        ge=2,
        validate_default=True,
    )

    """supplemental data"""
    SUPPLEMENTAL_DATA: Optional[dict] = pydantic.Field(
        default=None,
        validate_default=True,
    )

    """ElasticNet alpha parameter"""
    ALPHA: float = pydantic.Field(
        default=0.0326,
        ge=0,
        validate_default=True,
    )

    """ElasticNet l1_ratio parameter"""
    L1_RATIO: float = pydantic.Field(
        default=0.6289,
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
        validate_default=True,
    )

    """seed for any random state assignment (ElasticNet, Clustering)"""
    SEED: Optional[int] = pydantic.Field(
        default=None,
        ge=0,
        validate_default=True,
    )

    @pydantic.model_validator(mode="after")
    def _lowercase_features(self):
        self.TRAIN_FEATURES = [s.lower() for s in self.TRAIN_FEATURES]

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


class HourlySolarSettings(HourlySettings):
    """train features used within the model"""
    TRAIN_FEATURES: list[str] = pydantic.Field(
        default=['temperature', 'ghi'], 
        validate_default=True,
    )

    """ElasticNet alpha parameter"""
    ALPHA: float = pydantic.Field(
        default=0.0373,
        ge=0,
        validate_default=True,
    )

    """ElasticNet l1_ratio parameter"""
    L1_RATIO: float = pydantic.Field(
        default=0.4784,
        ge=0,
        le=1,
        validate_default=True,
    )


class HourlyNonSolarSettings(HourlySettings):
    pass


class SerializeModel(BaseSettings):
    class Config:
        arbitrary_types_allowed = True

    SETTINGS: Optional[HourlySettings] = None
    COEFFICIENTS: Optional[list[list[float]]] = None
    INTERCEPT: Optional[list[float]] = None
    FEATURE_SCALER: Optional[Dict[str, list[float]]] = None
    CATAGORICAL_SCALER: Optional[Dict[str, list[float]]] = None
    Y_SCALER: Optional[list[float]] = None
    BASELINE_METRICS: Optional[BaselineMetrics] = None