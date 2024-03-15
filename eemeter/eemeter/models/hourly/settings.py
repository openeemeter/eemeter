from __future__ import annotations

import pydantic

from enum import Enum
from typing import Any, Dict, Optional

from eemeter.common.base_settings import BaseSettings
from eemeter.common.metrics import BaselineMetrics


class SelectionChoice(str, Enum):
    CYCLIC = "cyclic"
    RANDOM = "random"


# analytic_features = ['GHI', 'Temperature', 'DHI', 'DNI', 'Relative Humidity', 'Wind Speed', 'Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI', 'Cloud Type']
class HourlySettings(BaseSettings):
    """train features used within the model"""
    TRAIN_FEATURES: list[str] = pydantic.Field(
        default=['ghi', 'temperature'], 
        validate_default=True,
    )

    """lagged train features used within the model"""
    LAGGED_FEATURES: list[str] = pydantic.Field(
        default=['temperature'], 
        validate_default=True,
    )

    # TODO: Armin what is this?
    """window"""
    WINDOW: int = pydantic.Field(
        default=1,
        ge=1,
        validate_default=True,
    )

    # TODO: Armin is this always true/false or can it be a list[string]?
    """supplemental data"""
    SUPPLEMENTAL_DATA: bool = pydantic.Field(
        default=False,
        validate_default=True,
    )

    # TODO: Armin what is this?
    """output"""
    OUTPUT: list[str] = pydantic.Field(
        default=['start_local', 'temperature', 'ghi', 'clearsky_ghi', 'observed', 'new_model', 'month'], 
        validate_default=True,
    )

    """ElasticNet alpha parameter"""
    ALPHA: float = pydantic.Field(
        default=0.1,
        ge=0,
        validate_default=True,
    )

    """ElasticNet l1_ratio parameter"""
    L1_RATIO: float = pydantic.Field(
        default=0.1,
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

    """ElasticNet random_state parameter"""
    SEED: Optional[int] = pydantic.Field(
        default=None,
        ge=0,
        validate_default=True,
    )


    @pydantic.model_validator(mode="after")
    def _lowercase_features(self):
        self.TRAIN_FEATURES = [s.lower() for s in self.TRAIN_FEATURES]
        self.LAGGED_FEATURES = [s.lower() for s in self.LAGGED_FEATURES]
        self.OUTPUT = [s.lower() for s in self.OUTPUT]

        return self

    @pydantic.model_validator(mode="after")
    def _check_features(self):
        if "temperature" not in self.TRAIN_FEATURES:
            self.TRAIN_FEATURES.insert(0, "temperature")

        # TODO: Armin, do we always want to include temperature in lagged features?
        if "temperature" not in self.LAGGED_FEATURES:
            self.TRAIN_FEATURES.insert(0, "temperature")

        # if feature in lagged features but not in train features raise error
        lag_feature_error = [f for f in self.LAGGED_FEATURES if f not in self.TRAIN_FEATURES]

        if lag_feature_error:
            raise ValueError(f"Features {lag_feature_error} are in LAGGED_FEATURES but not in TRAIN_FEATURES")

        return self
    

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
