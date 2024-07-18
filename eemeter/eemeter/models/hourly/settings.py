from __future__ import annotations

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
    #TODO: remove this
    """lagged train features used within the model"""
    LAGGED_FEATURES: Optional[list[str]] = pydantic.Field(
        default=None, 
        validate_default=True,
    )
    #TODO: remove this
    """window"""
    WINDOW: Optional[int] = pydantic.Field(
        default=None,
        ge=1,                       # TODO: CORRECT THIS BEFORE RELEASE
        validate_default=True,
    )

    N_BINS: Optional[int] = pydantic.Field(
        default=5,
        ge=1,                       
        validate_default=True,
    )

    INCLUDE_TEMP_BINS_CATAGORY: bool = pydantic.Field(
        default=True,
        validate_default=True,
    )

    SAME_SIZE_BIN: bool = pydantic.Field(
        default=True,
        validate_default=True,
    )
    
    INCLUDE_SEASONS_CATAGORY: bool = pydantic.Field(
        default=False,
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

    """ElasticNet random_state parameter"""
    SEED: Optional[int] = pydantic.Field(
        default=None,
        ge=0,
        validate_default=True,
    )


    @pydantic.model_validator(mode="after")
    def _lowercase_features(self):
        self.TRAIN_FEATURES = [s.lower() for s in self.TRAIN_FEATURES]

        if self.LAGGED_FEATURES is not None:
            self.LAGGED_FEATURES = [s.lower() for s in self.LAGGED_FEATURES]

        return self

    @pydantic.model_validator(mode="after")
    def _check_features(self):
        if "temperature" not in self.TRAIN_FEATURES:
            self.TRAIN_FEATURES.insert(0, "temperature")

        # Lag features
        if self.LAGGED_FEATURES is None:
            if self.WINDOW is not None:
                raise ValueError("WINDOW is set but LAGGED_FEATURES is not set")
        else:
            if len(self.LAGGED_FEATURES) == 0:
                raise ValueError("LAGGED_FEATURES is empty, set as None to remove lagged features")
            
            if self.WINDOW is None:
                raise ValueError("LAGGED_FEATURES is set but WINDOW is not set")
            
            # Check if feature in lagged features but not in train features raise error
            lag_feature_error = [f for f in self.LAGGED_FEATURES if f not in self.TRAIN_FEATURES]

            if lag_feature_error:
                raise ValueError(f"Features {lag_feature_error} are in LAGGED_FEATURES but not in TRAIN_FEATURES")

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