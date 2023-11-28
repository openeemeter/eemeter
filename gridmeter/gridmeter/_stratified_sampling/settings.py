"""
module defines settings to be used for individual meter matching
"""

from __future__ import annotations

import pydantic

import gridmeter._stratified_sampling.const as _const
from gridmeter._utils.base_settings import BaseSettings

from typing import Optional


class StratificationColumnSettings(BaseSettings):
    """column name to use for stratification"""
    COLUMN_NAME: str = pydantic.Field()

    """fixed number of bins to use for stratification"""
    N_BINS: int | None = pydantic.Field(
        default=8, 
        ge=2, 
        validate_default=True,
    )

    """minimum treatment value used to construct bins (used to remove outliers)"""
    MIN_VALUE_ALLOWED: int = pydantic.Field(
        default=3000, 
        ge=0, 
        validate_default=True,
    )

    """maximum treatment value used to construct bins (used to remove outliers)"""
    MAX_VALUE_ALLOWED: int = pydantic.Field(
        default=6000, 
        ge=0, 
        validate_default=True,
    )

    """whether to use fixed width bins or fixed proportion bins"""
    IS_FIXED_WIDTH: bool = pydantic.Field(
        default=False, 
    )

    """column requires equivalence when auto-binning"""
    AUTO_BIN_EQUIVALENCE: bool = pydantic.Field(
        default=True, 
    )


class Settings(BaseSettings):
    """
    n_samples_approx: int
        approximate number of total samples from df_pool. It is approximate because
        there may be some slight discrepencies around the total count to ensure
        that each bin has the correct percentage of the total.
    min_n_treatment_per_bin: int
        Minimum number of treatment samples that must exist in a given bin for 
        it to be considered a non-outlier bin (only applicable if there are 
        cols with fixed_width=True)
    min_n_sampled_to_n_treatment_ratio: int
    relax_n_samples_approx_constraint: bool
        If True, treats n_samples_approx as an upper bound, but gets as many comparison group
        meters as available up to n_samples_approx. If False, it raises an exception
        if there are not enough comparison pool meters to reach n_samples_approx.
    """
    
    N_SAMPLES_APPROX: Optional[int] = pydantic.Field(
        default=None, 
        ge=1, 
        validate_default=True,
    )

    RELAX_N_SAMPLES_APPROX_CONSTRAINT: bool = pydantic.Field(
        default=False, 
    )

    EQUIVALENCE_METHOD: _const.DistanceMetric | None = pydantic.Field(
        default=None,
        validate_default=True,
    )

    EQUIVALENCE_QUANTILE: int | None = pydantic.Field(
        default=None,
        validate_default=True,
    )

    MIN_N_BINS: int | None = pydantic.Field(
        default=None, 
        ge=1, 
        validate_default=True,
    )

    MAX_N_BINS: int | None = pydantic.Field(
        default=None, 
        ge=2, 
        validate_default=True,
    )

    MIN_N_TREATMENT_PER_BIN: int = pydantic.Field(
        default=0, 
        ge=0, 
        validate_default=True,
    )

    MIN_N_SAMPLED_TO_N_TREATMENT_RATIO: float = pydantic.Field(
        default=4, 
        ge=0, 
        validate_default=True,
    )

    SEED: int = pydantic.Field(
        default=42, 
        ge=0, 
        validate_default=True,
    )

    STRATIFICATION_COLUMN: list[StratificationColumnSettings] | list[dict] = pydantic.Field(
        default=[
            StratificationColumnSettings(column_name="summer_usage"),
            StratificationColumnSettings(column_name="winter_usage"),
        ],
    )

    """set stratification column classes with given dictionaries"""
    @pydantic.model_validator(mode="after")
    def _set_nested_classes(self):
        settings = []
        has_dict = False
        for strat_item in self.STRATIFICATION_COLUMN:
            if isinstance(strat_item, dict):
                has_dict = True
                strat_class = StratificationColumnSettings(**strat_item)

            else:
                strat_class = strat_item

            settings.append(strat_class)

        if has_dict:
            self.STRATIFICATION_COLUMN = settings

        return self
    
    """Check values if equivalence method is None"""
    @pydantic.model_validator(mode="after")
    def _check_false_equivalence_keys(self):
        if self.EQUIVALENCE_METHOD is not None:
            return self

        if self.EQUIVALENCE_QUANTILE is not None:
            raise ValueError("EQUIVALENCE_QUANTILE must be None if EQUIVALENCE_METHOD is NONE")

        if self.MIN_N_BINS is not None:
            raise ValueError("MIN_N_BINS must be None if EQUIVALENCE_METHOD is NONE")
        
        if self.MAX_N_BINS is not None:
            raise ValueError("MAX_N_BINS must be None if EQUIVALENCE_METHOD is NONE")

        return self
    
    """Check values if equivalence method is not None"""
    @pydantic.model_validator(mode="after")
    def _check_true_equivalence_keys(self):
        if self.EQUIVALENCE_METHOD is None:
            return self

        if self.EQUIVALENCE_QUANTILE is None:
            raise ValueError("EQUIVALENCE_QUANTILE must not be None if EQUIVALENCE_METHOD is not NONE")

        if self.MIN_N_BINS is None:
            raise ValueError("MIN_N_BINS must not be None if EQUIVALENCE_METHOD is not NONE")
        
        if self.MAX_N_BINS is None:
            raise ValueError("MAX_N_BINS must not be None if EQUIVALENCE_METHOD is not NONE")
        
        for col_settings in self.STRATIFICATION_COLUMN:
            if col_settings.N_BINS is not None:
                raise ValueError("N_BINS must be None if EQUIVALENCE_METHOD is not NONE")
            
            if not col_settings.AUTO_BIN_EQUIVALENCE:
                raise ValueError("AUTO_BIN_EQUIVALENCE must not be None if EQUIVALENCE_METHOD is not NONE")

        return self

def stratified_sampling_settings(**kwargs) -> Settings:
    settings = Settings(**kwargs)

    return settings


def distance_stratified_sampling_settings(**kwargs) -> Settings:
    settings = {
        "N_SAMPLES_APPROX": 5000,
        "RELAX_N_SAMPLES_APPROX_CONSTRAINT": True,
        "EQUIVALENCE_METHOD": _const.DistanceMetric.CHISQUARE,
        "MIN_N_BINS": 1,
        "MAX_N_BINS": 8,
        "EQUIVALENCE_QUANTILE": 25,
        "MIN_N_SAMPLED_TO_N_TREATMENT_RATIO": 0.25,        
    }

    settings.update(kwargs)

    settings = Settings(**settings)

    return settings


if __name__ == "__main__":
    # s = Settings()
    # s = stratified_sampling_settings()
    s = distance_stratified_sampling_settings()

    print(s.model_dump_json())
