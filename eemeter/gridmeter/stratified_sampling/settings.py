"""
module defines settings to be used for individual meter matching
"""

from __future__ import annotations

import pydantic

import gridmeter.stratified_sampling.const as _const
from gridmeter._utils.base_settings import BaseSettings

from typing import Optional, Literal, Union


class StratificationColumnSettings(BaseSettings):
    """column name to use for stratification"""
    COLUMN_NAME: str = pydantic.Field()

    """fixed number of bins to use for stratification"""
    N_BINS: Optional[int] = pydantic.Field(
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
    AUTO_BIN_EQUIVALENCE: Literal[False] = False


class DSS_StratificationColumnSettings(StratificationColumnSettings):
    """fixed number of bins to use for stratification"""
    N_BINS: Literal[None] = None

    """column requires equivalence when auto-binning"""
    AUTO_BIN_EQUIVALENCE: Literal[True] = True


class Settings(BaseSettings):
    """
    MIN_N_SAMPLED_TO_N_TREATMENT_RATIO: int
        TODO: FILL THIS OUT
    SEED: int
        Seed for random number generator
    """

    MIN_N_TREATMENT_PER_BIN: int = pydantic.Field(
        default=0, 
        ge=0, 
        validate_default=True,
    )

    SEED: int = pydantic.Field(
        default=42, 
        ge=0, 
        validate_default=True,
    )


class StratifiedSamplingSettings(Settings):
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

    EQUIVALENCE_METHOD: Literal[None] = None

    EQUIVALENCE_QUANTILE: Literal[None] = None

    MIN_N_BINS: Literal[None] = None

    MAX_N_BINS: Literal[None] = None

    MIN_N_SAMPLED_TO_N_TREATMENT_RATIO: float = pydantic.Field(
        default=4, 
        ge=0, 
        validate_default=True,
    )

    STRATIFICATION_COLUMN: Union[list[StratificationColumnSettings], list[dict]] = pydantic.Field(
        default=[
            StratificationColumnSettings(column_name="summer_usage"),
            StratificationColumnSettings(column_name="winter_usage"),
        ],
    )

    """set stratification column classes with given dictionaries"""
    @pydantic.model_validator(mode="after")
    def _set_nested_classes(self):
        if len(self.STRATIFICATION_COLUMN) > 3:
            raise ValueError("A maximum of 3 STRATIFICATION_COLUMN's are allowed")

        strat_settings = []
        has_dict = False
        for strat_item in self.STRATIFICATION_COLUMN:
            if isinstance(strat_item, dict):
                has_dict = True
                strat_class = StratificationColumnSettings(**strat_item)

            else:
                strat_class = strat_item

            strat_settings.append(strat_class)

        if has_dict:
            self.STRATIFICATION_COLUMN = strat_settings

        return self


# subclass Settings to change default values
class DistanceStratifiedSamplingSettings(Settings):
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
        default=5000, 
        ge=1, 
        validate_default=True,
    )

    RELAX_N_SAMPLES_APPROX_CONSTRAINT: bool = pydantic.Field(
        default=True, 
    )

    EQUIVALENCE_METHOD: _const.DistanceMetric = pydantic.Field(
        default=_const.DistanceMetric.CHISQUARE,
        validate_default=True,
    )

    EQUIVALENCE_QUANTILE: int = pydantic.Field(
        default=25,
        validate_default=True,
    )

    MIN_N_BINS: int = pydantic.Field(
        default=1, 
        ge=1, 
        validate_default=True,
    )

    MAX_N_BINS: int = pydantic.Field(
        default=8, 
        ge=2, 
        validate_default=True,
    )

    MIN_N_SAMPLED_TO_N_TREATMENT_RATIO: float = pydantic.Field(
        default=0.25, 
        ge=0, 
        validate_default=True,
    )

    STRATIFICATION_COLUMN: Union[list[DSS_StratificationColumnSettings],list[dict]] = pydantic.Field(
        default=[
            DSS_StratificationColumnSettings(column_name="summer_usage"),
            DSS_StratificationColumnSettings(column_name="winter_usage"),
        ],
    )

    """set stratification column classes with given dictionaries"""
    @pydantic.model_validator(mode="after")
    def _set_nested_classes(self):
        if len(self.STRATIFICATION_COLUMN) > 3:
            raise ValueError("A maximum of 3 STRATIFICATION_COLUMN's are allowed")

        strat_settings = []
        has_dict = False
        for strat_item in self.STRATIFICATION_COLUMN:
            if isinstance(strat_item, dict):
                has_dict = True
                strat_class = DSS_StratificationColumnSettings(**strat_item)

            else:
                strat_class = strat_item

            strat_settings.append(strat_class)

        if has_dict:
            self.STRATIFICATION_COLUMN = strat_settings

        return self


if __name__ == "__main__":
    s = StratifiedSamplingSettings()
    # s = DistanceStratifiedSamplingSettings()

    print(s.model_dump_json())