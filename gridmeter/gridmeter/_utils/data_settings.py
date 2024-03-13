"""
module defines settings to be used for individual meter matching
"""

from __future__ import annotations

import pydantic

from typing import Optional,Union

import gridmeter._utils.const as _const
from gridmeter._utils.base_settings import BaseSettings


min_data_pct = 0.8


# Note: Options list order defines how seasons will be ordered in the loadshape
class Season_Definition(BaseSettings):
    JANUARY: str = pydantic.Field(default="winter")
    FEBRUARY: str = pydantic.Field(default="winter")
    MARCH: str = pydantic.Field(default="shoulder")
    APRIL: str = pydantic.Field(default="shoulder")
    MAY: str = pydantic.Field(default="shoulder")
    JUNE: str = pydantic.Field(default="summer")
    JULY: str = pydantic.Field(default="summer")
    AUGUST: str = pydantic.Field(default="summer")
    SEPTEMBER: str = pydantic.Field(default="summer")
    OCTOBER: str = pydantic.Field(default="shoulder")
    NOVEMBER: str = pydantic.Field(default="winter")
    DECEMBER: str = pydantic.Field(default="winter")

    OPTIONS: list[str] = pydantic.Field(default=["summer", "shoulder", "winter"])

    """Set dictionaries of seasons"""
    @pydantic.model_validator(mode="after")
    def set_numeric_dict(self) -> Season_Definition:
        season_dict = {}
        for month, num in _const.season_num.items():
            val = getattr(self, month.upper())
            if val not in self.OPTIONS:
                raise ValueError(f"SeasonDefinition: {val} is not a valid option. Valid options are {self.OPTIONS}")
            
            season_dict[num] = val
        
        self._NUM_DICT = season_dict
        self._ORDER = {val: i for i, val in enumerate(self.OPTIONS)}

        return self


class Weekday_Weekend_Definition(BaseSettings):
    MONDAY: str = pydantic.Field(default="weekday")
    TUESDAY: str = pydantic.Field(default="weekday")
    WEDNESDAY: str = pydantic.Field(default="weekday")
    THURSDAY: str = pydantic.Field(default="weekday")
    FRIDAY: str = pydantic.Field(default="weekday")
    SATURDAY: str = pydantic.Field(default="weekend")
    SUNDAY: str = pydantic.Field(default="weekend")

    OPTIONS: list[str] = pydantic.Field(default=["weekday", "weekend"])

    """Set dictionaries of weekday/weekend"""
    @pydantic.model_validator(mode="after")
    def set_numeric_dict(self) -> Weekday_Weekend_Definition:
        weekday_dict = {}
        for day, num in _const.weekday_num.items():
            val = getattr(self, day.upper())
            if val not in self.OPTIONS:
                raise ValueError(f"WeekdayWeekendDefinition: {val} is not a valid option. Valid options are {self.OPTIONS}")
            
            weekday_dict[num] = val
        
        self._NUM_DICT = weekday_dict
        self._ORDER = {val: i for i, val in enumerate(self.OPTIONS)}

        return self
    

class Data_Settings(BaseSettings):
    """aggregation type for the loadshape"""
    AGG_TYPE: Optional[_const.AggType] = pydantic.Field(
        default=_const.AggType.MEAN,
        validate_default=True,
    )
    
    """type of loadshape to be used"""
    LOADSHAPE_TYPE: Optional[_const.LoadshapeType] = pydantic.Field(
        default=_const.LoadshapeType.MODELED, 
        validate_default=True,
    )

    """time period to be used for the loadshape"""
    TIME_PERIOD: Optional[_const.TimePeriod] = pydantic.Field(
        default=_const.TimePeriod.SEASONAL_HOURLY_DAY_OF_WEEK, 
        validate_default=True,
    )

    """interpolate missing values"""
    INTERPOLATE_MISSING: bool = pydantic.Field(
        default=True, 
        validate_default=True,
    )

    """minimum percentage of data required for a meter to be included"""
    MIN_DATA_PCT_REQUIRED: Optional[float] = pydantic.Field(
        default=min_data_pct, 
        validate_default=True,
    )

    @pydantic.field_validator("MIN_DATA_PCT_REQUIRED")
    @classmethod
    def validate_min_data_pct_required(cls, value):
        if value is None:
            pass

        elif value != min_data_pct:
            raise ValueError(f"MIN_DATA_PCT_REQUIRED must be {min_data_pct}")
        
        return value

    """season definition to be used for the loadshape"""
    SEASON: Union[dict, Season_Definition] = pydantic.Field(
        default=_const.default_season_def, 
    )

    """weekday/weekend definition to be used for the loadshape"""
    WEEKDAY_WEEKEND: Union[dict, Weekday_Weekend_Definition] = pydantic.Field(
        default=_const.default_weekday_weekend_def, 
    )

    """set season and weekday_weekend classes with given dictionaries"""
    @pydantic.model_validator(mode="after")
    def _set_nested_classes(self):
        if isinstance(self.SEASON, dict):
            self.SEASON = Season_Definition(**self.SEASON)

        if isinstance(self.WEEKDAY_WEEKEND, dict):
            self.WEEKDAY_WEEKEND = Weekday_Weekend_Definition(**self.WEEKDAY_WEEKEND)

        return self
    
    """validate loadshape/time series settings"""
    @pydantic.model_validator(mode="after")
    def _validate_loadshape_time_series_settings(self):
        ls_dict = {"AGG_TYPE": self.AGG_TYPE, "LOADSHAPE_TYPE": self.LOADSHAPE_TYPE, "TIME_PERIOD": self.TIME_PERIOD}
        is_set = {k: v is not None for k, v in ls_dict.items()}
        if any(is_set.values()):
            for k, v in is_set.items():
                if v is False:
                    raise ValueError(f"{k} must be set if any of the following are set: {list(is_set.keys())}")

        return self

    """set MIN_DATA_PCT_REQUIRED"""
    @pydantic.model_validator(mode="after")
    def _set_min_data_pct_on_interpolate(self):
        if self.INTERPOLATE_MISSING:
            self.MIN_DATA_PCT_REQUIRED = min_data_pct
        else:
            self.MIN_DATA_PCT_REQUIRED = None

        return self
    

if __name__ == "__main__":
    # Test SeasonDefinition
    # Note: Options list order defines how seasons will be orderd in the loadshape
    season_dict = {
        "options":  ["summer", "shoulder", "winter"],
        "January":   "winter", 
        "February":  "winter", 
        "March":     "shoulder", 
        "April":     "shoulder", 
        "May":       "shoulder", 
        "June":      "summer", 
        "July":      "summer", 
        "August":    "summer", 
        "September": "summer", 
        "October":   "shoulder", 
        "November":  "winter", 
        "December":  "winter",
        }

    # season = SeasonDefinition(**season_def)
    # print(season.model_dump_json())

    # Test WeekdayWeekendDefinition
    weekday_weekend_dict = {
        "options":  ["weekday", "weekend", "oops"],
        "Monday":    "weekday",
        "Tuesday":   "weekday",
        "Wednesday": "weekday",
        "Thursday":  "weekday",
        "Friday":    "weekend",
        "Saturday":  "weekend",
        "Sunday":    "weekday",
        }
    
    # weekday_weekend = WeekdayWeekendDefinition(**weekday_weekend_def)
    # weekday_weekend = WeekdayWeekendDefinition()
    # print(weekday_weekend.model_dump_json())

    # Test DataSettings
    settings = Data_Settings(
        agg_type="median",
        season=season_dict, 
        weekday_weekend=weekday_weekend_dict,
    )
    print(settings.model_dump_json())
    print(settings.SEASON._NUM_DICT)
    print(settings.SEASON._ORDER)