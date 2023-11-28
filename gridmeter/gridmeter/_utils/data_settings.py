"""
module defines settings to be used for individual meter matching
"""

from __future__ import annotations

import pydantic

from typing import (
    Union,
    Literal,
)

import gridmeter._utils.const as _const
from gridmeter._utils.base_settings import BaseSettings


# Note: Options list order defines how seasons will be orderd in the loadshape
class SeasonDefinition(BaseSettings):
    JANUARY: str = pydantic.Field(default="winter", validate_default=True)
    FEBRUARY: str = pydantic.Field(default="winter", validate_default=True)
    MARCH: str = pydantic.Field(default="shoulder", validate_default=True)
    APRIL: str = pydantic.Field(default="shoulder", validate_default=True)
    MAY: str = pydantic.Field(default="shoulder", validate_default=True)
    JUNE: str = pydantic.Field(default="summer", validate_default=True)
    JULY: str = pydantic.Field(default="summer", validate_default=True)
    AUGUST: str = pydantic.Field(default="summer", validate_default=True)
    SEPTEMBER: str = pydantic.Field(default="summer", validate_default=True)
    OCTOBER: str = pydantic.Field(default="shoulder", validate_default=True)
    NOVEMBER: str = pydantic.Field(default="winter", validate_default=True)
    DECEMBER: str = pydantic.Field(default="winter", validate_default=True)

    OPTIONS: list[str] = pydantic.Field(default=["summer", "shoulder", "winter"], validate_default=True)

    """Set dictionaries of seasons"""
    @pydantic.model_validator(mode="after")
    def set_numeric_dict(self) -> SeasonDefinition:
        season_dict = {}
        for month, num in _const.season_num.items():
            val = getattr(self, month.upper())
            if val not in self.OPTIONS:
                raise ValueError(f"SeasonDefinition: {val} is not a valid option. Valid options are {self.OPTIONS}")
            
            season_dict[num] = val
        
        self._NUM_DICT = season_dict
        self._ORDER = {val: i for i, val in enumerate(self.OPTIONS)}

        return self


class WeekdayWeekendDefinition(BaseSettings):
    MONDAY: str = pydantic.Field(default="weekday", validate_default=True)
    TUESDAY: str = pydantic.Field(default="weekday", validate_default=True)
    WEDNESDAY: str = pydantic.Field(default="weekday", validate_default=True)
    THURSDAY: str = pydantic.Field(default="weekday", validate_default=True)
    FRIDAY: str = pydantic.Field(default="weekday", validate_default=True)
    SATURDAY: str = pydantic.Field(default="weekend", validate_default=True)
    SUNDAY: str = pydantic.Field(default="weekend", validate_default=True)

    OPTIONS: list[str] = pydantic.Field(default=["weekday", "weekend"], validate_default=True)

    """Set dictionaries of weekday/weekend"""
    @pydantic.model_validator(mode="after")
    def set_numeric_dict(self) -> WeekdayWeekendDefinition:
        weekday_dict = {}
        for day, num in _const.weekday_num.items():
            val = getattr(self, day.upper())
            if val not in self.OPTIONS:
                raise ValueError(f"WeekdayWeekendDefinition: {val} is not a valid option. Valid options are {self.OPTIONS}")
            
            weekday_dict[num] = val
        
        self._NUM_DICT = weekday_dict
        self._ORDER = {val: i for i, val in enumerate(self.OPTIONS)}

        return self
    

class DataSettings(BaseSettings):
    """aggregation type for the loadshape"""
    AGG_TYPE: str = pydantic.Field(default=_const.AggType.MEAN, validate_default=True)
    
    """type of loadshape to be used"""
    LOADSHAPE_TYPE: str = pydantic.Field(default=_const.LoadshapeType.MODELED, validate_default=True)

    """time period to be used for the loadshape"""
    TIME_PERIOD: str = pydantic.Field(default=_const.TimePeriod.SEASON_HOURLY_DAY_OF_WEEK, validate_default=True)

    """interpolate missing values"""
    INTERPOLATE_MISSING: bool = pydantic.Field(default=True, validate_default=True)

    """minimum percentage of data required for a meter to be included"""
    MIN_DATA_PCT_REQUIRED: Literal[0.8]

    """season definition to be used for the loadshape"""
    SEASON: Union[dict, SeasonDefinition] = pydantic.Field(default=_const.default_season_def, validate_default=True)

    """weekday/weekend definition to be used for the loadshape"""
    WEEKDAY_WEEKEND: Union[dict, WeekdayWeekendDefinition] = pydantic.Field(default=_const.default_weekday_weekend_def, validate_default=True)


    """set season and weekday_weekend classes with given dictionaries"""
    @pydantic.model_validator(mode="after")
    def _set_nested_classes(self):
        if isinstance(self.SEASON, dict):
            self.SEASON = SeasonDefinition(**self.SEASON)

        if isinstance(self.WEEKDAY_WEEKEND, dict):
            self.WEEKDAY_WEEKEND = WeekdayWeekendDefinition(**self.WEEKDAY_WEEKEND)

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
    settings = DataSettings(
        AGG_TYPE="median",
        season=season_dict, 
        weekday_weekend=weekday_weekend_dict,
    )
    print(settings.model_dump_json())
    print(settings.SEASON._NUM_DICT)
    print(settings.SEASON._ORDER)