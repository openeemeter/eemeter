"""
contains constants used for GRIDmeter
"""
from __future__ import annotations

from enum import Enum


class DistanceMetric(str, Enum):
    """
    what distance method to use
    """
    EUCLIDEAN = "euclidean"
    SEUCLIDEAN = "seuclidean"
    MANHATTAN = "manhattan"
    COSINE = "cosine"


class AggType(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"


"""data_settings constants"""
class LoadshapeType(str, Enum):
    OBSERVED = "observed"
    MODELED = "modeled"
    ERROR = "error"
    MODEL_ERROR = "error" # an alias for ERROR


class TimePeriod(str, Enum):
    HOUR = "hour"
    DAY_OF_WEEK = "day_of_week"
    WEEKDAY_WEEKEND = "weekday_weekend"
    MONTH = "month"
    SEASON_HOURLY_DAY_OF_WEEK = "season_hourly_day_of_week"
    SEASON_WEEKDAY_WEEKEND = "season_weekday_weekend"


default_season_def = {
    "options":   ["summer", "shoulder", "winter"],
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
    "December":  "winter"
}


default_weekday_weekend_def = {
    "options":   ["weekday", "weekend"],
    "Monday":    "weekday",
    "Tuesday":   "weekday",
    "Wednesday": "weekday",
    "Thursday":  "weekday",
    "Friday":    "weekday",
    "Saturday":  "weekend",
    "Sunday":    "weekend",
}


season_num = {
    "january":   1, 
    "february":  2, 
    "march":     3, 
    "april":     4, 
    "may":       5, 
    "june":      6, 
    "july":      7, 
    "august":    8, 
    "september": 9, 
    "october":   10, 
    "november":  11, 
    "december":  12,
}


weekday_num = {
    "monday":    0,
    "tuesday":   1,
    "wednesday": 2,
    "thursday":  3,
    "friday":    4,
    "saturday":  5,
    "sunday":    6,
}