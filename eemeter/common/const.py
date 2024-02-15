from __future__ import annotations

from enum import Enum


default_season_def = {
    "options": ["summer", "shoulder", "winter"],
    "January": "winter",
    "February": "winter",
    "March": "shoulder",
    "April": "shoulder",
    "May": "shoulder",
    "June": "summer",
    "July": "summer",
    "August": "summer",
    "September": "summer",
    "October": "shoulder",
    "November": "winter",
    "December": "winter",
}


default_weekday_weekend_def = {
    "options": ["weekday", "weekend"],
    "Monday": "weekday",
    "Tuesday": "weekday",
    "Wednesday": "weekday",
    "Thursday": "weekday",
    "Friday": "weekday",
    "Saturday": "weekend",
    "Sunday": "weekend",
}


column_names = {
    "METER": "meter_value",
    "TEMPERATURE": "temperature_mean",
}


class TutorialDataChoice(str, Enum):
    """
    Options for the tutorial data to load.
    """

    FEATURES = "features"
    SEASONAL_HOUR_DAY_WEEK_LOADSHAPE = "seasonal_hourly_day_of_week_loadshape".replace("_", "")
    SEASONAL_DAY_WEEK_LOADSHAPE = "seasonal_day_of_week_loadshape".replace("_", "")
    MONTH_LOADSHAPE = "month_loadshape".replace("_", "")
    HOURLY_DATA = "hourly_data".replace("_", "")
