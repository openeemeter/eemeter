import pandas as pd
from pathlib import Path
import numpy as np

def clean_list(lst):
    return [x.lower().replace(" ", "_") for x in lst]

class NREL_Weather_API: #TODO: reload data for all years
    api_key = "PBjC0msokfcSXP3SD2fqT01IShnO9ZMtbNYx4WNZ"  # get your own key from https://developer.nrel.gov/signup/  #Required
    name = "Armin+Aligholian"  # required
    email = "armin@recurve.com"  # required
    interval = "60"  # required

    attributes = "ghi,dhi,dni,wind_speed,air_temperature,cloud_type,dew_point,clearsky_dhi,clearsky_dni,clearsky_ghi"  # not required
    leap_year = "false"  # not required
    utc = "false"  # not required
    reason_for_use = "beta+testing"  # not required
    your_affiliation = "Recurve"  # not required
    mailing_list = "false"  # not required

    # cache = Path("/app/.recurve_cache/data/MCE/MCE_weather_stations")
    cache = Path("/app/.recurve_cache/data/MCE/Weather_stations")

    use_cache = True

    round_minutes_method = "floor"  # [None, floor, ceil, round]

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.cache.mkdir(parents=True, exist_ok=True)
    
    def get_data(self, lat, lon, years=[2017, 2021]):
        data_path = self.cache / f"{lat}_{lon}.pkl"
        if data_path.exists() and self.use_cache:
            df = pd.read_pickle(data_path)

        else:
            years = list(range(min(years), max(years)+1))

            df = self.query_API(lat, lon, years)

            df.columns = clean_list(df.columns)

            if self.round_minutes_method == "floor":
                df["datetime"] = df["datetime"].dt.floor("H")
            elif self.round_minutes_method == "ceil":
                df["datetime"] = df["datetime"].dt.ceil("H")
            elif self.round_minutes_method == "round":
                df["datetime"] = df["datetime"].dt.round("H")

            df = df.set_index("datetime")

            if self.use_cache:
                df.to_pickle(data_path)

        return df

    def query_API(self, lat, lon, years):
        leap_year = self.leap_year
        interval = self.interval
        utc = self.utc
        api_key = self.api_key
        name = self.name
        email = self.email

        year_df = []
        for year in years:
            year = str(year)

            url = self._generate_url(
                lat, lon, year, leap_year, interval, utc, api_key, name, email
            )
            df = pd.read_csv(url, skiprows=2)

            # Set the time index in the pandas dataframe:
            # set datetime using the year, month, day, and hour
            df["datetime"] = pd.to_datetime(
                df[["Year", "Month", "Day", "Hour", "Minute"]]
            )

            df = df.drop(columns=["Year", "Month", "Day", "Hour", "Minute"])
            df = df.dropna()

            year_df.append(df)

        # merge the dataframes for different years
        df = pd.concat(year_df, axis=0)

        return df

    def _generate_url(
        self, lat, lon, year, leap_year, interval, utc, api_key, name, email
    ):
        query = f"?wkt=POINT({lon}%20{lat})&names={year}&interval={interval}&api_key={api_key}&full_name={name}&email={email}&utc={utc}"

        if year == "2021":
            # details: https://developer.nrel.gov/docs/solar/nsrdb/psm3-2-2-download/
            url = f"https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-2-2-download.csv{query}"

        elif year in [str(i) for i in range(1998, 2021)]:
            # details: https://developer.nrel.gov/docs/solar/nsrdb/psm3-download/
            url = f"https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv{query}"

        else:
            print("Year must be between 1998 and 2021")
            url = None

        return url

class HourlyData:
    def __init__(
            self, df: pd.DataFrame, 
            **kwargs: dict # consider solar data
        ):
        """
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if df is None:
            raise ValueError("df cannot be None")
        if not isinstance(kwargs, dict):
            raise TypeError("kwargs must be a dictionary")
        
        self.df = df
        self.kwargs = kwargs
        if "outputs" in self.kwargs:
            self.outputs = kwargs["outputs"]
        else:
            self.outputs = ["temperature", "observed"]
        
        self._prepare_dataframe()
 
    def _prepare_dataframe(self):

        def check_datetime(df):
            #get all the columns with datetime type #TODO: check if this is the best way to do this
            datetime_columns = df.select_dtypes(include=[np.datetime64]).columns
            # check if datetime is in the columns
            if "datetime" in df.columns:
                pass
            elif "datetime" in df.index.names:
                df['datetime'] = df.index
                df = df.reset_index(drop=True)
            elif "start_local" in df.columns:
                df['datetime'] = df['start_local']
                df = df.drop(columns=["start_local"])
            elif len(datetime_columns) > 0:
                df['datetime'] = df[datetime_columns[0]]
                df = df.drop(columns=[datetime_columns[0]])
            else:
                raise ValueError("datetime column not found")
            
            #reset index to ensure datetime is not the index
            df = df.reset_index()
            return df
        
        def get_contiguous_datetime(df):
            # get earliest datetime and latest datetime
            # make earliest start at 0 and latest end at 23, this ensures full days
            earliest_datetime = df["datetime"].min().replace(hour=0, minute=0, second=0, microsecond=0)
            latest_datetime = df["datetime"].max().replace(hour=23, minute=0, second=0, microsecond=0)

            # create a new index with all the hours between the earliest and latest datetime
            complete_dt = pd.date_range(start=earliest_datetime, end=latest_datetime, freq='H').to_frame(index=False, name="datetime")

            # merge meter data with complete_dt
            df = complete_dt.merge(df, on="datetime", how="left")

            return df

        def remove_duplicate_datetime(df):
            if "observed" in df.columns:
                # find duplicate datetime values and remove if nan
                duplicate_dt_mask = df.duplicated(subset="datetime", keep=False)
                observed_nan_mask = df['observed'].isna()
                df = df[~(duplicate_dt_mask & observed_nan_mask)]

                # if duplicated and observed is not nan, keep the largest abs(value)
                df["abs_observed"] = df["observed"].abs()
                df = df.sort_values(by=["datetime", "abs_observed"], ascending=[True, False])
                df = df.drop_duplicates(subset="datetime", keep="first")
                df = df.drop(columns=["abs_observed"])

            else:
                # TODO what if there is no observed column? Could have dup datetime with different temperatures
                df = df.drop_duplicates(subset="datetime", keep="first")

            return df
        
        self.df = check_datetime(self.df)

        # save the original datetime column
        self.datetime_original = self.df["datetime"]

        # fill in missing datetimes
        self.df = get_contiguous_datetime(self.df)

        # remove duplicate datetime values
        self.df = remove_duplicate_datetime(self.df)
        
        self.df = self.df.set_index("datetime")

        if "solar" in self.kwargs: 
            if "metadata" in self.kwargs:
                if "station_latitude" and "station_longitude" in self.kwargs["metadata"]:
                    self.station_latitude = self.kwargs["metadata"]["station_latitude"]
                    self.station_longitude = self.kwargs["metadata"]["station_longitude"]
                else: #TODO: add eeweather to get the station_latitude and station_longitude
                            # if just have lat and lon of the meter
                    raise ValueError("station_latitude and station_longitude are not in metadata")
            self._add_solar_data()

        # interpolate temperature and observed values
        self.df = self._interpolate()

        self.df = self.df[self.outputs]
    
    def _interpolate(self):
        # make column of interpolated boolean if any observed or temperature is nan
        #check if in each row of the columns in output has nan values, the interpolated column will be true
        output_columns = ['temperature', 'observed']
        if "ghi" in self.outputs:
            output_columns += ['ghi']
        
        self.df['interpolated'] = self.df[output_columns].isna().any(axis=1)
        if "interpolated" not in self.outputs:
            self.outputs += ['interpolated']

        # TODO: We need to think about this interpolation step more. Maybe do custom based on datetime, (temp if observed), and surrounding weeks?
        for column in output_columns:
            self.df[column] = self.df[column].interpolate()

        return self.df
    
    def _get_location_solar_data(self):
        # get unique years from sdf from 'start' column
        # years = np.unique(self.df.index.year)
        years = [2017, 2021]#TODO: get all years saved them then call as needed 

        lat = self.station_latitude
        lon = self.station_longitude

        nrel_weather = NREL_Weather_API(use_cache=True)
        solar_df = nrel_weather.get_data(lat, lon, years)
        # change the temperature column name
        solar_df = solar_df.rename(columns={"temperature": "temp_NRSDB"})

        # convert to kWh
        for feature in [
            "ghi",
            "dni",
            "dhi",
            "clearsky_dhi",
            "clearsky_dni",
            "clearsky_ghi",
        ]:
            if feature in solar_df.columns:
                solar_df[feature] /= 1000

        return solar_df

    def _add_solar_data(self, T_type="NOAA"): #TODO: should we consider NRSDB temp at all?

        # get solar data
        sdf = self._get_location_solar_data()

        # assign temperature column
        if T_type == "NOAA":
            pass
        elif T_type == "NRSDB":
            self.df["temperature"] = self.df["temp_NRSDB"]
        else:
            raise ValueError("T_type must be either NOAA or NRSDB")

        self.sdf = sdf.drop(columns=["temp_NRSDB"])

        # merge site data with solar data on index
        self.df = self.df.merge(sdf, left_index=True, right_index=True, how="left")

