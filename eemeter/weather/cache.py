from .base import WeatherSourceBase

import os
import json

import pandas as pd


class CachedWeatherSourceBase(WeatherSourceBase):

    def __init__(self, station, **kwargs):
        super(CachedWeatherSourceBase, self).__init__(station)

        self.prepare_cache(**kwargs)
        self.load_from_cache()

    def prepare_cache(self, **kwargs):
        message = 'The `prepare_cache()` method must be implemented'
        raise NotImplementedError(message)

    def save_to_cache(self):
        message = 'The `save_to_cache()` method must be implemented'
        raise NotImplementedError(message)

    def load_from_cache(self):
        message = 'The `load_from_cache()` method must be implemented'
        raise NotImplementedError(message)

    def clear_cache(self):
        message = 'The `clear_cache()` method must be implemented'
        raise NotImplementedError(message)

    def _get_cache_directory(self):
        """ Returns a directory to be used for caching.
        """
        directory = os.environ.get("EEMETER_WEATHER_CACHE_DIRECTORY",
                                   os.path.expanduser('~/.eemeter/cache'))
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory


class FileCachedWeatherSourceBase(CachedWeatherSourceBase):

    cache_date_format = None
    cache_filename_format = None
    freq = None

    def prepare_cache(self, cache_directory=None, cache_filename=None):
        if cache_filename is None:
            self.cache_filename = self._get_cache_filename(cache_directory)
        else:
            self.cache_filename = cache_filename

    def _get_cache_filename(self, cache_directory=None):
        if cache_directory is None:
            cache_directory = self._get_cache_directory()
        filename = self.cache_filename_format.format(self.station)
        return os.path.join(cache_directory, filename)

    def save_to_cache(self):
        data = [
            [
                d.strftime(self.cache_date_format), t
                if pd.notnull(t) else None
            ]
            for d, t in self.tempC.iteritems()
        ]
        with open(self.cache_filename, 'w') as f:
            json.dump(data, f)

    def load_from_cache(self):
        try:
            with open(self.cache_filename, 'r') as f:
                data = json.load(f)
        except IOError:
            return
        except ValueError:  # Corrupted json file
            self.clear_cache()
            return
        index = pd.to_datetime([d[0] for d in data],
                               format=self.cache_date_format, utc=True)
        values = [d[1] for d in data]

        # changed for pandas > 0.18
        self.tempC = pd.Series(values, index=index, dtype=float) \
            .sort_index().resample(self.freq).mean()

    def clear_cache(self):
        try:
            os.remove(self.cache_filename)
        except OSError:
            pass
