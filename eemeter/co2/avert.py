import logging

from .clients import AVERTClient
from .cache import SqlCO2Store

logger = logging.getLogger(__name__)


class AVERTSource(object):

    client = AVERTClient()

    def __init__(self, year, region, cache_url=None):
        self.year = year
        self.region = region
        self.co2_store = SqlCO2Store(cache_url)
        self._check_for_data()

    def _check_for_data(self):
        if not self.co2_store.key_exists(self.year, self.region):
            co2_by_load, load_by_hour = self.client.read_rdf_file(
                self.year, self.region)
            if len(co2_by_load) > 0 and len(load_by_hour) > 0:
                self.co2_store.save_json(self.year, self.region,
                                         co2_by_load, load_by_hour)

    def get_co2_by_load(self):
        return self.co2_store.retrieve_co2_by_load(self.year, self.region)

    def get_load_by_hour(self):
        return self.co2_store.retrieve_load_by_hour(self.year, self.region)
