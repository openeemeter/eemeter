from unittest import TestCase

from eemeter.uploader import api

import pandas as pd
from datetime import datetime
import pytz

class APITestCase(TestCase):

    def setUp(self):
        self.minimal_project_df = self._minimal_project_df_fixture()
        self.minimal_consumption_df = self._minimal_consumption_df_fixture()

    def _minimal_project_df_fixture(self):
        data = [
            {
                "project_id": "ID_1",
                "zipcode": "01234",
                "weather_station": "012345",
                "latitude": 89.0,
                "longitude": -42.0,
                "baseline_period_end": datetime(2015, 1, 1),
                "reporting_period_start": datetime(2015, 2, 1),
            },
        ]
        df = pd.DataFrame(data)
        return df

    def _minimal_consumption_df_fixture(self):
        data = [
            {
                "project_id": "ID_1",
                "start": datetime(2015, 1, 1),
                "end": datetime(2015, 1, 2),
                "fuel_type": "electricity",
                "unit_name": "kWh",
                "value": 0,
                "estimated": True,
            },
        ]
        df = pd.DataFrame(data)
        return df

    def test_get_project_attribute_keys_data(self):
        project_attribute_keys_data = \
                api._get_project_attribute_keys_data(self.minimal_project_df)
        assert project_attribute_keys_data == []

    def test_get_project_data(self):
        projects_data = api._get_project_data(self.minimal_project_df, [])
        project_data, project_attributes_data = next(projects_data)
        assert project_data["project_id"] == "ID_1"
        assert project_data["zipcode"] == "01234"
        assert project_data["weather_station"] == "012345"
        assert project_data["latitude"] == 89.0
        assert project_data["longitude"] == -42.0
        assert project_data["baseline_period_end"] == "2015-01-01T00:00:00+0000"
        assert project_data["reporting_period_start"] == "2015-02-01T00:00:00+0000"
        assert project_attributes_data == []

    def test_get_consumption_records_data(self):
        consumptions_data = api._get_consumption_data(self.minimal_consumption_df)
        consumption_metadata, consumption_records = next(consumptions_data)
        assert consumption_metadata["fuel_type"] == "E"
        assert consumption_metadata["energy_unit"] == "KWH"
        assert consumption_metadata["project_id"] == "ID_1"
        assert consumption_records[0]["value"] == 0.0
        assert consumption_records[0]["estimated"] == True
        assert consumption_records[0]["start"] == "2015-01-01T00:00:00+0000"
