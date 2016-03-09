from unittest import TestCase

from eemeter.uploader.uploaders import ProjectUploader
from eemeter.uploader.uploaders import ProjectAttributeUploader
from eemeter.uploader.uploaders import ProjectAttributeKeyUploader
from eemeter.uploader.uploaders import ConsumptionMetadataUploader

class TestProjectUploader(TestCase):

    @classmethod
    def setUp(self):
        self.record = {
            "project_id": "test_project_id",
            "baseline_period_end": "2014-05-27",
            "baseline_period_start": "2013-08-27",
            "reporting_period_start": "2014-08-25",
            "reporting_period_end": "2016-02-11",
            "latitude": 41.26996057364848,
            "longitude": -95.97935449486408,
            "zipcode": "68111",
            "weather_station": "725500",
            "predicted_electricity_savings": -1558.1,
            "predicted_natural_gas_savings": -43.2,
            "project_cost": 6592,
        }
        requester = Requester("https://example.com/", "TOKEN")
        self.uploader = ProjectUploader(requester)
