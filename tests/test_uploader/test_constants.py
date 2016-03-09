from eemeter.uploader import constants
from unittest import TestCase

class ConstantsTestCase(TestCase):

    def test_STANDARD_PROJECT_ATTRIBUTE_KEYS(self):
        for column_name, data in constants.STANDARD_PROJECT_ATTRIBUTE_KEYS.items():
            assert "name" in data
            assert "display_name" in data
            assert data["data_type"] in ["FLOAT", "INTEGER", "DATE",
                                         "DATETIME", "BOOLEAN", "CHAR"]

    def test_STANDARD_PROJECT_DATA_COLUMN_NAMES(self):
        assert len(constants.STANDARD_PROJECT_DATA_COLUMN_NAMES) == 9
