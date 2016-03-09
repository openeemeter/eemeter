from unittest import TestCase
from eemeter.uploader import Requester

class TestRequester(TestCase):

    @classmethod
    def setUp(self):
        self.requester = Requester("https://example.com", "TOKEN")

    def test_token(self):
        assert self.requester.token == "TOKEN"

    def test_headers(self):
        assert self.requester.headers == {'Authorization': "Bearer TOKEN"}

    def test_url(self):
        assert self.requester.url == "https://example.com"
