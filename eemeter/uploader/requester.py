import requests
import os

class Requester(object):

    def __init__(self, datastore_url=None, token=None, extra_headers={}):
        self.token = self.get_token(token)
        self.headers = self.get_headers(extra_headers)
        self.url = self.get_url(datastore_url)

    def get_token(self, token):
        if token is None:
            return os.environ["DATASTORE_ACCESS_TOKEN"]
        else:
            return token

    def get_headers(self, extra_headers):
        headers = {'Authorization': "Bearer " + self.token}
        headers.update(extra_headers)
        return headers

    def get_url(self, datastore_url):
        """Get the datastore URL
        """
        if datastore_url is None:
            datastore_url = os.environ["DATASTORE_URL"]

        if datastore_url[:4] != "http":
            message = (
                "Datastore url '{}' is missing protocol."
                " Please include http or https."
            ).format(datastore_url)
            raise ValueError(message)

        if datastore_url[-1] == "/":
            return datastore_url [:-1]
        else:
            return datastore_url

    def get(self, resource):
        """Makes a get request to the resource."""
        r = requests.get(self.url + '/api/v1/' + resource,
                         headers=self.headers)
        return r

    def post(self, resource, data):
        """Makes a post request to the resource, sending the data."""
        r = requests.post(self.url + '/api/v1/' + resource,
                          json=data,
                          headers=self.headers)
        return r
