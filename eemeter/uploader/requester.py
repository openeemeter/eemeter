import requests
import os

class Requester(object):
    """Makes datastore API requests.

    For example::

        requester = Requester("https://datastore.openeemeter.org", "MY_TOKEN")

        projects = requester.get("projects/?summary=True")

    Parameters
    ----------
    datastore_url : str, default: None
        URL of target datastore. If None, looks for `DATASTORE_URL`
        environment variable.
    token : str, default: None
        Access token for target datastore If None, looks for
        `DATASTORE_ACCESS_TOKEN` environment variable.
    extra_headers : str, default: {}
        Extra headers for requests.
    """

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
        """Makes a get request to the resource.

        Parameters
        ----------
        resource : str
            Resource URIs should not include the datastore url or the
            "/api/v1/" prefix, as these are automatically added.
        """
        r = requests.get(self.url + '/api/v1/' + resource,
                         headers=self.headers)
        return r

    def post(self, resource, data):
        """Makes a post request to the resource, sending the data.

        Parameters
        ----------
        resource : str
            Resource URIs should not include the datastore url or the
            "/api/v1/" prefix, as these are automatically added.
        data : object (json formattable)
            A python object that will be POSTed as data after being converted
            to json.
        """
        r = requests.post(self.url + '/api/v1/' + resource,
                          json=data,
                          headers=self.headers)
        return r
