"""Uploaders

Example
-------
How to use an uploader class::

    from oeem_uploader import Requester

    requester = Requester("https://datastore.openeemeter.org/", "TOKEN")
    uploader = ProjectUploader(requester)

    data = {
        "project_id": "MY_PROJECT_ID",
        ...
    }

    uploader.sync(data)

"""

from . import constants


class BaseUploader(object):

    item_name = None
    many = False

    def __init__(self, requester, verbose=True):
        self.requester = requester
        self.verbose = verbose

    def sync(self, data):
        response_data, created = self.get_or_create(data)

        if created:
            return response_data

        if len(response_data) > 1 and not self.many:
            message = (
                "More than one response received:\n {}"
                .format(response_data)
            )
            raise ValueError(message)

        if self.should_update(data, response_data):
            return self.update(data)

        if self.many:
            return response_data
        else:
            return response_data[0]

    def get_or_create(self, data):
        descriptor = self.get_descriptor(data)
        urls = self.get_urls(data)

        read_response = self.requester.get(urls["read"])

        if read_response.status_code != 200:
            message = "Read error GET ({}): {}\n{}".format(
                    read_response.status_code, urls["read"], read_response.text)
            raise ValueError(message)

        read_response_data = read_response.json()
        pks = [item["id"] for item in read_response_data]

        if pks == []:
            create_response = self.requester.post(urls["create"], data)

            if create_response.status_code != 201:
                message = "Create error POST ({}): {}\n{}\n{}".format(
                        create_response.status_code, urls["create"], data, create_response.text)
                raise ValueError(message)

            create_response_data = create_response.json()
            if self.many:
                pks = [response_data["id"] for response_data in create_response_data]
            else:
                pk = create_response_data["id"]

            if self.verbose:

                if self.many:
                    print("Created {} ({}, pks={})".format(self.item_name,
                                                          descriptor, pks))
                else:
                    print("Created {} ({}, pk={})".format(self.item_name,
                                                          descriptor, pk))

            return create_response_data, True

        else:
            if len(pks) > 1 and not self.many:
                message = (
                    "Found multiple {} instances ({}) for {}"
                    .format(self.item_name, pks, descriptor)
                )
                warnings.warn(message)

            if self.verbose:
                print("Existing {} ({}, pks={})".format(self.item_name,
                                                        descriptor, pks))

            return read_response_data, False

    def get_descriptor(self, data):
        raise NotImplementedError

    def get_urls(self, data):
        return {
            "create": self.get_create_url(data),
            "read": self.get_read_url(data),
        }

    def get_read_url(self, data):
        raise NotImplementedError

    def get_create_url(self, data):
        raise NotImplementedError


    def should_update(self, data, response_data):
        """
        Returns True/False if a project should
        HTTP update or not.
        """
        # just log for now
        print("Should update? N".format(data, response_data))
        return False

    def update(self, data):
        print("Updating: {}".format(data))


class ProjectAttributeKeyUploader(BaseUploader):

    item_name = "ProjectAttributeKey"

    def get_descriptor(self, data):
        return data["name"]

    def get_read_url(self, data):
        return (
            constants.PROJECT_ATTRIBUTE_KEY_URL +
            "?name={}".format(data["name"])
        )

    def get_create_url(self, data):
        return constants.PROJECT_ATTRIBUTE_KEY_URL


class ProjectUploader(BaseUploader):

    item_name = "Project"

    def get_descriptor(self, data):
        return data["project_id"]

    def get_read_url(self, data):
        return constants.PROJECT_URL + "?project_id={}".format(data["project_id"])

    def get_create_url(self, data):
        return constants.PROJECT_URL


class ProjectAttributeUploader(BaseUploader):

    item_name = "ProjectAttribute"

    def get_descriptor(self, data):
        return data["key"]

    def get_read_url(self, data):
        return (
            constants.PROJECT_ATTRIBUTE_URL +
            "?project={}&key={}".format(data["project"], data["key"])
        )

    def get_create_url(self, data):
        return constants.PROJECT_ATTRIBUTE_URL


class ConsumptionMetadataUploader(BaseUploader):

    item_name = "ConsumptionMetadata"

    def get_descriptor(self, data):
        return "{}/{}".format(data["project"], data["fuel_type"])

    def get_read_url(self, data):
        return (
            constants.CONSUMPTION_METADATA_URL +
            "?project={}&fuel_type={}&summary=True"
            .format(data["project"], data["fuel_type"])
        )

    def get_create_url(self, data):
        return constants.CONSUMPTION_METADATA_URL + "?summary=True"


class ConsumptionRecordUploader(BaseUploader):

    item_name = "ConsumptionRecord"
    many = True

    def get_descriptor(self, data):
        return "n={}".format(len(data))

    def get_read_url(self, data):
        return (
            constants.CONSUMPTION_RECORD_URL +
            "?metadata={}"
            .format(data[0]["metadata"])
        )

    def get_create_url(self, data):
        return constants.CONSUMPTION_RECORD_URL
