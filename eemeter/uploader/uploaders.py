from . import constants


class BaseUploader(object):

    item_name = None
    many = False

    def __init__(self, requester, verbose=True):
        self.requester = requester
        self.verbose = verbose

    def bulk_sync(self, items):
        # descriptor = self.get_descriptor(data)
        url = self.get_urls(items)['sync']

        read_response = self.requester.post(url, items)

        if read_response.status_code != 200:
            message = "Read error GET ({}): {}\n{}".format(
                    read_response.status_code, url, read_response.text)
            raise ValueError(message)

        for item_status in read_response.json():
            print(item_status)



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
            "sync": self.get_sync_url(data),
        }

    def get_read_url(self, data):
        raise NotImplementedError

    def get_create_url(self, data):
        raise NotImplementedError

    def get_sync_url(self, data):
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
    """Upload project attribute keys

    Basic usage::

        from eemeter.uploader import Requester
        from eemeter.uploader import ProjectAttributeKeyUploader

        requester = Requester("https://datastore.openeemeter.org", "MYACCESSTOKEN")
        uploader = ProjectAttributeKeyUploader(requester)

        data = {
            "name": ..., # e.g. "project_cost"
            "display_name": ..., # e.g. "Project Cost"
            "data_type": ..., # one of "BOOLEAN", "CHAR", "FLOAT", "INTEGER", "DATE", "DATETIME"
        }

        uploader.sync(data)

    """

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

    def get_sync_url(self, data):
        return constants.CONSUMPTION_RECORD_SYNC_URL


class ProjectUploader(BaseUploader):
    """Upload projects

    Basic usage::

        from eemeter.uploader import Requester
        from eemeter.uploader import ProjectUploader

        requester = Requester("https://datastore.openeemeter.org", "MYACCESSTOKEN")
        uploader = ProjectUploader(requester)

        data = {
            "project_id": ..., # Must be unique
            "zipcode": ..., # 5 digit str, no ZIP+4
            "weather_station": ..., # 6 digit USAF ID
            "latitude": ...,
            "longitude": ...,
            "baseline_period_start": None, # Implied by consumption data range
            "baseline_period_end": ..., # ISO 8601 Combined date and time with timezone, e.g. 2016-01-01T00:00:00Z
            "reporting_period_start": ..., # ISO 8601 Combined date and time with timezone, e.g. 2016-01-01T00:00:00Z
            "reporting_period_end": None, # Implied by consumption data range
        }

        uploader.sync(data)

    """

    item_name = "Project"

    def get_descriptor(self, data):
        return data["project_id"]

    def get_read_url(self, data):
        return constants.PROJECT_URL + "?project_id={}".format(data["project_id"])

    def get_create_url(self, data):
        return constants.PROJECT_URL

    def get_sync_url(self, data):
        return constants.CONSUMPTION_RECORD_SYNC_URL


class ProjectAttributeUploader(BaseUploader):
    """Upload project attributes

    Basic usage::

        from eemeter.uploader import Requester
        from eemeter.uploader import ProjectAttributeUploader

        requester = Requester("https://datastore.openeemeter.org", "MYACCESSTOKEN")
        uploader = ProjectAttributeUploader(requester)

        data = {
            "project": ..., # primary key or project
            "key": ..., # primary key of project attribute key
            "boolean_value": ..., # in addition to "boolean_value", you could also use
                                  # "integer_value", "float_value", "date_value",
                                  # "datetime_value", or "char_value"; however, it
                                  # must match the "data_type" field of the project attribute key
        }

        uploader.sync(data)

    """

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

    def get_sync_url(self, data):
        return constants.CONSUMPTION_RECORD_SYNC_URL


class ConsumptionMetadataUploader(BaseUploader):
    """Upload consumption metadata

    Basic usage::

        from eemeter.uploader import Requester
        from eemeter.uploader import ConsumptionMetadataUploader

        requester = Requester("https://datastore.openeemeter.org", "MYACCESSTOKEN")
        uploader = ConsumptionMetadataUploader(requester)

        data = {
            "project": ..., # primary key of project
            "fuel_type": ..., # one of "E" or "NG"
            "energy_unit": ..., # one of "KWH" or "THM"
        }

        uploader.sync(data)

    """

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

    def get_sync_url(self, data):
        return constants.CONSUMPTION_RECORD_SYNC_URL


class ConsumptionRecordUploader(BaseUploader):
    """Upload consumption records (bulk)

    Basic usage::

        from eemeter.uploader import Requester
        from eemeter.uploader import ConsumptionRecordUploader

        requester = Requester("https://datastore.openeemeter.org", "MYACCESSTOKEN")
        uploader = ConsumptionRecordUploader(requester)

        data = [
            {
                "metadata": ..., # primary key of consumption metadata object
                "start": ..., # ISO 8601 Combined date and time with timezone, e.g. 2016-01-01T00:00:00Z
                "value": ..., # consumption until next record in units matching the consumption metadata energy_unit field, or None, if there is no end date
                "estimated": bool(estimated),
            },
            ... # more records
        ]

        uploader.sync(data)

    """

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

    def get_sync_url(self, data):
        return constants.CONSUMPTION_RECORD_SYNC_URL
