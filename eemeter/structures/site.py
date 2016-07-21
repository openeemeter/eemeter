import re


class ZIPCodeSite(object):
    ''' ZIP-code-based site location descriptor.

    Parameters
    ----------
    zipcode : str
        A five-digit zipcode identifier.
    '''

    def __init__(self, zipcode):
        self.zipcode = self._validate(zipcode)

    def __repr__(self):
        return 'ZIPCodeSite("{}")'.format(self.zipcode)

    def _validate(self, zipcode):

        zipcode_str = str(zipcode)

        result = re.match(r'^\d{5}$', zipcode_str)

        if result is None:
            message = (
                'ZIP code not valid as given: `zipcode={}`.'
                ' Should be given as a 5-digit string, such as "01234"'
                .format(zipcode)
            )
            raise ValueError(message)
        return zipcode_str
