__all__ = ("EEMeterWarning",)


class EEMeterWarning(object):
    """ An object representing a warning and data associated with it.

    Attributes
    ----------
    qualified_name : :any:`str`
        Qualified name, e.g., `'eemeter.method_abc.missing_data'`.
    description : :any:`str`
        Prose describing the nature of the warning.
    data : :any:`dict`
        Data that reproducibly shows why the warning was issued. Data should
        be JSON serializable.
    """

    def __init__(self, qualified_name, description, data):
        self.qualified_name = qualified_name
        self.description = description
        self.data = data

    def __repr__(self):
        return "EEMeterWarning(qualified_name={})".format(self.qualified_name)

    def json(self):
        """ Return a JSON-serializable representation of this result.

        The output of this function can be converted to a serialized string
        with :any:`json.dumps`.
        """
        return {
            "qualified_name": self.qualified_name,
            "description": self.description,
            "data": self.data,
        }
