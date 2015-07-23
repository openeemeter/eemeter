class Period:
    """Represents a period of time with a start and an end.

    Parameters
    ----------

    start : datetime.datetime or None, default None
        The start date of the Period.
    end : datetime.datetime or None, default None
        The end date of the Period.
    """
    def __init__(self, start=None, end=None):
        self.start = start
        self.end = end

    @property
    def timedelta(self):
        """Property representing the timedelta between the start and end
        datetimes.
        """
        if self.start is None or self.end is None:
            return None
        return self.end - self.start

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "Period({} - {})".format(self.start, self.end)
