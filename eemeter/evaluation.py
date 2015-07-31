class Period:
    """Represents a period of time with a start and an end.

    Parameters
    ----------

    start : datetime.datetime or None, default None
        The start date of the Period.
    end : datetime.datetime or None, default None
        The end date of the Period.
    """
    def __init__(self, start=None, end=None, name=None):
        self.start = start
        self.end = end
        self.name = name

    @property
    def timedelta(self):
        """Property representing the timedelta between the start and end
        datetimes.
        """
        if self.closed:
            return self.end - self.start
        return None

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "Period({} - {})".format(self.start, self.end)

    def __contains__(self,key):
        if self.start is None and self.end is None:
            return True
        elif self.start is None and self.end is not None:
            if key.__class__ == Period:
                if key.end is None:
                    return False
                else:
                    return key.end <= self.end
            else:
                return key <= self.end
        elif self.end is None:
            if key.__class__ == Period:
                if key.start is None:
                    return False
                else:
                    return self.start <= key.start
            else:
                return self.start <= key
        else:
            if key.__class__ == Period:
                return self.start <= key.start and key.end <= self.end
            else:
                return self.start <= key <= self.end

    @property
    def closed(self):
        """True if the period interval has both a start and an end point.
        """
        return self.start is not None and self.end is not None
