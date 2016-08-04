import warnings


class Intervention(object):
    ''' Represents an intervention with a start date, and maybe an end date.
    Multiple interventions can be composed within a project.

    Parameters
    ----------
    start_date : datetime.datetime
        Must be timezone aware
    end_date : datetime.datetime or None, default None
        Must be timezone aware. If None, intervention is assumed to be ongoing.
    '''

    def __init__(self, start_date, end_date=None):
        self.start_date = self._validate_start_date(start_date)
        self.end_date = self._validate_end_date(end_date)

    def __repr__(self):
        return (
            "Intervention(start_date={}, end_date={})"
            .format(self.start_date, self.end_date)
        )

    def _validate_start_date(self, dt):
        if dt is None:
            message = 'Intervention `start_date` cannot be None.'
            raise ValueError(message)

        if not self._is_tz_aware(dt):
            message = 'Given datetime is not tz-aware: {}'.format(dt)
            raise ValueError(message)
        return dt

    def _validate_end_date(self, dt):

        if dt is None:
            return None

        if not self._is_tz_aware(dt):
            message = 'Given datetime is not tz-aware: {}'.format(dt)
            raise ValueError(message)

        if self.start_date > dt:
            message = (
                'Ignoring end_date because it is before start_date: '
                'start_date={} > end_date={}'
                .format(self.start_date, dt)
            )
            warnings.warn(message)
            return None

        return dt

    @staticmethod
    def _is_tz_aware(dt):
        return dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None
