from contextlib import contextmanager, closing
import logging
import six


class LogCollector(object):

    def __init__(self):
        self.items = {}

    @contextmanager
    def collect_logs(self, key):
        # purposefully doesn't use global named getLogger b/c there could be
        # multiple instances
        logger = logging.Logger("temp")
        logger.setLevel(logging.DEBUG)

        with closing(six.StringIO()) as log_stream:
            handler = logging.StreamHandler(log_stream)
            handler.setLevel(logging.DEBUG)

            formatter = logging.Formatter(
                    '%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)

            logger.addHandler(handler)

            yield logger

            handler.flush()

            self.items[key] = log_stream.getvalue().splitlines()
