import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

# Version
VERSION = (1, 5, 0)


def get_version():
    return '{}.{}.{}'.format(VERSION[0], VERSION[1], VERSION[2])
