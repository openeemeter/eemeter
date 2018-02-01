import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

# Version
VERSION = (1, 3, 3)


def get_version():
    return '{}.{}.{}'.format(VERSION[0], VERSION[1], VERSION[2])
