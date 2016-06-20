from contextlib import contextmanager
from functools import wraps


class Collector(object):

    def __init__(self):
        self.items = {}

    @contextmanager
    def collect(self, key):
        new_items = {}
        yield new_items
        self.items[key] = new_items


def collects():
    def collects_decorator(func):
        @wraps(func)
        def func_wrapper(collector, *args, **kwargs):
            result, collectables = func(*args, **kwargs)
            collector.update(collectables)
            return result
        return func_wrapper
    return collects_decorator
