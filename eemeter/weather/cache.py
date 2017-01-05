import os
import json
from sqlalchemy import (
    create_engine,
    Table,
    MetaData,
    Column,
    Integer,
    String,
    DateTime,
)
from sqlalchemy.sql import select, func


class SqlJSONStore(object):

    def __init__(self, url=None):
        self._prepare_db(url)

    def __repr__(self):
        return 'SqlJSONStore("{}")'.format(self.url)

    def _get_url(self):
        url = os.environ.get("EEMETER_WEATHER_CACHE_URL")
        if url is None:
            directory = "{}/.eemeter/cache".format(os.path.expanduser('~'))
            if not os.path.exists(directory):
                os.makedirs(directory)
            url = "sqlite:///{}/weather_cache.db".format(directory)
        return url

    def _prepare_db(self, url=None):
        if url is None:
            url = self._get_url()

        self.url = url

        eng = create_engine(url)
        metadata = MetaData(eng)

        tbl_items = Table(
            "items",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("data", String),
            Column("key", String, unique=True),
            Column("dt", DateTime)
        )

        tbl_items.create(checkfirst=True)

        self.items = tbl_items

    def key_exists(self, key):
        s = select([self.items.c.key]).where(self.items.c.key == key)
        result = s.execute()
        return result.fetchone() is not None

    def save_json(self, key, data):
        data = json.dumps(data)
        if self.key_exists(key):
            s = self.items.update().where(self.items.c.key == key).values(
                key=key, data=data, dt=func.now())
        else:
            s = self.items.insert().values(key=key, data=data, dt=func.now())
        s.execute()

    def retrieve_json(self, key):
        s = select([self.items.c.data]).where(self.items.c.key == key)
        result = s.execute()
        data = result.fetchone()
        if data is None:
            return None
        else:
            return json.loads(data[0])

    def retrieve_datetime(self, key):
        s = select([self.items.c.dt]).where(self.items.c.key == key)
        result = s.execute()
        data = result.fetchone()
        if data is None:
            return None
        else:
            return data[0]

    def clear(self, key=None):
        if key is None:
            s = self.items.delete()
        else:
            s = self.items.delete().where(self.items.c.key == key)
        s.execute()
