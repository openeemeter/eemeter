import os
import json
from sqlalchemy import (
    create_engine,
    Table,
    MetaData,
    Column,
    Integer,
    String,
)
from sqlalchemy.sql import select
import pandas as pd


class SqlCO2Store(object):

    def __init__(self, url=None):
        self._prepare_db(url)

    def __repr__(self):
        return 'SqlCO2Store("{}")'.format(self.url)

    def _get_url(self):
        url = os.environ.get("EEMETER_CO2_CACHE_URL")
        if url is None:
            directory = "{}/.eemeter/co2_cache".format(os.path.expanduser('~'))
            if not os.path.exists(directory):
                os.makedirs(directory)
            url = "sqlite:///{}/co2_cache.db".format(directory)
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
            Column("year", Integer),
            Column("region", String),
            Column("co2_by_load", String),
            Column("load_by_hour", String)
        )

        tbl_items.create(checkfirst=True)

        self.items = tbl_items

    def key_exists(self, year, region):
        s = select([self.items.c.year, self.items.c.region]).where(
            (self.items.c.year == year) & (self.items.c.region == region))
        result = s.execute()
        return result.fetchone() is not None

    def save_json(self, year, region, co2_by_load, load_by_hour):
        co2_by_load = json.dumps({str(k): v for k, v in
                                  co2_by_load.to_dict().items()})
        load_by_hour = json.dumps({str(k): v for k, v in
                                   load_by_hour.to_dict().items()})
        if self.key_exists(year, region):
            s = self.items.update().where(
                (self.items.c.year == year) &
                (self.items.c.region == region)).values(
                year=year, region=region, co2_by_load=co2_by_load,
                load_by_hour=load_by_hour)
        else:
            s = self.items.insert().values(
                year=year, region=region, co2_by_load=co2_by_load,
                load_by_hour=load_by_hour)
        s.execute()

    def retrieve_co2_by_load(self, year, region):
        s = select([self.items.c.co2_by_load]).where(
            (self.items.c.year == year) & (self.items.c.region == region))
        result = s.execute()
        data = result.fetchone()
        if data is None:
            return None
        else:
            this_json = json.loads(data[0])
            k = list(this_json.keys())
            v = [this_json[i] for i in k]
            k = [float(i) for i in k]
            return pd.Series(v, index=k).sort_index()

    def retrieve_load_by_hour(self, year, region):
        s = select([self.items.c.load_by_hour]).where(
            (self.items.c.year == year) & (self.items.c.region == region))
        result = s.execute()
        data = result.fetchone()
        if data is None:
            return None
        else:
            this_json = json.loads(data[0])
            k = list(this_json.keys())
            v = [this_json[i] for i in k]
            return pd.Series(v, index=pd.to_datetime(k)).sort_index()

    def clear(self, key=None):
        if key is None:
            s = self.items.delete()
        else:
            s = self.items.delete().where(self.items.c.key == key)
        s.execute()
