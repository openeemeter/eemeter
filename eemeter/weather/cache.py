import os
import json
import sqlite3


class SqliteJSONStore(object):

    def __init__(self, directory=None):

        # creates the self.conn attribute
        self._prepare_db(directory)

    def __repr__(self):
        return 'SqliteJSONStore("{}")'.format(self.directory)

    def _get_directory(self):
        """ Returns a directory to be used for caching.
        """
        directory = os.environ.get("EEMETER_WEATHER_CACHE_DIRECTORY",
                                   os.path.expanduser('~/.eemeter/cache'))
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    def _prepare_db(self, directory=None):

        if directory is None:
            directory = self._get_directory()

        self.directory = directory

        self.db_filename = os.path.join(directory, "weather_cache.db")

        exists = os.path.exists(self.db_filename)
        conn = sqlite3.connect(self.db_filename,
                               detect_types=sqlite3.PARSE_DECLTYPES)

        if not exists:
            conn.execute(
                'CREATE TABLE IF NOT EXISTS items('
                'id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, '
                'data TEXT NOT NULL, '
                'key TEXT UNIQUE NOT NULL, '
                'dt TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL);'
            )
            conn.commit()

        self.conn = conn

    def key_exists(self, key):
        cursor = self.conn.cursor()
        cursor.execute('SELECT key FROM items WHERE key=?;', (key,))
        key = cursor.fetchone()
        return key is not None

    def save_json(self, key, data):
        data = json.dumps(data)

        if self.key_exists(key):
            sql = (
                'UPDATE items SET'
                ' data=?,'
                ' dt=datetime(\'now\')'
                ' WHERE key=?'
            )
        else:
            sql = (
                'INSERT INTO items'
                ' (data, key)'
                ' VALUES (?, ?)'
            )
        self.conn.execute(sql, (data, key))
        self.conn.commit()

    def retrieve_json(self, key):
        cursor = self.conn.cursor()
        cursor.execute('SELECT data FROM items WHERE key=?;', (key,))
        data = cursor.fetchone()
        if data is None:
            return None
        return json.loads(data[0])

    def retrieve_datetime(self, key):
        cursor = self.conn.cursor()
        cursor.execute('SELECT dt FROM items WHERE key=?;', (key,))
        data = cursor.fetchone()
        if data is None:
            return None
        return data[0]

    def clear(self, key=None):
        if key is None:
            self.conn.execute('DELETE FROM items;')
            self.conn.commit()
        else:
            self.conn.execute('DELETE FROM items WHERE key=?;', (key,))
            self.conn.commit()
