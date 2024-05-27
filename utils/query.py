try:
    import uuid

    from datetime import datetime
    import os
    import logging

    from functools import wraps
    from enum import Enum
    from abc import ABC, abstractmethod
    from logging import StreamHandler

    import psycopg2

    import psycopg2.extras as extras

except Exception as e:
    raise Exception("Error : {}".format(e))

from dotenv import load_dotenv
load_dotenv()

class Logging(object):
    def __init__(self):
        format = "[%(asctime)s] %(name)s %(levelname)s %(message)s"
        # Logs to file
        logging.basicConfig(
            filename="logfile",
            filemode="a",
            format=format,
            level=logging.INFO,
        )
        self.logger = logging.getLogger("python")
        formatter = logging.Formatter(
            format,
        )
        # Logs to Console
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)


global logger
logger = Logging()


def error_handling_with_logging(argument=None):
    def real_decorator(function):
        @wraps(function)
        def wrapper(self, *args, **kwargs):
            function_name = function.__name__
            response = None
            try:
                if kwargs == {}:
                    response = function(self)
                else:
                    response = function(self, **kwargs)
            except Exception as e:
                response = {
                    "status": -1,
                    "error": {"message": str(e), "function_name": function.__name__},
                }
                logger.logger.info(response)
            return response

        return wrapper

    return real_decorator


class Settings(object):
    """settings class"""

    def __init__(
            self,
            port="",
            server="",
            username="",
            password="",
            timeout=100,
            database_name="",
            **kwargs,
    ):
        self.port = port
        self.server = server
        self.username = username
        self.password = password
        self.timeout = timeout
        self.database_name = database_name


class DatabaseInterface(ABC):
    @abstractmethod
    def get_data(self, query):
        """
        For given query fetch the data
        :param query: Str
        :return: Dict
        """

    def execute(self, query, data):
        """
        Inserts data into SQL Server
        :param query:  Str
        :return: Dict
        """

    def insert_many(self, query, data):
        """
        Insert Many items into database
        :param query: str
        :param data: tuple
        :return: Dict
        """

    def get_data_batch(self, batch_size=10, query=""):
        """
        Gets data into batches
        :param batch_size: INT
        :param query: STR
        :return: DICT
        """


class DatabaseAurora(DatabaseInterface):
    """Aurora database class"""

    def __init__(self, data_base_settings):
        self.data_base_settings = data_base_settings
        self.client = psycopg2.connect(
            host=self.data_base_settings.server,
            port=self.data_base_settings.port,
            database=self.data_base_settings.database_name,
            user=self.data_base_settings.username,
            password=self.data_base_settings.password,
        )

    #@error_handling_with_logging()
    def get_data(self, query):
        self.query = query
        cursor = self.client.cursor()
        cursor.execute(self.query)
        result = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        data = [dict(zip(columns, item)) for item in result]
        cursor.close()
        _ = {"statusCode": 200, "data": data}

        return _

    @error_handling_with_logging()
    def execute(self, query, data):
        self.query = query
        cursor = self.client.cursor()
        cursor.execute(self.query, data)
        self.client.commit()
        cursor.close()
        return {"statusCode": 200, "data": True}

    @error_handling_with_logging()
    def get_data_batch(self, batch_size=10, query=""):
        self.query = query
        cursor = self.client.cursor()
        cursor.execute(self.query)
        columns = [column[0] for column in cursor.description]
        while True:
            result = cursor.fetchmany(batch_size)
            if not result:
                break
            else:
                items = [dict(zip(columns, data)) for data in result]
                yield items

    @error_handling_with_logging()
    def insert_many(self, query, data):
        self.query = query
        cursor = self.client.cursor()
        extras.execute_batch(cursor, self.query, data)
        self.client.commit()
        cursor.close()
        return {"statusCode": 200, "data": True}
    
def query_cv(ids):
    settings = Settings(
        port=os.environ.get("PGVECTOR_PORT"),
        server=os.environ.get("PGVECTOR_READER_HOST"),
        username=os.environ.get("PGVECTOR_USER"),
        password=os.environ.get("PGVECTOR_PASSWORD"),
        database_name=os.environ.get("PGVECTOR_DATABASE"),
    )
    db = DatabaseAurora(settings)
    query = "SELECT * FROM user_cv WHERE id IN ({})".format(','.join(map(str, ids)))
    
    return db.get_data(query)

if __name__ == "__main__":
    settings = Settings(
        port=os.environ.get("PGVECTOR_PORT"),
        server=os.environ.get("PGVECTOR_WRITER_HOST"),
        username=os.environ.get("PGVECTOR_USER"),
        password=os.environ.get("PGVECTOR_PASSWORD"),
        database_name=os.environ.get("PGVECTOR_DATABASE"),
    )
    db = DatabaseAurora(settings)
    ids_to_delete = [471,472]  # replace with your actual list of ids
    query = """
    DELETE FROM public.user_cv
    WHERE id IN ({});
    """.format(','.join(map(str, ids_to_delete)))
    db.execute(query)