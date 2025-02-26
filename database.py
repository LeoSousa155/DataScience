import kagglehub
import sqlite3 as sql
import pandas as pd
import numpy as np

class Database:
    """
    This class is responsable for interacting with kagglehub SQLite database
    """
    def __init__(self, month: int):
        """
        Initializes all inner variables
        :param month: Should be the month index
        """
        self._path = kagglehub.dataset_download("dhruvildave/new-york-city-taxi-trips-2019")
        self._month = month

        if month < 1 or month > 12:
            raise ValueError("Invalid month")

        self._db_path =self._path + f"\\2019\\2019-{month:02}.sqlite"
        self._conn = sql.connect(self._db_path)
        self._cursor = self._conn.cursor()


    def get_table_name(self) -> str:
        """
        :return: name of the table
        """
        table_name_query = "SELECT name FROM sqlite_master WHERE type='table';"
        return self._cursor.execute(table_name_query).fetchone()[0]


    def get_column_names(self) -> list:
        """
        :return: list with all column names
        """
        self._cursor.execute("PRAGMA table_info(tripdata);")
        columns = [col[1] for col in self._cursor.fetchall()]
        return columns


    def get_data_array(self) -> np.ndarray:
        """
        This method fetches all data from the table and convert it into a numpy array
        :return: Numpy array with all data from the table
        """
        query = "SELECT * FROM tripdata"
        self._cursor.execute(query)
        data = self._cursor.fetchall()

        if not data:
            return np.array([])

        column_names = self.get_column_names()
        df = pd.DataFrame(data, columns=column_names)

        return df.to_numpy()


    def get_dataframe(self) -> pd.DataFrame:
        """
        This method fetches all data from the table and convert it into a pandas dataframe
        :return: Pandas dataframe with all data from the table
        """
        query = "SELECT * from tripdata"
        data = self._cursor.execute(query).fetchall()
        df = pd.DataFrame.from_records(data)
        df.columns = self.get_column_names()
        return df


    def get_dataframe_nrows(self, n: int) -> pd.DataFrame:
        """
        This method fetches all data from the table and convert it into a pandas dataframe
        :return: Pandas dataframe with all data from the table
        """
        query = "SELECT * from tripdata"
        data = self._cursor.execute(query).fetchmany(n)
        df = pd.DataFrame.from_records(data)
        df.columns = self.get_column_names()
        return df