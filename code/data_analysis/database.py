import kagglehub
import pandas as pd
import numpy as np
import sqlite3 as sql

class KagglehubDatabaseLoader:
    """
    A class to download and manage datasets from KaggleHub.

    This class automates the process of downloading a dataset from KaggleHub
    and provides a method to retrieve the local file path.

    Attributes:
        _dataset (str): The name of the dataset to be downloaded.
        _path (str or None): The local directory path where the dataset is saved.
    """

    def __init__(self, dataset: str):
        """
        Initializes the KagglehubDatabaseLoader class.

        Args:
            dataset (str): The name of the dataset to be downloaded.
        """
        self._dataset = dataset
        self._path = None

        self._download_dataset()


    def _download_dataset(self) -> None:
        """
        Downloads the specified dataset from KaggleHub.

        This method attempts to download the dataset and assigns the local
        file path to the `_path` attribute. If an error occurs during the
        download, an exception is caught and printed.
        """
        try:
            self._path = kagglehub.dataset_download(self._dataset)
            print("Dataset downloaded Successfully ")
            print("Dataset saved on: ", self._path)
        except Exception as e:
            print("Error downloading dataset:", e)


    def get_path(self) -> str | None:
        """
        Returns the local directory path of the downloaded dataset.

        Returns:
            str or None: The file path of the dataset if the download was
                        successful, otherwise None.
        """
        return self._path




class KagglehubSQLiteLoader(KagglehubDatabaseLoader):
    """
    A subclass of KagglehubDatabaseLoader to interact with an SQLite dataset.

    This class adds methods to retrieve information about tables, numpy arrays and dataframes.
    the SQLite dataset is downloaded from KaggleHub.

    Attributes:
        _dataset (str): The name of the dataset to be downloaded from KaggleHub.
        _path (str or None): The local directory path where the dataset is saved.
        _file_path (str or None): The complete path to the specific SQLite file within the dataset.
        _conn: The SQLite connection object used to interact with the database.
        _cursor: The cursor object for executing SQL queries on the database.
    """

    def __init__(self, dataset: str, file: str):
        """
        Initializes the KagglehubSQLLoader class by calling the superclass constructor
        to download the dataset and setting up the SQLite connection.

        Args:
            dataset (str): The dataset name to be downloaded.
            file (str): The name of the SQLite file in the downloaded dataset to be opened.
        """
        super().__init__(dataset)
        self._file_path = self._path + file
        self._conn = sql.connect(self._file_path)
        self._cursor = self._conn.cursor()


    def get_table_names(self) -> list | None:
        """
        Retrieves all table names from the SQLite database.

        Returns:
            list: A list of table names if successful, None if an error occurs.
        """
        try:
            table_name_query = "SELECT name FROM sqlite_master WHERE type='table';"
            return self._cursor.execute(table_name_query).fetchall()
        except Exception as e:
            print("Error getting table names:", e)


    def get_column_names(self, table: str) -> list | None:
        """
        Retrieves the column names for a given table in the SQLite database.

        Args:
            table (str): The name of the table for which to get column names.

        Returns:
            list: A list of column names if successful, None if an error occurs.
        """
        try:
            self._cursor.execute(f"PRAGMA table_info({table});")
            columns = [col[1] for col in self._cursor.fetchall()]
            return columns
        except Exception as e:
            print("Error getting column names:", e)


    def get_table_row_count(self, table: str) -> int | None:
        """
        Retrieves the row count for a given table.

        Args:
            table (str): The name of the table for which to get the row count.

        Returns:
            int: The number of rows in the table if successful, None if an error occurs.
        """
        try:
            self._cursor.execute(f"SELECT COUNT(*) FROM {table};")
            return self._cursor.fetchone()[0]
        except Exception as e:
            print("Error getting table row count:", e)


    def get_table_data(self, table: str) -> np.ndarray | None:
        """
        Retrieves all data from a given table as a NumPy array.

        Args:
            table (str): The name of the table to retrieve data from.

        Returns:
            np.ndarray: A NumPy array containing the data from the table if successful, None if an error occurs.
        """
        try:
            self._cursor.execute(f"SELECT * FROM {table}")
            data = self._cursor.fetchall()
            column_names = self.get_column_names(table)
            df = pd.DataFrame(data, columns=column_names)
            return df.to_numpy()
        except Exception as e:
            print("Error getting table data array:", e)


    def get_table_dataframe(self, table: str) -> pd.DataFrame | None:
        """
        Retrieves all data from a given table as a Pandas DataFrame.

        Args:
            table (str): The name of the table to retrieve data from.

        Returns:
            pd.DataFrame: A DataFrame containing the data from the table if successful, None if an error occurs.
        """
        try:
            self._cursor.execute(f"SELECT * from {table}")
            data = self._cursor.fetchall()
            df = pd.DataFrame.from_records(data)
            df.columns = self.get_column_names(table)
            return df
        except Exception as e:
            print("Error getting table dataframe:", e)


    def get_table_dataframe_nrows(self, table: str, n: int) -> pd.DataFrame | None:
        """
        Retrieves the first `n` rows of a given table as a Pandas DataFrame.

        Args:
            table (str): The name of the table to retrieve data from.
            n (int): The number of rows to retrieve.

        Returns:
            pd.DataFrame: A DataFrame containing the first `n` rows from the table if successful, None if an error occurs.
        """
        try:
            query = "SELECT * from tripdata"
            data = self._cursor.execute(query).fetchmany(n)
            df = pd.DataFrame.from_records(data)
            df.columns = self.get_column_names(table)
            return df
        except Exception as e:
            print(f"Error getting the first {n} rows from table:", e)


    def get_table_dataframe_random_sample(self, table: str, percentage: float = 0.05) -> pd.DataFrame | None:
        """
        Retrieves a random sample of rows from the specified table based on a given percentage.

        This method selects a random subset of the rows from the given table. The percentage of
        rows to sample is specified by the `percentage` argument. The sampling is done using
        the SQLite `RANDOM()` function, which provides a pseudo-random selection of rows.

        Args:
            table (str): The name of the table from which to retrieve the random sample.
            percentage (float, optional): The fraction of rows to sample, between 0.0 and 1.0.
                                          Defaults to 0.05 (5%).

        Returns:
            pd.DataFrame | None: A Pandas DataFrame containing the random sample of rows.
                                  Returns an empty DataFrame if no data is found, or None if an error occurs.

        Raises:
            ValueError: If the `percentage` is not between 0.0 and 1.0.
        """
        try:
            if not 0.0 <= percentage <= 1.0:
                raise ValueError("Percentage must be between 0.0 and 1.0")

            size = self.get_table_row_count(table)
            threshold = int(round(percentage * size))

            query = "SELECT * FROM tripdata WHERE ABS(RANDOM()) % ? < ?"

            self._cursor.execute(query, (size, threshold))
            data = self._cursor.fetchall()

            if not data:
                return pd.DataFrame(columns=self.get_column_names(table))

            df = pd.DataFrame.from_records(data)
            df.columns = self.get_column_names(table)
            return df
        except Exception as e:
            print(f"Error getting random sample from table:", e)