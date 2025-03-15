import pandas as pd
from sklearn.model_selection import train_test_split


class DataAnalizer:
    """
    A class to load and split a dataset into training and testing sets.

    This class is designed to handle the process of dividing a DataFrame into
    training and testing sets for machine learning tasks. It takes in a DataFrame,
    a target column, and splits the data into features and labels, which are then
    separated into training and testing datasets.

    Attributes:
        df (pd.DataFrame): The DataFrame containing the data.
        target (str): The name of the target column in the dataset.
        test_size (float): The proportion of the dataset to include in the test split (default is 0.2).
        random_state (int or None): The random seed for reproducibility of the split (default is None).
        data_train (pd.DataFrame): The feature data for the training set.
        labels_train (pd.Series): The labels for the training set.
        data_test (pd.DataFrame): The feature data for the testing set.
        labels_test (pd.Series): The labels for the testing set.
    """

    def __init__(self, df: pd.DataFrame, target: str,  test_size= 0.2, random_state=None):
        """
        Initializes the DataLoader instance and splits the dataset into training and test sets.

        Args:
            df (pd.DataFrame): The dataset to be split.
            target (str): The name of the target column in the dataset.
            test_size (float, optional): The proportion of the dataset to include in the test split. Default is 0.2.
            random_state (int or None, optional): The random seed used for shuffling the data. Default is None.
        """
        self.df = df
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.data_train = None
        self.labels_train = None
        self.data_test = None
        self.labels_test = None

        #divide data in train and test sets
        self._divide_data()


    def _divide_data(self):
        """
        Splits the DataFrame into features (X) and labels (y), and divides them into training and testing sets.

        The method uses `train_test_split()` from scikit-learn to randomly split the data into
        training and test sets based on the specified `test_size` and `random_state`.

        The method sets the following attributes:
            - data_train: Feature data for training.
            - labels_train: Labels for training.
            - data_test: Feature data for testing.
            - labels_test: Labels for testing.
        """
        y = self.df[self.target]
        x = self.df.drop(columns=[self.target])
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=self.test_size,random_state=self.random_state
        )
        self.data_train  = x_train
        self.labels_train = y_train
        self.data_test    = x_test
        self.labels_test  = y_test
        print("Data divided successfully.")


    def _drop_columns(self, cols):
        """
        Drops a specified feature columns from both the training and testing datasets.

        Args:
            cols (str): The list of the columns to be dropped from both the training and testing datasets.
        """
        try:
            self.data_train.drop(cols, axis=1, inplace=True)
            self.data_test.drop(cols, axis=1, inplace=True)
            print(f"Feature columns {cols} dropped successfully.")
        except Exception as e:
            print(f"Error dropping columns {cols}:", e)


    def save_dataset(self, file_path: str):
        """
        Saves the entire dataset (DataFrame) to disk as a CSV file.

        Args:
            file_path (str): The path (including filename) where the dataset will be saved.
        """
        try:
            self.df.to_csv(file_path, index=False, chunksize=100000)
            print(f"Dataset saved successfully to {file_path}.")
        except Exception as e:
            print(f"Error saving dataset to {file_path}:", e)

    @classmethod
    def load_dataset(cls, file_path: str, target: str, test_size=0.2, random_state=None):
        """
        Loads a dataset from disk and returns a new instance of DataAnalizer.

        Args:
            file_path (str): The path (including filename) from where the dataset will be loaded.
            target (str): The name of the target column in the dataset.
            test_size (float, optional): The proportion of the dataset to include in the test split. Default is 0.2.
            random_state (int or None, optional): The random seed used for shuffling the data. Default is None.

        Returns:
            DataAnalizer: An instance of the DataAnalizer class with the loaded dataset.
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully from {file_path}.")
            return cls(df, target, test_size, random_state)
        except Exception as e:
            print(f"Error loading dataset from {file_path}:", e)
            return None


class TripDataAnalizer(DataAnalizer):
    """
    TripDataAnalizer is a class that inherits from DataAnalizer and extends its functionality
    to preprocess and engineer features specifically for trip data.
    """

    def __init__(self, df: pd.DataFrame, target: str, test_size=0.2, random_state=None):
        """
        Initializes the TripDataAnalizer instance and processes the dataset.
        """
        self.df = df.copy()
        self.target = target
        self.test_size = test_size
        self.random_state = random_state

        # Preprocess data
        self.extract_datetime_features()
        self.drop_columns()
        self.order_numerical_categorical()

        # Initialize parent class
        super().__init__(self.df, self.target, self.test_size, self.random_state)

    def order_numerical_categorical(self):
        """
        Groups the features into numerical and categorical groups.
        """
        self.df = self.df.loc[:,[
             # continuous data
             'trip_distance',
             'fare_amount',
             'tip_amount',
             'tolls_amount',
             'extra',
             'pickup_time_in_seconds',
             'dropoff_time_in_seconds',

             # discrete data
             'passenger_count',
             'pickup_hour',
             'pickup_day_of_week',
             'pickup_day_of_month',
             'pickup_month',
             'dropoff_hour',
             'dropoff_day_of_week',
             'dropoff_day_of_month',
             'dropoff_month',
             'mta_tax',
             'congestion_surcharge',

             # categorical data
             'vendorid',
             'ratecodeid',
             'pulocationid',
             'dolocationid',
             'payment_type',
         ]]

    def extract_datetime_features(self):
        """
        Converts pickup and dropoff datetimes into datetime objects and extracts features.
        """
        self.df['tpep_pickup_datetime'] = pd.to_datetime(self.df['tpep_pickup_datetime'])
        self.df['tpep_dropoff_datetime'] = pd.to_datetime(self.df['tpep_dropoff_datetime'])

        # Extract features from pickup time
        self.df['pickup_time_in_seconds'] = (self.df['tpep_pickup_datetime'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        self.df['pickup_hour'] = self.df['tpep_pickup_datetime'].dt.hour
        self.df['pickup_day_of_week'] = self.df['tpep_pickup_datetime'].dt.dayofweek
        self.df['pickup_day_of_month'] = self.df['tpep_pickup_datetime'].dt.day
        self.df['pickup_month'] = self.df['tpep_pickup_datetime'].dt.month

        # Extract features from dropoff time
        self.df['dropoff_time_in_seconds'] = (self.df['tpep_dropoff_datetime'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        self.df['dropoff_hour'] = self.df['tpep_dropoff_datetime'].dt.hour
        self.df['dropoff_day_of_week'] = self.df['tpep_dropoff_datetime'].dt.dayofweek
        self.df['dropoff_day_of_month'] = self.df['tpep_dropoff_datetime'].dt.day
        self.df['dropoff_month'] = self.df['tpep_dropoff_datetime'].dt.month

        print("Feature columns extracted successfully.")

    def drop_columns(self):
        """
        Drops unnecessary or problematic features from the dataset.
        """
        columns_to_drop = [
            'store_and_fwd_flag',       # Irrelevant flag
            'total_amount',             # Problematic for analysis (fare = total - taxes)
            'improvement_surcharge'     # Constant feature
            'tpep_pickup_datetime',     # Decomposed in other features
            'tpep_dropoff_datetime',    # Decomposed in other features
        ]

        # tpep_pickup_datetime and tpep_dropoff_datetime are dropped inside featureGenerator
        # because they are necessary for the creation of  travel time feature

        self.df.drop(columns=[col for col in columns_to_drop if col in self.df.columns], inplace=True)


    def drop_column(self, col_name: str):
        if col_name in self.df.columns:
            self.df.drop(col_name, inplace=True)
            print(f"Feature '{col_name}' dropped successfully.")
        else:            print(f"Feature '{col_name}' was not found in the dataset.")