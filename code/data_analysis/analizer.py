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
        x = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        x_train, y_train, x_test, y_test = train_test_split(
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



class TripDataAnalizer(DataAnalizer):
    """
    TripDataAnalizer is a class that inherits from DataAnalizer and extends its functionality
    to preprocess and engineer features specifically for trip data.

    This class performs several data manipulation tasks, including:

    -   Extracting datetime features from pickup and dropoff timestamps.
    -   Calculating the trip duration in minutes.
    -   Calculating the average speed of the trip in miles per hour.

    It assumes the input DataFrame contains columns 'tpep_pickup_datetime',
    'tpep_dropoff_datetime', and 'trip_distance'.

    The class initializes with a DataFrame, target column, test size for train-test split,
    and a random state for reproducibility. It then automatically calls methods to extract
    datetime features, calculate trip duration, and calculate average speed.
    """

    def __init__(self, df: pd.DataFrame, target: str, test_size=0.2, random_state=None):
        """
        Initializes the TripDataAnalizer instance and splits the dataset into training and test sets.
        It also calls class specific methods of the class.

        Args:
            df (pd.DataFrame): The dataset to be split.
            target (str): The name of the target column in the dataset.
            test_size (float, optional): The proportion of the dataset to include in the test split. Default is 0.2.
            random_state (int or None, optional): The random seed used for shuffling the data. Default is None.
        """

        #Manipulate data
        df = self.extract_datetime_features(df)
        df = self.calculate_trip_duration(df)
        df = self.calculate_average_speed(df)

        super().__init__(df, target, test_size, random_state)



    # Metodo para separar "tpep_pickup_datetime" e "tpep_dropoff_datetime" em várias features
    def extract_datetime_features(self, df):
        """
        Converts pickup and dropoff datetimes into a datetime object.
        This method also creates usefull new features from this object to extract future insights about time
        """

        # Ensure datetime columns are in the correct format
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

        # Extract features from pickup time
        df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
        df['pickup_day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek
        df['pickup_day_of_month'] = df['tpep_pickup_datetime'].dt.day
        df['pickup_month'] = df['tpep_pickup_datetime'].dt.month

        # Extract features from dropoff time
        df['dropoff_hour'] = df['tpep_dropoff_datetime'].dt.hour
        df['dropoff_day_of_week'] = df['tpep_dropoff_datetime'].dt.dayofweek
        df['dropoff_day_of_month'] = df['tpep_dropoff_datetime'].dt.day
        df['dropoff_month'] = df['tpep_dropoff_datetime'].dt.month

        print("Feature columns extracted successfully.")
        return df

    # Metodo para calcular o tempo de duração da viagem
    def calculate_trip_duration(self, df):
        """Calculates the duration of the trip in minutes."""

        # Ensure the dataset has the necessary distance column
        if 'tpep_dropoff_dateime' and 'tpep_pickup_datetime' not in df.columns:
            raise ValueError("tpep time stamps collums are missing from the dataset.")

        # Compute trip duration in minutes
        df['trip_duration_min'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60

        print("Trip duration calculated successfully.")
        return df

    # Metodo para calcular a velocidade média da viagem
    def calculate_average_speed(self, df):
        """Calculates the average speed of the trip in km/h."""

        # Ensure the dataset has the necessary distance column
        if 'trip_distance'  not in df.columns:
            raise ValueError("Column 'trip_distance' is missing from the dataset.")

        # Avoid division by zero for extremely short trips
        df['average_speed_mph'] = df['trip_distance'] / (df['trip_duration_min'] / 60)
        df['average_speed_mph'].fillna(0, inplace=True)  # Replace NaN values with 0

        print("Average speed calculated successfully.")
        return df


    def drop_columns(self, df):
        """Drops unnecessary or problematic features from the dataset"""
        df.drop(columns=['tpep_pickup_datetime'], inplace=True)
        df.drop(columns=['tpep_dropoff_datetime'], inplace=True)
        return df
