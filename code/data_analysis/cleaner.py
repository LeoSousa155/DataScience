# Data Cleaning
class DataCleaning:
    """
    Class for cleaning operations.

    Methods:
        remove_duplicates(): Remove duplicate rows from the dataset.
        handle_missing_values(strategy='mean'): Handle missing values using the specified strategy.
        remove_outliers(threshold=3): Remove outliers from the dataset
    """
    def __init__(self, data_loader):
        """
        Initializes the DataPreprocessing class with a DataLoader object.
        """
        self.data_loader = data_loader

    def remove_duplicates(self):
        """
        Remove duplicate rows from the train dataset.
        """
        try:
            # Check if data and labels are not None
            if self.data_loader.data_train is None:
                raise ValueError("Data has not been loaded yet.")
            if self.data_loader.labels_train is None:
                raise ValueError("Labels have not been loaded yet.")

            # Remove duplicate rows from training data (do not apply to test data)
            self.data_loader.data_train.drop_duplicates(inplace=True)
            self.data_loader.labels_train = self.data_loader.labels_train[self.data_loader.data_train.index]

            print("Duplicate rows removed from training data.")

        except ValueError as ve:
            print("Error:", ve)

    def handle_missing_values(self, strategy='drop'):
        """
        Handle missing values using the specified strategy.

        Parameters:
            strategy (str): The strategy to handle missing values ('mean', 'median', 'most_frequent', or a constant value).
        """
        try:
            # Check if data is not None
            if self.data_loader.data_train is None or self.data_loader.data_test is None :
                raise ValueError("Data has not been loaded yet.")

            # Check if there are missing values
            if self.data_loader.data_train.isnull().sum().sum() == 0 and self.data_loader.data_test.isnull().sum().sum() == 0:
                print("No missing values found in the data.")
                return

            # Handle missing values based on the specified strategy
            if strategy == 'mean':
                self.data_loader.data_train.fillna(self.data_loader.data_train.mean(), inplace=True)
                self.data_loader.data_test.fillna(self.data_loader.data_test.mean(), inplace=True)
            elif strategy == 'median':
                self.data_loader.data_train.fillna(self.data_loader.data_train.median(), inplace=True)
                self.data_loader.data_test.fillna(self.data_loader.data_test.median(), inplace=True)
            elif strategy == 'most_frequent':
                self.data_loader.data_train.fillna(self.data_loader.data_train.mode().iloc[0], inplace=True)
                self.data_loader.data_test.fillna(self.data_loader.data_test.mode().iloc[0], inplace=True)
            elif strategy == 'fill_nan':
                self.data_loader.data_train.fillna(strategy, inplace=True)
                self.data_loader.data_test.fillna(strategy, inplace=True)
            elif strategy == 'drop':
                self.data_loader.data_train = self.data_loader.data_train.dropna(axis=0)
                self.data_loader.labels_train = self.data_loader.labels_train[self.data_loader.data_train.index]
                self.data_loader.data_test = self.data_loader.data_test.dropna(axis=0)
                self.data_loader.labels_test = self.data_loader.labels_test[self.data_loader.data_test.index]

            else:
                raise ValueError("Invalid strategy.")
            print("Missing values handled using strategy:", strategy)

        except ValueError as ve:
            print("Error:", ve)

    def _detect_outliers(self, threshold=4):
        """
        Detect outliers in numerical features using z-score method.

        Parameters:
            threshold (float): The threshold value for determining outliers.

        Returns:
            outliers (DataFrame): DataFrame containing the outliers.
        """
        try:
            # Check if test data is not None
            if self.data_loader.data_train is None:
                raise ValueError("Data has not been loaded yet.")

            # Identify numerical features
            numerical_features = self.data_loader.data_train.select_dtypes(include=['number'])

            # Calculate z-scores for numerical features
            z_scores = (numerical_features - numerical_features.mean()) / numerical_features.std()

            # Find outliers based on threshold
            outliers = self.data_loader.data_train[(z_scores.abs() > threshold).any(axis=1)]

            return outliers

        except ValueError as ve:
            print("Error:", ve)

    def remove_outliers(self, threshold=2):
        """
        Remove outliers from the dataset using z-score method.

        Parameters:
            threshold (float): The threshold value for determining outliers.
        """
        try:
            # Check if data_loader.data is not None
            if self.data_loader.data_train is None:
                raise ValueError("Data has not been loaded yet.")

            # Detect outliers
            outliers = self._detect_outliers(threshold)

        except ValueError as ve:
            print("Error:", ve)
