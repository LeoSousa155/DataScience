from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataPreprocessing:
    """
    Class responsible for preprocessing the loaded dataset. Need to pass the data with first columns as numerical and the last columns as categorical, then indicate how many categorical features are present in the dataset

    Methods:
        _normalize_features(): Normalizes all features using standard scaling for numerical and min-max scalling for categorical.
    """

    def __init__(self, data_loader, number_categorical_features):
        """
        Initializes the DataPreprocessing class with a DataLoader object.
        """
        self.data_loader = data_loader
        self.number_categorical_features = number_categorical_features

        # Preprocess data
        self._normalize_features()

    def _normalize_features(self):
        """
        Normalizes features using standard scaling for numerical and min-max scalling for categorical.
        """
        try:
            # Check if data_train and data_test are not None
            if self.data_loader.data_train is None or self.data_loader.data_test is None:
                raise ValueError("Data has not been loaded yet.")
            # Check if labels_train and labels_test are not None
            if self.data_loader.labels_train is None or self.data_loader.labels_test is None:
                raise ValueError("Labels have not been loaded yet.")

            # Identify real numerical features (excluding the last two columns we know are categorical)
            real_numerical_features = self.data_loader.data_train.columns[:-self.number_categorical_features]

            # Normalize real numerical features using StandardScaler
            scaler = StandardScaler()
            self.data_loader.data_train[real_numerical_features] = scaler.fit_transform(
                self.data_loader.data_train[real_numerical_features])
            self.data_loader.data_test[real_numerical_features] = scaler.transform(
                self.data_loader.data_test[real_numerical_features])

            # Identify encoded features
            encoded_features = self.data_loader.data_train.columns[-self.number_categorical_features:]

            # Normalize encoded features using MinMaxScaler
            scaler = MinMaxScaler()
            self.data_loader.data_train[encoded_features] = scaler.fit_transform(
                self.data_loader.data_train[encoded_features])
            self.data_loader.data_test[encoded_features] = scaler.transform(
                self.data_loader.data_test[encoded_features])

            print("Features normalized successfully.")

        except ValueError as ve:
            print("Error:", ve)
