import numpy as np

class FeatureGenerator:
    def __init__(self, data_analizer):
        self.data_analizer = data_analizer


    def generate_features(self):
        """
        Generate and add new features to the dataset.

        This method orchestrates the creation of new features by calling
        individual methods responsible for different types of feature engineering.

        Returns:
            None
        """
        self.add_domain_knowledge_features()
        self.add_statistical_features()
        self.add_interaction_features()
        self.add_nonlinear_interaction_features()


    def add_domain_knowledge_features(self):
        self.calculate_trip_duration()
        self.calculate_average_speed()


    def add_statistical_features(self):
        """
        Add statistical features to the dataset.

        Statistical features provide insights into how much each sample's value is above or below the mean of the feature.

        Returns:
            None
        """
        data_train = self.data_analizer.data_train
        data_test = self.data_analizer.data_test

        # Mean features
        data_train['trip_distance_mean'] = data_train['trip_distance'].mean()
        data_train['tip_amount_mean'] = data_train['tip_amount'].mean()
        data_train['tolls_amount_mean'] = data_train['tolls_amount'].mean()
        data_train['extra_mean'] = data_train['extra'].mean()
        data_train['pickup_time_in_seconds_mean'] = data_train['pickup_time_in_seconds'].mean()
        data_train['dropoff_time_in_seconds_mean'] = data_train['dropoff_time_in_seconds'].mean()

        data_test['trip_distance_mean'] = data_test['trip_distance'].mean()
        data_test['tip_amount_mean'] = data_test['tip_amount'].mean()
        data_test['tolls_amount_mean'] = data_test['tolls_amount'].mean()
        data_test['extra_mean'] = data_test['extra'].mean()
        data_test['pickup_time_in_seconds_mean'] = data_test['pickup_time_in_seconds'].mean()
        data_test['dropoff_time_in_seconds_mean'] = data_test['dropoff_time_in_seconds'].mean()

        # Standard deviation features
        data_train['trip_distance_std'] = data_train['trip_distance'].std()
        data_train['tip_amount_std'] = data_train['tip_amount'].std()
        data_train['tolls_amount_std'] = data_train['tolls_amount'].std()
        data_train['pickup_time_in_seconds_std'] = data_train['pickup_time_in_seconds'].std()
        data_train['dropoff_time_in_seconds_std'] = data_train['dropoff_time_in_seconds'].std()

        data_test['trip_distance_std'] = data_test['trip_distance'].std()
        data_test['tip_amount_std'] = data_test['tip_amount'].std()
        data_test['tolls_amount_std'] = data_test['tolls_amount'].std()
        data_test['pickup_time_in_seconds_std'] = data_test['pickup_time_in_seconds'].std()
        data_test['dropoff_time_in_seconds_std'] = data_test['dropoff_time_in_seconds'].std()

        # Ratio features
        data_train['trip_time_ratio'] = (data_train['dropoff_time_in_seconds'] - data_train['pickup_time_in_seconds']) / data_train['trip_distance']
        data_train['tip_per_distance'] = data_train['tip_amount'] / data_train['trip_distance']

        data_test['trip_time_ratio'] = (data_test['dropoff_time_in_seconds'] - data_test['pickup_time_in_seconds']) / data_test['trip_distance']
        data_test['tip_per_distance'] = data_test['tip_amount'] / data_test['trip_distance']

        # Percentile-based features (e.g., 90th percentile)
        data_train['trip_distance_90th_percentile'] = data_train['trip_distance'].quantile(0.9)
        data_train['tip_amount_90th_percentile'] = data_train['tip_amount'].quantile(0.9)

        data_test['trip_distance_90th_percentile'] = data_test['trip_distance'].quantile(0.9)
        data_test['tip_amount_90th_percentile'] = data_test['tip_amount'].quantile(0.9)

        # Add the modified dataframe back to data_analizer or return it if needed
        self.data_analizer.data_train = data_train
        self.data_analizer.data_test = data_test


    def add_interaction_features(self):
        """
        Add interaction features to the dataset.

        Interaction features capture interactions between existing attributes, potentially revealing
        complex relationships not captured by individual features alone.

        Returns:
            None
        """
        self.data_analizer.data_train['trip_distance_passenger'] = self.data_analizer.data_train['trip_distance'] * self.data_analizer.data_train['passenger_count']
        self.data_analizer.data_train['tip_per_mile'] = self.data_analizer.data_train['tip_amount'] / (self.data_analizer.data_train['trip_distance'] + 1e-8)
        self.data_analizer.data_train['tolls_per_mile'] = self.data_analizer.data_train['tolls_amount'] / (self.data_analizer.data_train['trip_distance'] + 1e-8)
        self.data_analizer.data_train['pickup_dropoff_duration'] = self.data_analizer.data_train['dropoff_time_in_seconds'] - self.data_analizer.data_train['pickup_time_in_seconds']
        self.data_analizer.data_train['pickup_dropoff_hour_diff'] = self.data_analizer.data_train['dropoff_hour'] - self.data_analizer.data_train['pickup_hour']

        self.data_analizer.data_test['trip_distance_passenger'] = self.data_analizer.data_test['trip_distance'] * self.data_analizer.data_test['passenger_count']
        self.data_analizer.data_test['tip_per_mile'] = self.data_analizer.data_test['tip_amount'] / (self.data_analizer.data_test['trip_distance'] + 1e-8)
        self.data_analizer.data_test['tolls_per_mile'] = self.data_analizer.data_test['tolls_amount'] / (self.data_analizer.data_test['trip_distance'] + 1e-8)
        self.data_analizer.data_test['pickup_dropoff_duration'] = self.data_analizer.data_test['dropoff_time_in_seconds'] - self.data_analizer.data_test['pickup_time_in_seconds']
        self.data_analizer.data_test['pickup_dropoff_hour_diff'] = self.data_analizer.data_test['dropoff_hour'] - self.data_analizer.data_test['pickup_hour']

        print("Interaction features added successfully.")


    def add_nonlinear_interaction_features(self):
        """
        Add nonlinear interaction features to the dataset.

        Nonlinear interaction features capture nonlinear relationships between existing attributes,
        potentially revealing complex patterns not captured by linear interactions.

        Returns:
            None
        """
        self.data_analizer.data_train['trip_distance_squared'] = self.data_analizer.data_train['trip_distance'] ** 2
        self.data_analizer.data_train['log_trip_distance'] = np.log(self.data_analizer.data_train['trip_distance'].replace(0,1e-8))  # Log transform to handle skew
        self.data_analizer.data_train['exp_tip_per_mile'] = np.exp(self.data_analizer.data_train['tip_per_mile'].clip(-10,10))

        self.data_analizer.data_test['trip_distance_squared'] = self.data_analizer.data_test['trip_distance'] ** 2
        self.data_analizer.data_test['log_trip_distance'] = np.log(self.data_analizer.data_test['trip_distance'] + 1e-8)  # Log transform to handle skew
        self.data_analizer.data_test['exp_tip_per_mile'] = np.exp(self.data_analizer.data_test['tip_per_mile'])

        # Encoding cyclical time features
        self.data_analizer.data_train['pickup_hour_sin'] = np.sin(2 * np.pi * self.data_analizer.data_train['pickup_hour'] / 24)
        self.data_analizer.data_train['pickup_hour_cos'] = np.cos(2 * np.pi * self.data_analizer.data_train['pickup_hour'] / 24)
        self.data_analizer.data_train['pickup_day_sin'] = np.sin(2 * np.pi * self.data_analizer.data_train['pickup_day_of_week'] / 7)
        self.data_analizer.data_train['pickup_day_cos'] = np.cos(2 * np.pi * self.data_analizer.data_train['pickup_day_of_week'] / 7)

        self.data_analizer.data_test['pickup_hour_sin'] = np.sin(2 * np.pi * self.data_analizer.data_test['pickup_hour'] / 24)
        self.data_analizer.data_test['pickup_hour_cos'] = np.cos(2 * np.pi * self.data_analizer.data_test['pickup_hour'] / 24)
        self.data_analizer.data_test['pickup_day_sin'] = np.sin(2 * np.pi * self.data_analizer.data_test['pickup_day_of_week'] / 7)
        self.data_analizer.data_test['pickup_day_cos'] = np.cos(2 * np.pi * self.data_analizer.data_test['pickup_day_of_week'] / 7)

        self.data_analizer.data_train.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data_analizer.data_train.fillna(0, inplace=True)

        self.data_analizer.data_test.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data_analizer.data_test.fillna(0, inplace=True)

    # Metodo para calcular o tempo de duração da viagem
    def calculate_trip_duration(self):
        """Calculates the duration of the trip in minutes."""

        # Ensure the dataset has the necessary distance column
        if 'dropoff_time_in_seconds' and 'pickup_time_in_seconds' not in self.data_analizer.data_train.columns:
            raise ValueError("time in seconds collums are missing from the dataset.")

        # Compute trip duration in minutes
        self.data_analizer.data_train['trip_duration_min'] = (self.data_analizer.data_train['dropoff_time_in_seconds'] - self.data_analizer.data_train['pickup_time_in_seconds']) / 60
        self.data_analizer.data_test['trip_duration_min'] = (self.data_analizer.data_test['dropoff_time_in_seconds'] - self.data_analizer.data_test['pickup_time_in_seconds']) / 60

        print("Trip duration calculated successfully.")


    # Metodo para calcular a velocidade média da viagem
    def calculate_average_speed(self):
        """Calculates the average speed of the trip in miles/h."""

        # Ensure the dataset has the necessary distance column
        if 'trip_distance' not in self.data_analizer.data_train.columns:
            raise ValueError("Column 'trip_distance' is missing from the dataset.")

        # Avoid division by zero for extremely short trips
        self.data_analizer.data_train['average_speed_mph'] = self.data_analizer.data_train['trip_distance'] / (self.data_analizer.data_train['trip_duration_min'] / 60)
        self.data_analizer.data_train.fillna({'average_speed_mph': 0}, inplace=True)  # Replace NaN values with 0
        self.data_analizer.data_test['average_speed_mph'] = self.data_analizer.data_test['trip_distance'] / (self.data_analizer.data_test['trip_duration_min'] / 60)
        self.data_analizer.data_test.fillna({'average_speed_mph': 0}, inplace=True)  # Replace NaN values with 0

        print("Average speed calculated successfully.")