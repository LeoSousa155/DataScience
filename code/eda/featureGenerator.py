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
        self.calculate_trip_duration()


    def add_statistical_features(self):
        """
        Add statistical features to the dataset.

        Statistical features provide insights into how much each sample's value is above or below the mean of the feature.

        Returns:
            None
        """
        pass


    def add_interaction_features(self):
        """
        Add interaction features to the dataset.

        Interaction features capture interactions between existing attributes, potentially revealing
        complex relationships not captured by individual features alone.

        Returns:
            None
        """
        pass


    def add_nonlinear_interaction_features(self):
        """
        Add nonlinear interaction features to the dataset.

        Nonlinear interaction features capture nonlinear relationships between existing attributes,
        potentially revealing complex patterns not captured by linear interactions.

        Returns:
            None
        """
        pass


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