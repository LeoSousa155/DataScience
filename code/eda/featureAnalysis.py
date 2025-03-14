import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA


class FeatureAnalysis:
    """
    A class to analyze feature importance and perform PCA analysis on a dataset.

    This class works with an instance of DataAnalizer or its subclass to extract numerical
    and categorical features, compute feature importance using a Random Forest model,
    and perform PCA for dimensionality reduction.
    """

    def __init__(self, data_analizer):
        """
        Initializes the FeatureAnalysis instance.

        Args:
            data_analizer (DataAnalizer): An instance of DataAnalizer or its subclass.
        """
        self.data_analizer = data_analizer
        self.target_column = data_analizer.target

        # Identify numerical and categorical features
        self.numerical_features = self.data_analizer.data_train.select_dtypes(
            include=['int64', 'float64']).columns.tolist()
        self.categorical_features = self.data_analizer.data_train.select_dtypes(include=['object']).columns.tolist()

        # Ensure target column is not considered a feature
        if self.target_column in self.numerical_features:
            self.numerical_features.remove(self.target_column)
        if self.target_column in self.categorical_features:
            self.categorical_features.remove(self.target_column)

    def feature_importance(self):
        """
        Computes and plots feature importance using a Random Forest model.
        """
        X_train = self.data_analizer.data_train
        y_train = self.data_analizer.labels_train

        # Encode categorical features
        if self.categorical_features:
            X_train = X_train.copy()
            X_train[self.categorical_features] = X_train[self.categorical_features].apply(LabelEncoder().fit_transform)

        # Determine if target is continuous or categorical
        if y_train.dtype in ['int64', 'float64']:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        model.fit(X_train, y_train)

        # Plot feature importances
        feature_importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        feature_importance.plot(kind="bar", figsize=(12, 6), title="Feature Importance (Random Forest)")
        plt.show()

    def pca_analysis(self):
        """
        Performs PCA and visualizes the dataset in 2D space.
        """
        X_train = self.data_analizer.data_train
        y_train = self.data_analizer.labels_train

        # Encode categorical features
        if self.categorical_features:
            X_train = X_train.copy()
            X_train[self.categorical_features] = X_train[self.categorical_features].apply(LabelEncoder().fit_transform)

        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap="viridis", alpha=0.5)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("PCA Projection")
        plt.colorbar()
        plt.show()

    def run_all(self):
        """
        Runs both feature importance analysis and PCA visualization.
        """
        self.feature_importance()
        self.pca_analysis()
        print("Feature Analysis completed successfully.")
