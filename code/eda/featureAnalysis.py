from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


class FeatureAnalysis:

    def __init__(self, data_analizer, target_column):

        self.data_analizer = data_analizer
        self.target_column = target_column
        self.numerical_features = self.data_analizer.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = self.data_analizer.select_dtypes(include=['object']).columns.tolist()
        if self.target_column in self.numerical_features:
            self.numerical_features.remove(self.target_column)
        if self.target_column in self.categorical_features:
            self.categorical_features.remove(self.target_column)

    def feature_importance(self):
        """Computes and plots feature importance using a Random Forest model."""
        X = self.data_analizer.drop(columns=[self.target_column])
        y = self.data_analizer[self.target_column]

        # Encode categorical features
        if self.categorical_features:
            X[self.categorical_features] = X[self.categorical_features].apply(LabelEncoder().fit_transform)

        # Use RandomForestRegressor if target is continuous
        if y.dtype in ["int64", "float64"]:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        model.fit(X, y)

        # Plot feature importances
        feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        feature_importance.plot(kind="bar", figsize=(12, 6), title="Feature Importance (Random Forest)")
        plt.show()

    def pca_analysis(self):

        X = self.data_analizer.drop(columns=[self.target_column])

        # Encode categorical features
        if self.categorical_features:
            X[self.categorical_features] = X[self.categorical_features].apply(LabelEncoder().fit_transform)

        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.data_analizer[self.target_column], cmap="viridis", alpha=0.5)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("PCA Projection")
        plt.colorbar()
        plt.show()

    def run_all(self):
        self.feature_importance()
        self.pca_analysis()
        print("Feature Analysis completed successfully.")